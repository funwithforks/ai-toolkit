"""
Remote sampling backend: offloads training sample generation to a ComfyUI
instance reached through the Comfy API v2 proxy.

Enabled via the job config under `sample:`:
    comfy: true
    comfy_url: http://host:8189
    # optional: comfy_workflow_dir (defaults to docs/Comfy/workflows).
    # The workflow file is picked 1:1 by model arch: <arch>.json, e.g. a job
    # with arch "minimax_h3" samples with docs/Comfy/workflows/minimax_h3.json.
    # comfy_workflow_path overrides the lookup for one-off cases.

The workflow is an API-format graph whose LoRA slot and prompt are tagged
with placeholder strings ({{AITK_LORA}}, {{AITK_LORA_STRENGTH}},
{{AITK_PROMPT}}). Each sample round:
  1. snapshots the network (EMA-aware, multiplier 1.0) to a local safetensors
  2. uploads it to the proxy under models/loras/aitk/<job_name>/
  3. submits one job per sample item (only lora strength + prompt vary;
     the seed is static, owned by the workflow)
  4. polls, downloads outputs into {save_root}/samples/ with the standard
     [time]_{step}_[count].<ext> naming

Training never blocks on the remote. An ordered journal
({save_root}/comfy_samples.json) survives process death: undelivered rounds
stay queued in order and drain when the remote is reachable again.

User/agent guide: docs/guide/comfy-remote-sampling.md
"""
import copy
import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections import OrderedDict
from typing import Dict, List, Optional

import requests

try:
    import fcntl
except ImportError:  # windows: lock degrades to pid-file best effort
    fcntl = None

from toolkit.print import print_acc
from toolkit.basic import flush

TAG_LORA = '{{AITK_LORA}}'
TAG_LORA_STRENGTH = '{{AITK_LORA_STRENGTH}}'
TAG_PROMPT = '{{AITK_PROMPT}}'
ALL_TAGS = (TAG_LORA, TAG_LORA_STRENGTH, TAG_PROMPT)

RETRY_INTERVAL_SEC = 15.0
WAKE_INTERVAL_SEC = 5.0
# the detached collector lives until the journal drains; the ceiling matches
# the proxy's ~24h job/output expiry window (beyond that, pending jobs would
# expire anyway)
COLLECTOR_TIMEOUT_SEC = 86400.0
JOURNAL_FILENAME = 'comfy_samples.json'
SNAPSHOT_FOLDER = 'comfy_snapshots'
ROUND_TERMINAL = ('complete', 'failed')
SAVE_NODE_TYPES = {
    'SaveImage', 'SaveVideo', 'SaveAnimatedWEBP', 'SaveAnimatedPNG',
    'DecodeAndSaveVideo', 'VHS_VideoCombine',
}
JOB_TERMINAL_OK = 'succeeded'
JOB_TERMINAL_BAD = ('failed', 'canceled', 'cancelled', 'error')


# workflows are stored one per model arch, named <arch>.json
DEFAULT_WORKFLOW_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'docs', 'Comfy', 'workflows'))


class ComfyTransientError(Exception):
    """Remote unreachable / not ready: keep journal state and retry later."""


def resolve_comfy_workflow_path(sc, arch):
    """Map the job's model arch onto its workflow file."""
    if sc.comfy_workflow_path:
        return sc.comfy_workflow_path
    wf_dir = sc.comfy_workflow_dir or DEFAULT_WORKFLOW_DIR
    if not arch:
        raise ValueError(
            'comfy sampling: model arch is unknown, cannot pick a workflow; '
            'set sample.comfy_workflow_path explicitly')
    candidate = os.path.join(wf_dir, f'{arch}.json')
    if os.path.isfile(candidate):
        return candidate
    available = sorted(
        f[:-len('.json')] for f in os.listdir(wf_dir) if f.endswith('.json')
    ) if os.path.isdir(wf_dir) else []
    raise ValueError(
        f'comfy sampling: no workflow for arch "{arch}" in {wf_dir} '
        f'(available: {", ".join(available) if available else "none"}). '
        f'Add it as {arch}.json')


def _find_tags(workflow) -> set:
    found = set()

    def walk(obj):
        if isinstance(obj, dict):
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)
        elif isinstance(obj, str):
            for tag in ALL_TAGS:
                if tag in obj:
                    found.add(tag)

    walk(workflow)
    return found


def _deep_replace(obj, mapping: Dict[str, object]):
    if isinstance(obj, dict):
        return {k: _deep_replace(v, mapping) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_deep_replace(v, mapping) for v in obj]
    if isinstance(obj, str):
        if obj in mapping:
            return mapping[obj]
        for tag in ALL_TAGS:
            if tag in obj and tag in mapping:
                obj = obj.replace(tag, str(mapping[tag]))
        return obj
    return obj


def _strip_lora_nodes(workflow, node_id):
    """
    Remove the LoRA node(s) containing the lora tag and rewire every
    consumer onto the model that fed the removed node — mechanically the
    same graph as exporting the workflow with that node bypassed in the UI.
    """
    def rewire(obj):
        if isinstance(obj, dict):
            return {k: rewire(v) for k, v in obj.items()}
        if isinstance(obj, list):
            if len(obj) == 2 and obj[0] == node_id and isinstance(obj[1], int):
                return list(upstream)
            return [rewire(v) for v in obj]
        return obj

    node = workflow.pop(node_id)
    upstream = node.get('inputs', {}).get('model')
    if not isinstance(upstream, list) or len(upstream) != 2:
        raise ValueError(
            'comfy sampling: tagged lora node has no model input link, '
            'cannot be bypassed for the first sample')
    return {k: rewire(v) for k, v in workflow.items()}


def _lora_node_ids(workflow):
    return sorted(
        nid for nid, node in workflow.items()
        if isinstance(node, dict)
        and any(isinstance(v, str) and TAG_LORA in v
                for v in node.get('inputs', {}).values()))


def _bypassed_workflow(template):
    wf = copy.deepcopy(template)
    for nid in _lora_node_ids(wf):
        wf = _strip_lora_nodes(wf, nid)
    return wf


def _set_save_prefixes(workflow, prefix: str):
    # per-round unique output prefix so parallel jobs never overwrite
    # each other's files inside the remote output folder
    for node in workflow.values():
        if isinstance(node, dict) and node.get('class_type') in SAVE_NODE_TYPES:
            inputs = node.get('inputs')
            if isinstance(inputs, dict) and isinstance(inputs.get('filename_prefix'), str):
                inputs['filename_prefix'] = prefix


def _check_workflow_file(path: str, errors: list):
    if not os.path.isfile(path):
        errors.append(f'comfy workflow not found: {path}')
        return
    try:
        with open(path, 'r') as f:
            wf = json.load(f)
    except Exception as e:
        errors.append(f'comfy workflow is not valid JSON: {path} ({e})')
        return
    if not isinstance(wf, dict) or 'nodes' in wf:
        errors.append(
            f'comfy workflow must be API-format JSON (Save (API Format) from '
            f'the ComfyUI UI), not the UI save format: {path}')
        return
    missing = set(ALL_TAGS) - _find_tags(wf)
    if missing:
        errors.append(
            f'comfy workflow missing placeholder tags {sorted(missing)}: {path}')


def validate_comfy_sample_setup(process):
    """
    Fail loudly at job start when comfy sampling cannot do its job.
    LoRA-family networks only: the remote consumes a snapshot file, so
    fine-tunes, embeddings, adapters and decorators have nothing to ship.
    """
    # config objects only: this runs in __init__ before the network,
    # embedding and adapter instances exist
    sc = process.sample_config
    if not sc.comfy:
        sc = process.first_sample_config
    errors = []
    if not sc.comfy_url:
        errors.append('sample.comfy_url is required when sample.comfy is true')
    # arch is not resolved yet at init time, so the per-arch workflow file
    # only gets checked when the sampler starts (at first sample round). An
    # explicit path or dir can still be checked now.
    if sc.comfy_workflow_path:
        _check_workflow_file(sc.comfy_workflow_path, errors)
    else:
        wf_dir = sc.comfy_workflow_dir or DEFAULT_WORKFLOW_DIR
        if not os.path.isdir(wf_dir):
            errors.append(f'comfy workflow folder not found: {wf_dir}')

    if process.network_config is None:
        errors.append(
            'sample.comfy needs a LoRA-family network: the sampler ships a '
            'snapshot of it to ComfyUI along with the workflow')
    if errors:
        raise ValueError(
            'comfy sampling is not usable for this job:\n  - ' + '\n  - '.join(errors))


def _read_lock_pid(fh) -> Optional[int]:
    try:
        fh.seek(0)
        txt = fh.read().strip()
        return int(txt) if txt else None
    except (OSError, ValueError):
        return None


def _signal_pid(pid: int, sig):
    if pid == os.getpid():
        return
    proc_cmd = f'/proc/{pid}/cmdline'
    if os.path.exists(proc_cmd):
        # only signal processes that are actually ours (pid may have been reused)
        try:
            with open(proc_cmd, 'rb') as fh:
                cmdline = fh.read().decode('utf-8', errors='ignore')
        except OSError:
            return
        if 'comfy_sampler' not in cmdline:
            return
    try:
        os.kill(pid, sig)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def get_comfy_sampler(process):
    sampler = getattr(process, '_comfy_sampler', None)
    if sampler is None:
        sampler = ComfySampler(process)
        process._comfy_sampler = sampler
    return sampler


class ComfySampler:
    def __init__(self, process):
        self.process = process
        sc = process.sample_config
        if not sc.comfy:
            sc = process.first_sample_config
        self.base_url = sc.comfy_url.rstrip('/')
        sd = getattr(process, 'sd', None)
        arch = getattr(sd, 'arch', None) or getattr(
            getattr(process, 'model_config', None), 'arch', None)
        self.workflow_path = resolve_comfy_workflow_path(sc, arch)
        self.job_name = str(process.job.name)
        self.save_root = process.save_root
        self._init_common(take_over=True)

    @classmethod
    def for_drain(cls, journal_path: str, lock_wait: float = 60.0):
        """
        Headless mode used by the detached collector child process: rebuilds
        everything needed to finish rounds from the journal alone. Returns
        None when another process (a resumed trainer) holds the journal.
        """
        with open(journal_path, 'r') as f:
            data = json.load(f)
        url = data.get('comfy_url')
        wf_path = data.get('comfy_workflow_path')
        if not url or not wf_path:
            raise ValueError(
                f'{journal_path} predates detached collection and cannot be '
                'resumed stand-alone; run the job once to migrate it')
        obj = cls.__new__(cls)
        obj.process = None
        obj.base_url = url.rstrip('/')
        obj.workflow_path = wf_path
        obj.job_name = data.get('job_name') or os.path.basename(
            os.path.dirname(journal_path))
        obj.save_root = os.path.dirname(journal_path)
        # the spawning trainer still holds the lock for a moment after the
        # handoff; wait for it to be released by process exit (a resumed
        # job that grabs it first means we are not needed and we exit)
        if not obj._init_common(take_over=False, lock_wait=lock_wait):
            return None
        return obj

    def _init_common(self, take_over: bool, lock_wait: float = 0.0) -> bool:
        errors = []
        _check_workflow_file(self.workflow_path, errors)
        if errors:
            raise ValueError(errors[0])
        print_acc(f'[comfy] workflow: {self.workflow_path}')
        with open(self.workflow_path, 'r') as f:
            self.workflow_template = json.load(f)
        # the pre-training baseline round needs the tagged lora node to be
        # bypassable (model input link present); fail before training, not
        # at first sample
        try:
            _bypassed_workflow(self.workflow_template)
        except ValueError as e:
            raise ValueError(f'comfy sampling: {e}')
        self.journal_path = os.path.join(self.save_root, JOURNAL_FILENAME)
        self.snapshot_folder = os.path.join(self.save_root, SNAPSHOT_FOLDER)
        self._lock = threading.Lock()
        self._dirty = threading.Event()
        if not self._acquire_journal_lock(take_over, lock_wait):
            return False
        self._rounds: List[dict] = self._load_journal()
        resumed = [r for r in self._rounds if r['status'] not in ROUND_TERMINAL]
        if resumed:
            print_acc(
                f'[comfy] resuming {len(resumed)} unfinished sample round(s) '
                f'from {self.journal_path}')
        self._session = requests.Session()
        self._worker = threading.Thread(
            target=self._worker_loop, name='comfy-sampler', daemon=True)
        self._worker.start()
        return True

    # ------------------------------------------------------------------ journal lock
    # One writer per journal. A resumed trainer (take_over=True) terminates a
    # lingering detached collector to reclaim it; a collector (take_over=False)
    # simply exits when the journal is already owned.
    def _acquire_journal_lock(self, take_over: bool,
                              lock_wait: float = 0.0) -> bool:
        lock_path = self.journal_path + '.lock'
        os.makedirs(os.path.dirname(lock_path) or '.', exist_ok=True)
        fh = open(lock_path, 'a+')
        if fcntl is not None:
            acquired = False
            patience = 80 if take_over else max(1, int(lock_wait / 0.25))
            for attempt in range(patience):  # take_over: ~20s; wait: lock_wait
                try:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                    break
                except OSError:
                    if not take_over:
                        time.sleep(0.25)
                        continue
                    holder = _read_lock_pid(fh)
                    if attempt == 0 and holder:
                        _signal_pid(holder, signal.SIGTERM)
                    elif attempt == 60 and holder:
                        _signal_pid(holder, signal.SIGKILL)
                    time.sleep(0.25)
            if not acquired:
                if not take_over:
                    # the job resumed (or is still running); it owns the journal
                    fh.close()
                    return False
                print_acc('[comfy] journal lock never released; continuing '
                          'without it (another writer may be active)')
        else:
            holder = _read_lock_pid(fh)
            if take_over and holder:
                _signal_pid(holder, signal.SIGTERM)
                time.sleep(1.0)
        try:
            fh.seek(0)
            fh.truncate()
            fh.write(str(os.getpid()))
            fh.flush()
        except OSError:
            pass
        self._lock_fh = fh
        return True

    # ----------------------------------------------------------------- journal
    def _load_journal(self) -> List[dict]:
        if not os.path.isfile(self.journal_path):
            return []
        try:
            with open(self.journal_path, 'r') as f:
                data = json.load(f)
            return data.get('rounds', [])
        except Exception as e:
            print_acc(f'[comfy] journal {self.journal_path} unreadable ({e}); '
                      'starting fresh')
            return []

    def _save_journal(self):
        os.makedirs(self.save_root, exist_ok=True)
        tmp = self.journal_path + '.tmp'
        # self-describing payload: a detached collector can finish rounds
        # from this file alone, without the training config
        payload = {
            'version': 2,
            'job_name': self.job_name,
            'comfy_url': self.base_url,
            'comfy_workflow_path': os.path.abspath(self.workflow_path),
            'rounds': self._rounds,
        }
        with open(tmp, 'w') as f:
            json.dump(payload, f, indent=1)
        os.replace(tmp, self.journal_path)

    # -------------------------------------------------------------- submission
    def submit_round(self, image_configs: List, step: Optional[int],
                     is_first: bool = False):
        """
        Called from the main thread where EMA weights (if any) are already
        applied. The weight snapshot is written synchronously (cheap, and it
        pins the exact weights this round is about); upload, submit and
        collect happen on the worker so training continues immediately.

        is_first rounds are the pre-training baseline: nothing has been
        trained yet, so no lora is shipped at all — the submitted workflow
        has the tagged lora node bypassed (graph-rewired around it) and
        renders the base model.
        """
        round_id = f"{(step if step is not None else 0):09d}_{int(time.time() * 1000)}"
        snapshot_path = None
        if not is_first:
            os.makedirs(self.snapshot_folder, exist_ok=True)
            snapshot_path = os.path.join(
                self.snapshot_folder, f'{round_id}.safetensors')
            self._write_snapshot(snapshot_path, step)

        max_count = max(len(image_configs) - 1, 0)
        items = []
        for idx, cfg in enumerate(image_configs):
            target = cfg.get_image_path(idx, max_count)
            items.append({
                'idx': idx,
                'prompt': cfg.prompt,
                'strength': float(cfg.network_multiplier),
                'output_base': os.path.splitext(target)[0],
                'status': 'waiting_submit',
                'job_id': None,
                'error': None,
            })
        round_rec = {
            'id': round_id,
            'step': step,
            'status': 'queued',
            'baseline': bool(is_first),
            'snapshot': snapshot_path,
            # placement name is decided now (not at upload time) so a
            # detached collector can resubmit items from the journal alone
            'remote_lora': None if is_first else
            f'aitk/{self.job_name}/{self.job_name}_{round_id}.safetensors',
            'asset_id': None,
            'items': items,
        }
        with self._lock:
            self._rounds.append(round_rec)
            self._save_journal()
        self._dirty.set()
        print_acc(f'[comfy] sample round {round_id} queued '
                  f'({len(items)} job(s)); training continues')

    def _write_snapshot(self, path: str, step):
        from toolkit.train_tools import get_torch_dtype
        net = self.process.network
        prev_multiplier = net.multiplier
        net.multiplier = 1.0
        try:
            meta = OrderedDict({
                'ss_output_name': self.job_name,
                'aitk_sample_snapshot_step': str(step if step is not None else 0),
            })
            net.save_weights(
                path,
                dtype=get_torch_dtype(self.process.save_config.dtype),
                metadata=meta,
            )
        finally:
            net.multiplier = prev_multiplier
        flush()

    # -------------------------------------------------------------------- http
    def _upload_lora(self, round_rec: dict):
        remote_rel = round_rec.get('remote_lora') or \
            f"aitk/{self.job_name}/{self.job_name}_{round_rec['id']}.safetensors"
        round_rec['remote_lora'] = remote_rel
        file_name = os.path.basename(remote_rel)
        file_path = f'models/loras/{remote_rel}'
        try:
            with open(round_rec['snapshot'], 'rb') as fh:
                resp = self._session.post(
                    f'{self.base_url}/api/v2/assets',
                    data={
                        'content_type': 'application/octet-stream',
                        'file_path': file_path,
                    },
                    files={'file': (file_name, fh, 'application/octet-stream')},
                    timeout=(10, 600),
                )
        except requests.RequestException as e:
            raise ComfyTransientError(f'asset upload failed: {e}')
        if resp.status_code == 422 and 'refusing to overwrite' in resp.text:
            # same path already placed by a previous run; reuse it
            round_rec['remote_lora'] = file_path[len('models/loras/'):]
            return
        if resp.status_code not in (200, 201):
            raise ComfyTransientError(
                f'asset upload HTTP {resp.status_code}: {resp.text[:300]}')
        data = resp.json()
        round_rec['asset_id'] = data.get('id')
        round_rec['remote_lora'] = data.get('file_path', file_path[len('models/loras/'):])

    def _submit_job(self, round_rec: dict, item: dict) -> str:
        template = self.workflow_template
        mapping = {TAG_PROMPT: item['prompt']}
        if round_rec.get('baseline'):
            template = _bypassed_workflow(template)
        else:
            mapping[TAG_LORA] = round_rec['remote_lora']
            mapping[TAG_LORA_STRENGTH] = item['strength']
        workflow = _deep_replace(template, mapping)
        _set_save_prefixes(
            workflow, f'aitk/{self.job_name}/{round_rec["id"]}/{item["idx"]}')
        try:
            resp = self._session.post(
                f'{self.base_url}/api/v2/jobs',
                json={
                    'workflow': workflow,
                    'metadata': f'aitk-trainer {self.job_name} {round_rec["id"]}',
                },
                timeout=(10, 60),
            )
        except requests.RequestException as e:
            raise ComfyTransientError(f'job submit failed: {e}')
        if resp.status_code not in (200, 201):
            raise ComfyTransientError(
                f'job submit HTTP {resp.status_code}: {resp.text[:300]}')
        return resp.json()['id']

    def _job_state(self, job_id: str):
        try:
            resp = self._session.get(
                f'{self.base_url}/api/v2/jobs/{job_id}', timeout=(10, 30))
        except requests.RequestException as e:
            raise ComfyTransientError(f'job poll failed: {e}')
        if resp.status_code == 404:
            return 'expired', None
        if resp.status_code != 200:
            raise ComfyTransientError(
                f'job poll HTTP {resp.status_code}: {resp.text[:200]}')
        data = resp.json()
        return data.get('status'), data

    def _collect_output(self, data: dict, output_base: str) -> int:
        outputs = data.get('outputs') or []
        if not outputs:
            raise ComfyTransientError('job succeeded with no outputs')
        written = 0
        for oi, out in enumerate(outputs):
            url = out.get('url')
            if not url:
                continue
            ext = os.path.splitext(out.get('name') or '')[1] or '.png'
            dest = output_base + (ext if oi == 0 else f'_{oi + 1}{ext}')
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            try:
                with self._session.get(url, timeout=(10, 600), stream=True) as r:
                    if r.status_code != 200:
                        raise ComfyTransientError(
                            f'output download HTTP {r.status_code}')
                    tmp = dest + '.part'
                    with open(tmp, 'wb') as fh:
                        for chunk in r.iter_content(chunk_size=1 << 20):
                            fh.write(chunk)
                    os.replace(tmp, dest)
                written += 1
            except requests.RequestException as e:
                raise ComfyTransientError(f'output download failed: {e}')
        if not written:
            raise ComfyTransientError('job outputs had no downloadable urls')
        return written

    def _delete_asset(self, asset_id: str):
        try:
            self._session.delete(
                f'{self.base_url}/api/v2/assets/{asset_id}', timeout=(10, 30))
        except requests.RequestException:
            pass

    # ------------------------------------------------------------------ worker
    def _worker_loop(self):
        while True:
            self._dirty.wait(timeout=WAKE_INTERVAL_SEC)
            self._dirty.clear()
            try:
                self._process_pass()
            except ComfyTransientError as e:
                self._dirty.wait(timeout=RETRY_INTERVAL_SEC)
            except Exception as e:
                print_acc(f'[comfy] sampler worker error (will retry): {e}')
                self._dirty.wait(timeout=RETRY_INTERVAL_SEC)

    def _process_pass(self):
        with self._lock:
            pending = [r for r in self._rounds if r['status'] not in ROUND_TERMINAL]
        for round_rec in pending:
            self._advance_round(round_rec)
            with self._lock:
                self._save_journal()
        self._prune_journal()

    def _advance_round(self, round_rec: dict):
        status = round_rec['status']
        if status == 'queued':
            if round_rec.get('baseline'):
                # pre-training baseline: no lora to ship, the tagged node is
                # bypassed at submit time
                round_rec['status'] = 'uploaded'
                return
            if not round_rec['snapshot'] or not os.path.isfile(round_rec['snapshot']):
                self._fail_round(round_rec, 'weight snapshot missing on disk')
                return
            self._upload_lora(round_rec)
            round_rec['status'] = 'uploaded'
            return
        if status in ('uploaded', 'submitted'):
            for item in round_rec['items']:
                if item['status'] == 'waiting_submit':
                    item['job_id'] = self._submit_job(round_rec, item)
                    item['status'] = 'submitted'
                    round_rec['status'] = 'submitted'
            self._poll_items(round_rec)
            return
        if status == 'collecting':
            self._poll_items(round_rec)

    def _poll_items(self, round_rec: dict):
        """
        Advance every submitted item. Returns with status 'collecting' when
        some jobs are still running on the remote (the worker re-checks them
        on the next pass; other rounds keep being submitted in the meantime).
        A ComfyTransientError here means the remote itself is unreachable,
        which aborts the whole pass.
        """
        still_waiting = False
        for item in round_rec['items']:
            if item['status'] != 'submitted':
                continue
            job_status, data = self._job_state(item['job_id'])
            if job_status == JOB_TERMINAL_OK:
                try:
                    self._collect_output(data, item['output_base'])
                    item['status'] = 'collected'
                    print_acc(f"[comfy] sample collected -> "
                              f"{os.path.basename(item['output_base'])}")
                except ComfyTransientError as e:
                    # succeeded but outputs not fetchable yet; retry next pass
                    still_waiting = True
                    item['error'] = str(e)[:300]
            elif job_status in JOB_TERMINAL_BAD or job_status == 'expired':
                err = None
                if data and data.get('error'):
                    err = json.dumps(data['error'])[:300]
                item['status'] = 'failed'
                item['error'] = err or f'job {job_status}'
                print_acc(f"[comfy] sample job failed "
                          f"({item['error']}); continuing")
            else:
                still_waiting = True
        if still_waiting:
            round_rec['status'] = 'collecting'
            return
        # all items terminal -> cleanup round
        if round_rec.get('asset_id'):
            self._delete_asset(round_rec['asset_id'])
            round_rec['asset_id'] = None
        snap = round_rec.get('snapshot')
        if snap and os.path.isfile(snap):
            try:
                os.remove(snap)
            except OSError:
                pass
        failed = [i for i in round_rec['items'] if i['status'] == 'failed']
        round_rec['status'] = 'failed' if failed and len(failed) == len(
            round_rec['items']) else 'complete'

    def _fail_round(self, round_rec: dict, reason: str):
        print_acc(f"[comfy] sample round {round_rec['id']} failed: {reason}")
        for item in round_rec['items']:
            if item['status'] not in ('collected', 'failed'):
                item['status'] = 'failed'
                item['error'] = reason
        round_rec['status'] = 'failed'

    def _prune_journal(self):
        # keep the file small: only the 50 most recent rounds
        with self._lock:
            if len(self._rounds) > 50:
                self._rounds = self._rounds[-50:]
                self._save_journal()

    # ---------------------------------------------------------------- lifecycle
    def pending_rounds(self) -> List[dict]:
        with self._lock:
            return [r for r in self._rounds if r['status'] not in ROUND_TERMINAL]

    def spawn_collector(self):
        """
        Called when the training process is ending (normally or via stop):
        hand the unfinished rounds to a detached child process so exiting
        never waits on the remote box. The child drains the journal into the
        job's samples folder and exits on its own; a resumed job reclaims the
        journal by taking over the lock.
        """
        if not self.pending_rounds():
            return
        if getattr(self, '_lock_fh', None) is None:
            return
        repo_root = os.path.abspath(os.path.join(
            os.path.dirname(__file__), os.pardir))
        log_path = os.path.join(self.save_root, 'comfy_collector.log')
        try:
            with open(log_path, 'a') as fh:
                subprocess.Popen(
                    [sys.executable, '-m', 'toolkit.comfy_sampler',
                     'drain', self.journal_path],
                    cwd=repo_root,
                    stdin=subprocess.DEVNULL,
                    stdout=fh,
                    stderr=fh,
                    start_new_session=True,  # survive the training process
                )
            print_acc('[comfy] detached collector spawned; remaining samples '
                      f'will land in {os.path.join(self.save_root, "samples")} '
                      'after this job exits')
        except OSError as e:
            print_acc(f'[comfy] collector spawn failed ({e}); rounds stay '
                      f'queued in {self.journal_path} for the next run')


def _drain_cli(argv) -> int:
    import argparse
    parser = argparse.ArgumentParser(
        prog='python -m toolkit.comfy_sampler',
        description='finish pending comfy sample rounds from a job journal')
    parser.add_argument('command', choices=['drain'])
    parser.add_argument('journal', help='path to comfy_samples.json')
    parser.add_argument('--timeout', type=float,
                        default=COLLECTOR_TIMEOUT_SEC)
    args = parser.parse_args(argv)

    sampler = ComfySampler.for_drain(args.journal)
    if sampler is None:
        print('[comfy] journal is owned by a running job; collector exiting',
              flush=True)
        return 0
    print(f'[comfy] collector started for {args.journal}', flush=True)
    deadline = time.time() + args.timeout
    while time.time() < deadline and sampler.pending_rounds():
        time.sleep(WAKE_INTERVAL_SEC)
    left = len(sampler.pending_rounds())
    print(f'[comfy] collector exiting; {left} round(s) still pending',
          flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(_drain_cli(sys.argv[1:]))
