"""h3_vae_fixtures_runner: VAE-only phase-1 harness (the h3_vae_fixtures
suite's inner job; probe-style spawn, no run.py, no TE, no transformer).

mode=encode rides the REAL machinery (verified anchors):
  ModelConfig/get_model_class like BaseSDTrainProcess.py:2044-2052,
  build_datasets(defer_caching=True) (toolkit/data_loader.py:759),
  model.load_vae() (minimax_h3.py:791 -> video+audio bundle),
  dataset.cache_latents_all_latents() (the process's own entry,
  BaseSDTrainProcess.py:2159-2161). Nothing bypassed: same FileItemDTO
  scan/bucket/hash path that writes production caches.

Discipline this file exists to enforce (each one is a past failure mode):
  * COLD-LOUD start: every pre-existing cache file for the dataset is
    deleted first, counted on stdout. The caching pass SKIPS existing
    files, so a warm start would silently grade the previous arm's bytes
    (the filename hash covers dataset facts, NOT vae precision -- arms
    collide on identical names by design).
  * PINNED DRAW STREAM: the cached video latent IS a posterior sample
    from the harness's global RNG (production encode_images passes no
    generator). Unseeded, two cold runs of one dataset cannot match
    (measured 1.14 maxdiff); seeded, a byte diff means a pipeline
    difference. The pin prints its seed (fixt_rng_pinned).
  * ONE encode pass per run: byte-identity across arms is compare mode's
    job across two submissions. (An in-process double-encode check was
    built, found to grade its own copies, and deleted -- checks that
    cannot fail are assumptions with a number attached.)
  * frame parity is arithmetic from the manifest's own numbers (17n+5 ->
    5n+2 latent T; stills T=1; packed audio rows even and 32 wide), not
    "geometry exists". Warm-read fidelity is NOT graded here: a reader
    check inside the process that just encoded grades that process's own
    state; it belongs to real training under the vae_skip law.
  * decode artifacts are mandatory (owner gate is the media-player
    watch): per fixture, ORIGINAL | ENCODE->DECODE concat under
    artifacts/concat/, roundtrip dB recorded as a number beside the
    watch (audio compared only after resampling the source to 32 kHz --
    native-rate-vs-32k differencing measures the harness), never as the
    verdict.
  * the cache files this arm produced are copied into the record
    (artifacts/cache/) + fingerprinted (cache_manifest.json): records are
    what mode=compare reads -- records are the database.

mode=compare pairs two encode records' cache files by name and grades
stage A (byte-identity) or stage B (declared-diff under the owner's
ceiling). Shape/dtype/finiteness/missing-pair mismatches are ALWAYS
unexplained -- precision changes never excuse geometry changes.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys


# ------------------------------------------------------------------ helpers

def _sha(b):
    return hashlib.sha256(b).hexdigest()


def _header_dict(path):
    """Parsed safetensors header. Dict equality is order-insensitive on
    purpose: the serialized key ORDER of __metadata__ is per-process
    hashed (measured: seeded arms produce dict-equal headers with 366
    permuted bytes), so raw header bytes cannot carry the identity law --
    its values can."""
    import struct
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def _bits(t):
    """Byte view of a tensor: bit-equality, where identical NaN bit
    patterns DO count equal (finiteness is its own named check, not an
    identity question)."""
    import torch
    t = t.detach().cpu().contiguous()
    if t.dim() == 0:
        t = t.reshape(1)  # 0-dim scalars (e.g. num_frames) carry no
        # element-size to view from; one element is still their bits
    return t.view(torch.uint8)


def _plain(val):
    # get_latent may hand back a DTO-wrapped value; the payload is the
    # first element -- unwrap without importing private shapes
    while isinstance(val, (tuple, list)) and val:
        val = val[0]
    return val


def _load(path):
    from safetensors.torch import load_file
    return load_file(path, device="cpu")


def _load_with_meta(path):
    from safetensors import safe_open
    tensors, meta = {}, {}
    with safe_open(path, framework="pt", device="cpu") as f:
        meta = f.metadata() or {}
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    return tensors, meta


def _rm_caches(datasets):
    """Cold-LOUD: the caching pass SKIPS existing files, so a warm start
    would silently grade whatever the previous arm left behind (and the
    filename hash does not cover vae precision -- arms collide on
    identical names by design). Deletion is the measurement's
    precondition; it is counted, never silent."""
    n = 0
    for ds in datasets:
        for item in ds.file_list:
            p = item.get_latent_path(recalculate=True)
            if os.path.exists(p):
                os.remove(p)
                n += 1
    print(f"fixt cold reset: deleted={n}", flush=True)
    return n


# the harness's pinned draw stream (see _encode: cached latents ARE a
# posterior sample; unseeded there, a fixture can never reproduce itself)
FIXT_DRAW_SEED = 20261029


def _db_float(orig, dec):
    """roundtrip dB of the decode path in float, pre-codec: full-scale 2.0
    (range [-1,1]) over rms error. inf when bit-exact (reported as 999)."""
    import torch
    o, d = orig.float(), dec.float()
    t = min(o.shape[2], d.shape[2])
    o, d = o[:, :, :t], d[:, :, :t]
    rms = (o - d).pow(2).mean().sqrt().item()
    return 999.0 if rms <= 0 else round(20.0 * __import__("math").log10(2.0 / rms), 1)


def _db_audio(orig, dec):
    """Shape-fidelity SNR of the audio decode path, gain-fitted:
    err is minimized over a single scalar gain (least-squares fit of
    orig ~= k*dec). Without that, a loudness normalization between the
    raw file and the pipeline waveform (dataset audio_normalize) shows up
    as catastrophic damage -- the early -3.8 dB was exactly that
    artifact, scale mismatch not corruption. Loudness errors are the
    owner's listen on the concat, not this number; this number is: does
    the waveform SHAPE survive encode->decode."""
    import math
    import torch
    o = orig.float().mean(0) if orig.dim() == 2 else orig.float().flatten()
    d = dec.float().mean(0) if dec.dim() == 2 else dec.float().flatten()
    n = min(o.numel(), d.numel())
    o, d = o[:n], d[:n]
    sig = o.pow(2).mean().sqrt().item()
    if sig <= 0:
        return None  # silence: SNR undefined, not a finding
    k = torch.dot(o, d) / torch.dot(d, d).clamp_min(1e-12)
    err = (o - k * d).pow(2).mean().sqrt().item()
    if err <= 0:
        return 999.0
    return round(20.0 * math.log10(sig / err), 1)


def _write_video(path, frames_u8, fps, audio=None, audio_sr=32000):
    """frames (T,H,W,C) uint8 (+ optional (C,L) float waveform) -> mp4.
    ffmpeg rawvideo pipe: this torchvision build has no write_video, and
    ffmpeg is already required for the concat artifact."""
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{frames_u8.shape[2]}x{frames_u8.shape[1]}",
           "-r", str(fps), "-i", "-"]
    wav_tmp = None
    if audio is not None:
        import torchaudio
        wav_tmp = path + ".audio.wav"
        torchaudio.save(wav_tmp, audio.float(), audio_sr)
        cmd += ["-i", wav_tmp, "-c:a", "aac", "-shortest"]
    cmd += ["-c:v", "libx264", "-crf", "10", "-pix_fmt", "yuv420p", path]
    r = subprocess.run(cmd, input=frames_u8.numpy().tobytes(),
                       capture_output=True)
    if wav_tmp and os.path.exists(wav_tmp):
        os.remove(wav_tmp)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg write failed: {r.stderr[-300:].decode(errors='replace')}")


def _concat(orig_path, decoded_path, out_path, has_audio, w, h, fps):
    """ORIGINAL | ENCODE->DECODE side-by-side in one viewable file.
    filter_complex normalizes geometry/fps (bucketed decode vs. native
    original); near-lossless h264 for the copy so the watch is not
    degraded by the container it ships in."""
    lst = [orig_path, decoded_path]
    cmd = ["ffmpeg", "-y", "-loglevel", "error"]
    for p in lst:
        cmd += ["-i", p]
    a = ":a" if has_audio else ""
    fc = (f"[0:v]scale={w}:{h}:force_original_aspect_ratio=decrease,"
          f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,fps={fps}[v0];"
          f"[1:v]scale={w}:{h},fps={fps}[v1];"
          f"[v0][0{a}][v1][1{a}]concat=n=2:v=1:a={1 if has_audio else 0}[outv][outa]"
          if has_audio else
          f"[0:v]scale={w}:{h}:force_original_aspect_ratio=decrease,"
          f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,fps={fps}[v0];"
          f"[1:v]scale={w}:{h},fps={fps}[v1];"
          f"[v0][v1]concat=n=2:v=1:a=0[outv]")
    cmd += ["-filter_complex", fc, "-map", "[outv]"]
    if has_audio:
        cmd += ["-map", "[outa]", "-c:a", "aac"]
    cmd += ["-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", out_path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg concat failed: {r.stderr[-400:]}")


# ------------------------------------------------------------------- encode

def _encode(args):
    import torch
    import yaml
    # the runner is spawned as a script file; the repo root (cwd) is not on
    # the script path automatically -- toolkit imports need it (same
    # treatment the probe runner gives run.main)
    sys.path.insert(0, os.getcwd())
    from toolkit.accelerator import get_accelerator
    from toolkit.config_modules import ModelConfig
    from toolkit.util.get_model import get_model_class

    if args.arm != "old_path":
        print(f"arm not implemented: {args.arm!r} -- no vae_id registry in "
              f"the tree yet (single-load-path door is Stage A/B work). "
              f"Failing loudly per spec; old_path was NOT substituted.",
              flush=True)
        return 1

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    proc = cfg["config"]["process"][0]
    train = proc["train"]

    import extensions_built_in.diffusion_models  # registers AI_TOOLKIT_MODELS
    mc = ModelConfig(**proc["model"])
    ModelClass = get_model_class(mc)
    # dtype REQUIRED from config, no default: a defaulted dtype would make
    # this arm's bytes differ from a real training run for a reason with
    # no name. The string passes through exactly like
    # BaseSDTrainProcess.py:2050 (BaseModel converts via get_torch_dtype,
    # base_model.py:124) -- no local re-interpretation.
    if "dtype" not in train:
        print("fixt encode refused: base template sets no train.dtype; "
              "refusing to default -- encode dtype must equal what real "
              "training would use", flush=True)
        return 1
    dtype_str = str(train["dtype"])
    device = get_accelerator().device
    model = ModelClass(device=device, model_config=mc, dtype=dtype_str)
    print(f"latent_space_version={model.get_latent_space_version()}", flush=True)

    # dataset dicts verbatim from the merged config (resolution/frames/fps/
    # do_audio/do_i2v all feed the cache key); only the cache flags are
    # forced, loudly: this harness IS the caching pass. cache_latents stays
    # False: this is a disk-cache measurement, and with it True the DTOs
    # also hold a memory layer the record copy could silently disagree
    # with (a memory cache is still a cache -- keep it out of the frame).
    raw = [dict(d) for d in proc["datasets"]]
    for d in raw:
        d["cache_latents"] = False
        d["cache_latents_to_disk"] = True
        d["cache_text_embeddings"] = False
    from toolkit.data_loader import build_datasets
    datasets, _ = build_datasets(raw, batch_size=int(train.get("batch_size", 1)),
                                 sd=model, defer_caching=True)

    model.load_vae()  # real door: video+audio bundle, frozen (minimax_h3.py:791)
    _rm_caches(datasets)

    # PIN THE DRAW STREAM. The cached video bytes ARE a posterior sample:
    # encode() draws z = mean + std*eps from the GLOBAL RNG (src/vae.py:831;
    # production encode_images passes no generator, minimax_h3.py:914 --
    # measured: maxdiff 0.56 between two encodes of one tensor in one
    # process, 1.14 between cold runs of one dataset; loader bit-stable,
    # audio rows bit-stable -- no sampler there). An unseeded fixture
    # grades the RNG, never the pipeline. Pinning the harness's own stream
    # is the same mechanism the model itself uses for the released
    # keyframe recipe (KEYFRAME_ENCODE_SEED, minimax_h3.py:930): the
    # record's bytes become one reproducible draw -- both arms draw the
    # same eps, so a byte diff after pinning means a real pipeline
    # difference, which is exactly what stage A exists to see. Production
    # stays unseeded: sampler-per-cache-pass is production's design.
    torch.manual_seed(FIXT_DRAW_SEED)
    print(f"fixt rng pinned: seed={FIXT_DRAW_SEED}", flush=True)

    # -- ONE encode pass per run (identity across arms is compare mode's
    # job across two submissions; an in-process second encode is not part
    # of the protocol) ------------------------------------------------------
    for ds in datasets:
        if ds.is_caching_latents:
            ds.cache_latents_all_latents()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print("fixt encode pass complete", flush=True)

    cache_dir = os.path.join(args.artifacts, "cache")
    manifest = {}
    for i, ds in enumerate(datasets):
        for item in ds.file_list:
            p = item.get_latent_path(recalculate=True)
            if not os.path.exists(p):
                print(f"fixt unexplained diff: no cache file produced for "
                      f"{os.path.basename(item.path)}", flush=True)
                continue
            blob = open(p, "rb").read()
            dstdir = os.path.join(cache_dir, f"ds{i}")
            os.makedirs(dstdir, exist_ok=True)
            dst = os.path.join(dstdir, os.path.basename(p))
            with open(dst, "wb") as f:
                f.write(blob)
            tensors, meta = _load_with_meta(dst)
            manifest[f"ds{i}/{os.path.basename(p)}"] = {
                "bytes": len(blob), "sha256": _sha(blob),
                "source": item.path,
                "keys": sorted(tensors),
                "shapes": {k: list(v.shape) for k, v in tensors.items()},
                "dtypes": {k: str(v.dtype) for k, v in tensors.items()},
                "meta": {k: meta.get(k) for k in
                         ("num_frames", "latent_space_version")},
            }
    with open(os.path.join(args.artifacts, "cache_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    total = len(manifest)

    # -- frame parity: arithmetic, not geometry-presence --------------------
    # The fork's own grid (src/vae.py:625): clip frames F = 17n+5 encode
    # to latent T = 5n+2 (73 -> 22); a still is T = 1; packed audio rows
    # are even-count (interleaved stereo) and 32 wide. Each claim is
    # computed from the manifest's own numbers -- garbage latents fail
    # this check, which is the point. (Warm-read behavior is NOT graded
    # here: reader-vs-cache is graded in real training under the vae_skip
    # law, from a fresh process. An in-process reader check grades this
    # process's own state and was deleted for that.)
    def _expected_t(num_frames):
        if num_frames is None:
            return 1  # still image
        n = (int(num_frames) - 5) // 17
        if n < 0 or int(num_frames) != 17 * n + 5:
            return None  # off the 17n+5 grid: arithmetic says no
        return 5 * n + 2

    par = 0
    for key, row in manifest.items():
        lat = row["shapes"].get("latent")
        t_exp = _expected_t(row["meta"].get("num_frames"))
        problems = []
        if t_exp is None:
            problems.append(f"num_frames={row['meta'].get('num_frames')} "
                            f"off the 17n+5 grid")
        if not lat or len(lat) != 4 or lat[1] != t_exp:
            problems.append(f"latent {lat} lacks expected T={t_exp}")
        if "audio_latent" in row["shapes"]:
            al = row["shapes"]["audio_latent"]
            if not (al and len(al) == 2 and al[1] == 32 and al[0] % 2 == 0):
                problems.append(f"audio rows {al} not even-count 32-wide")
        if problems:
            for pr in problems:
                print(f"fixt unexplained diff: parity {key}: {pr}", flush=True)
        else:
            par += 1
    print(f"fixt frame parity ok: {par}/{total}", flush=True)

    # -- decode artifacts: the owner's watch (mandatory) --------------------
    concat_dir = os.path.join(args.artifacts, "concat")
    os.makedirs(concat_dir, exist_ok=True)
    import torchaudio

    def _canon(x):
        """best-effort (N,C,T,H,W) normalization from whatever the loader
        hands back ((3,H,W), (1,3,H,W), (3,T,H,W), (1,3,T,H,W)); returns
        None when the shape is not one the harness knows -- then dB is
        skipped for that item with a NAMED line showing the real shape
        (truth beats a guess that crashes the run)."""
        if x.dim() == 3 and x.shape[0] == 3:
            return x.unsqueeze(0).unsqueeze(2)          # (1,3,1,H,W)
        if x.dim() == 4:
            if x.shape[0] == 1 and x.shape[1] == 3:
                return x.unsqueeze(2)                    # (1,3,1,H,W)
            if x.shape[0] == 3:
                return x.unsqueeze(0)                    # (1,3,T,H,W)
            if x.shape[1] == 3:
                # video pixels as (T,3,H,W): a known layout, not "unknown"
                return x.permute(1, 0, 2, 3).unsqueeze(0)
        if x.dim() == 5:
            return x
        return None

    concats, dbs_v, dbs_a = 0, [], []
    with torch.no_grad():  # the decodes are deliberately differentiable
        for i, ds in enumerate(datasets):
            fps = int(getattr(ds.dataset_config, "fps", 24) or 24)
            # the watch artifact needs the ORIGINAL pixels alongside the
            # cached latent: this dataset-level flag makes
            # load_and_process_image fall through the latent-cache short
            # circuit and load pixel tensors too (dataloader_mixins
            # :1144-1157) -- dB compares like with like
            ds.dataset_config.load_image_when_caching_latents = True
            for item in ds.file_list:
                key = f"ds{i}/{os.path.basename(item.get_latent_path(recalculate=True))}"
                if key not in manifest:
                    continue
                state = _load(os.path.join(cache_dir, f"ds{i}",
                                           os.path.basename(key)))
                stem = os.path.splitext(os.path.basename(item.path))[0]
                item.load_and_process_image(ds.transform, only_load_latents=False)
                orig = item.tensor
                if orig is None:
                    print(f"fixt unexplained diff: prep produced no pixels "
                          f"for {stem}", flush=True)
                    continue
                lat = state["latent"].to(device)
                if lat.dim() == 3:  # still without a time axis
                    lat = lat.unsqueeze(1)
                dec = _canon(_plain(model.decode_latents(
                    lat.unsqueeze(0))))
                if dec is None:
                    print(f"fixt unexplained diff: decode shape unknown for "
                          f"{stem} (latent {tuple(lat.shape)})", flush=True)
                    continue
                dec = dec.float().cpu()  # (N,3,T,H,W) [-1,1]
                o = _canon(item.tensor)
                if o is None:
                    print(f"fixt unexplained diff: orig pixel shape unknown "
                          f"for {stem} (tensor {tuple(item.tensor.shape)})",
                          flush=True)
                else:
                    try:
                        dbs_v.append(_db_float(o.float().cpu(), dec))
                    except RuntimeError as e:
                        print(f"fixt unexplained diff: dB shape mismatch "
                              f"for {stem}: orig {tuple(o.shape)} vs decoded "
                              f"{tuple(dec.shape)} ({e})", flush=True)
                h, w = dec.shape[-2:]
                frames = ((dec.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
                frames = frames[0].permute(0, 2, 3, 1).contiguous()
                audio = audio_sr = None
                if "audio_latent" in state:
                    rows = state["audio_latent"].to(device)
                    wav = _plain(model.decode_packed_audio_rows(
                        rows.unsqueeze(0))).float().cpu()  # (1,2,L) @32k
                    audio = wav[0]
                    audio_sr = 32000
                    if getattr(item, "audio_data", None) is not None \
                            and isinstance(item.audio_data, dict):
                        # like-with-like at 32 kHz: the decode IS 32 kHz and
                        # the source waveform is native-rate -- direct
                        # differencing measured the harness, not the pipeline
                        # (that is all the earlier -3.8 dB ever was)
                        from torchaudio.functional import resample
                        sr = int(item.audio_data.get("sample_rate") or 0)
                        if sr <= 0:
                            print(f"fixt unexplained diff: no sample rate on "
                                  f"audio for {stem} -- dB not measured",
                                  flush=True)
                        else:
                            o_w = resample(item.audio_data["waveform"].float(),
                                           sr, 32000)
                            dbs_a.append(_db_audio(o_w, wav[0]))
                mp4 = os.path.join(concat_dir, f"{stem}_decoded.mp4")
                _write_video(mp4, frames, fps,
                             audio=audio, audio_sr=audio_sr or 32000)
                try:
                    _concat(item.path, mp4,
                            os.path.join(concat_dir, f"{stem}_orig_vs_decoded.mp4"),
                            audio is not None, w, h, fps)
                    concats += 1
                except RuntimeError as e:
                    print(f"fixt concat FAILED: {stem}: {e}", flush=True)
                if audio is not None:
                    torchaudio.save(
                        os.path.join(concat_dir, f"{stem}_decoded_audio.wav"),
                        audio, audio_sr)
    if concats == 0 and total > 0:
        print("fixt unexplained diff: no concat artifacts written (the "
              "owner's watch gate has nothing to view)", flush=True)
        return 1
    print(f"fixt concat written: {concats}", flush=True)

    out = {
        "mode": "encode", "arm": args.arm, "stage": args.stage,
        "fixt_items": total,
        "frame_parity": f"{par}/{total}",
        "concats": concats,
        "roundtrip_db_video": min(dbs_v) if dbs_v else None,
        "roundtrip_db_audio": min(dbs_a) if dbs_a else None,
    }
    with open(os.path.join(args.artifacts, "fixt.json"), "w") as f:
        json.dump(out, f, indent=1)
    model.vae = None  # phase-1 boundary free, mirroring the process (:2174)
    return 0


# ------------------------------------------------------------------ compare

def _compare_standalone(args):
    import json
    import os
    from safetensors.torch import load_file
    import sys
    repo = os.getcwd()
    sys.path.insert(0, repo)
    from tools.runtool.runner import RUNS_ROOT

    rows = {}
    for tag in ("a", "b"):
        rid = getattr(args, f"run_{tag}")
        root = os.path.join(RUNS_ROOT, rid)
        with open(os.path.join(root, "artifacts", "cache_manifest.json")) as f:
            rows[tag] = (rid, json.load(f), os.path.join(root, "artifacts", "cache"))

    (ida, ma, ca), (idb, mb, cb) = rows["a"], rows["b"]
    only_a = sorted(set(ma) - set(mb))
    only_b = sorted(set(mb) - set(ma))
    for k in only_a + only_b:
        print(f"fixt unexplained diff: cache file present in only one arm: {k}",
              flush=True)

    import torch
    tot = max_diff = 0.0
    n = same_ident = audio_diff = 0.0
    n_items = 0
    for key in sorted(set(ma) & set(mb)):
        n_items += 1
        pa, pb = os.path.join(ca, key), os.path.join(cb, key)
        ra, rb = ma[key], mb[key]
        if ra["shapes"] != rb["shapes"] or ra["dtypes"] != rb["dtypes"]:
            print(f"fixt unexplained diff: shape/dtype mismatch {key}: "
                  f"{ra['shapes']}/{ra['dtypes']} vs {rb['shapes']}/{rb['dtypes']}",
                  flush=True)
            continue
        ta = load_file(pa, device="cpu")
        tb = load_file(pb, device="cpu")
        # IDENTITY LAW (measured 2026-10-29): a cache file is identical iff
        # the tensor BITS match and the metadata VALUES match. The raw file
        # hash is not the law: safetensors serializes __metadata__ as a map
        # whose key order is process-hashed (seeded arms: dict-equal header,
        # 366 header bytes permuted), so byte-for-byte files across runs
        # are unattainable by design, pipeline innocent.
        ident = _header_dict(pa) == _header_dict(pb) and set(ta) == set(tb)
        for k in ta:
            if k in tb:
                ident = ident and torch.equal(_bits(ta[k]), _bits(tb[k]))
            a, b = ta[k].float(), tb[k].float()
            if not (torch.isfinite(a).all() and torch.isfinite(b).all()):
                print(f"fixt unexplained diff: nonfinite values in {key}:{k}",
                      flush=True)
            d = (a - b).abs()
            if k == "audio_latent":
                audio_diff = max(audio_diff, d.max().item())
            tot += d.mean().item()
            n += 1
            max_diff = max(max_diff, d.max().item())
        if ident:
            same_ident += 1

    out = {"mode": "compare", "run_a": ida, "run_b": idb,
           "fixt_items": n_items,
           "fixt_diff_max": round(max_diff, 8),
           "fixt_diff_mean": round(tot / n, 10) if n else None,
           "fixt_audio_row_diff": round(float(audio_diff), 8)}

    if args.stage == "A":
        print(f"fixt bytes identical: {int(same_ident)}/{n_items}", flush=True)
        # marker prints ONLY on all-equal; any shortfall already printed
        # per-file unexplained rows above (or the identity mismatch itself):
        if same_ident != n_items:
            print(f"fixt unexplained diff: {int(n_items - same_ident)} of "
                  f"{n_items} cache states differ in stage A (identity = "
                  f"tensor bits + meta values; structure-only move must "
                  f"preserve both)", flush=True)
    else:  # stage B: nonzero is EXPECTED; within ceiling it is DECLARED
        if args.ceiling is None:  # precheck should have refused; belt
            print("fixt unexplained diff: stage B without --ceiling", flush=True)
        elif max_diff == 0.0:
            print("fixt unexplained diff: stage B but byte-identical caches "
                  "-- the quality change did not reach the cache (or the "
                  "arms are the same pipeline)", flush=True)
        elif max_diff <= args.ceiling:
            print(f"fixt diff declared: max {max_diff:.3e} <= ceiling "
                  f"{args.ceiling}", flush=True)
        else:
            print(f"fixt unexplained diff: max diff {max_diff:.3e} exceeds "
                  f"declared ceiling {args.ceiling}", flush=True)

    with open(os.path.join(args.artifacts, "compare.json"), "w") as f:
        json.dump(out, f, indent=1)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", required=True, choices=("encode", "compare"))
    ap.add_argument("--arm", default="old_path")
    ap.add_argument("--stage", default="A", choices=("A", "B"))
    ap.add_argument("--artifacts", required=True)
    ap.add_argument("--run-a", default=None)
    ap.add_argument("--run-b", default=None)
    ap.add_argument("--ceiling", type=float, default=None)
    args = ap.parse_args()
    if args.mode == "encode":
        return _encode(args)
    return _compare_standalone(args)


if __name__ == "__main__":
    sys.exit(main())
