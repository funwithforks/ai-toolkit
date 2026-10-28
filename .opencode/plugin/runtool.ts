import { Plugin } from "@opencode/plugin"

// Exposes the repo runtool (tools/runtool) as five custom tools.
// All real logic lives in the Python runner; these are thin argument
// shippers. No shell strings: every call is an argv array.
function runtoolCall(dir: string, args: string[], signal: AbortSignal): Promise<{ code: number; out: string; err: string }> {
  const proc = Bun.spawn([`${dir}/venv/bin/python`, "-m", "tools.runtool", ...args], {
    cwd: dir,
    stdout: "pipe",
    stderr: "pipe",
    stdin: "ignore",
  })
  const kill = () => {
    try {
      proc.kill()
    } catch {}
  }
  signal.addEventListener("abort", kill)
  return Promise.all([
    new Response(proc.stdout).text(),
    new Response(proc.stderr).text(),
    proc.exited,
  ]).then(([out, err, code]) => {
    signal.removeEventListener("abort", kill)
    return { code, out, err }
  })
}

type Args = Record<string, unknown>
const str = (v: unknown): string => (typeof v === "string" ? v : v == null ? "" : String(v))

// V1 zod defaults become explicit fallbacks here: JSON Schema inputs are
// not default-applied before execute.
const wrap =
  (dir: string, build: (a: Args) => string[]) =>
  async (input: unknown, ctx: { signal: AbortSignal }) => {
    const a = (input ?? {}) as Args
    const r = await runtoolCall(dir, build(a), ctx.signal)
    const content = r.out.trim() || r.err.trim() || `(no output, exit ${r.code})`
    return { content, metadata: { exit: r.code } }
  }

export default Plugin.define({
  id: "runtool",
  async setup(ctx) {
    const dir = ctx.location.directory
    await ctx.tool.transform((editor) => {
      editor.add({
        name: "runtool_suites",
        description:
          "List the runtool suite registry: suite ids, variants, typed params " +
          "(defaults/bounds), declared metrics, and declared markers (the named " +
          "log facts you can set expectations on). Call before submit.",
        input: { type: "object", properties: {}, additionalProperties: false },
        execute: wrap(dir, () => ["suites"]),
      })
      editor.add({
        name: "runtool_submit",
        description:
          "Submit a registered suite run. Validates suite/variant/params BEFORE " +
          "launch (unknown anything = rejected, nothing starts). Returns a run_id " +
          "immediately; use runtool_wait to block until done. Params is a JSON " +
          'object string, e.g. {"dataset":"<key from datasets.json>",' +
          '"steps":120,"warmup":20}. ' +
          "Declare the run's INTENT with expect (JSON object): marker presence " +
          "or captured values from the suite's declared markers (see " +
          "runtool_suites), and metric bounds. runtool_wait then returns a " +
          "PASS/FAIL verdict with one line per check — never grep the logs to " +
          'decide if a run did what it meant to. Example: {"markers":' +
          '{"te_skip":"present","te_load":"absent","plan":{"planned":"19"}},' +
          '"metrics":{"loss_final":{"max":1.5}}}.',
        input: {
          type: "object",
          properties: {
            suite: { type: "string", description: "suite id (see runtool_suites)" },
            variant: { type: "string", description: "variant within the suite" },
            params: {
              type: "string",
              description: "JSON object of registered params only",
              default: "{}",
            },
            expect: {
              type: "string",
              description:
                "JSON intent object graded at wait: markers 'present'/'absent' or " +
                "{namedGroup: value}, metrics {min|max|eq}. Rejected at submit if " +
                "it names undeclared markers or metrics.",
            },
            config_patch: {
              type: "string",
              description:
                "JSON object of dotted config paths (suite allowlist; see " +
                "runtool_suites patchable). Loud merge: provenance echoed in " +
                "config_resolved; conflicts with template/params reject at " +
                "submit naming every claim.",
            },
            note: {
              type: "string",
              description: "free-text note for the record",
              default: "",
            },
            after: {
              type: "string",
              description:
                "Queue behind this run_id instead of running now: starts " +
                "when the predecessor finishes (done/failed/crash all " +
                "start it; a cancelled predecessor cancels the chain). " +
                "Jobs-subtree runs are EXCLUSIVE — while any is live a " +
                "plain submit is refused with the holder named; queue " +
                "with after instead (max 3 live).",
            },
          },
          required: ["suite", "variant"],
          additionalProperties: false,
        },
        execute: wrap(dir, (a) => [
          "submit",
          "--suite",
          str(a.suite),
          "--variant",
          str(a.variant),
          "--params",
          typeof a.params === "string" ? a.params : "{}",
          ...(typeof a.expect === "string" && a.expect.trim()
            ? ["--expect", a.expect]
            : []),
          ...(typeof a.config_patch === "string" && a.config_patch.trim()
            ? ["--config-patch", a.config_patch]
            : []),
          ...(typeof a.after === "string" && a.after.trim()
            ? ["--after", a.after]
            : []),
          "--note",
          typeof a.note === "string" ? a.note : "",
        ]),
      })
      editor.add({
        name: "runtool_wait",
        description:
          "Block until the run exits / is cancelled / times out, then return " +
          "its verdict in the SAME response: status, exit code, metrics, " +
          "extracted markers, phase summary, and the PASS/FAIL expect verdict " +
          "when one was declared at submit. This replaces polling and " +
          "log-scraping; it watches the recorded pid. status is one of done, " +
          "failed, crash (exit code attached), cancelled, timeout. A declared " +
          "marker that never appeared is a named FAIL, not a shrug. Queued " +
          "runs count as in-flight: waiting on one also drives the queue " +
          "forward and returns when the run itself finishes (queue time " +
          "counts toward timeout_s).",
        input: {
          type: "object",
          properties: {
            run_id: { type: "string" },
            timeout_s: {
              type: "number",
              description: "override the suite default wait timeout",
            },
          },
          required: ["run_id"],
          additionalProperties: false,
        },
        execute: wrap(dir, (a) => [
          "wait",
          str(a.run_id),
          ...(a.timeout_s ? ["--timeout-s", String(a.timeout_s)] : []),
        ]),
      })
      editor.add({
        name: "runtool_status",
        description:
          "One-shot live check of a run without waiting: recorded-pid liveness, " +
          "classification, latest structured progress event, plus markers " +
          "extracted so far and the live phase summary (te/cache_plan/...). " +
          "Mid-run the expect verdict reads PENDING: only runtool_wait renders " +
          "the final PASS/FAIL.",
        input: {
          type: "object",
          properties: { run_id: { type: "string" } },
          required: ["run_id"],
          additionalProperties: false,
        },
        execute: wrap(dir, (a) => ["status", str(a.run_id)]),
      })
      editor.add({
        name: "runtool_inspect",
        description:
          "Diagnostic digest of a run — for when something looks strange and " +
          "nobody predicted what to look for. Returns: ugly-line signals " +
          "(traceback/OOM/NaN/Killed counts + last line), stall seconds for " +
          "live runs, repeated-line spam, last events, marker positions, " +
          "record sizes, and the raw log tails. tailLines chooses how much " +
          "raw text (0 = whole file); grep runs YOUR regex over the full " +
          "record with line numbers (context = surrounding lines); showConfig " +
          "inlines the derived config. The sanctioned way to read a run's " +
          "own logs — richer than shell grep and one call.",
        input: {
          type: "object",
          properties: {
            run_id: { type: "string" },
            tail_lines: {
              type: "number",
              description: "tail lines per log; 0 or negative = whole file",
            },
            grep: { type: "string", description: "regex over full stdout+stderr" },
            context: {
              type: "number",
              description: "surrounding lines per grep hit",
            },
            show_config: { type: "boolean" },
          },
          required: ["run_id"],
          additionalProperties: false,
        },
        execute: wrap(dir, (a) => [
          "inspect",
          str(a.run_id),
          ...(a.tail_lines != null ? ["--tail", String(a.tail_lines)] : []),
          ...(typeof a.grep === "string" && a.grep
            ? ["--grep", a.grep, "--context", String(a.context ?? 0)]
            : []),
          ...(a.show_config ? ["--show-config"] : []),
        ]),
      })
      editor.add({
        name: "runtool_gc",
        description:
          "Reap the runtool jobs subtree: finished tool runs, orphan dirs, " +
          "stale fixed-embed caches (regenerate on demand). Without apply: " +
          "lists candidates with sizes/ages/reasons. apply deletes ONLY " +
          "inside output/runtool/jobs — records are never touched, and " +
          "judged (sample=true) runs need include_judged after the owner " +
          "has judged them. Owner ruling: tool jobs are disposable.",
        input: {
          type: "object",
          properties: {
            days: { type: "number", description: "age threshold (default 7)" },
            apply: { type: "boolean" },
            include_judged: { type: "boolean" },
          },
          additionalProperties: false,
        },
        execute: wrap(dir, (a) => [
          "gc",
          ...(a.days != null ? ["--days", String(a.days)] : []),
          ...(a.apply ? ["--apply"] : []),
          ...(a.include_judged ? ["--include-judged"] : []),
        ]),
      })
      editor.add({
        name: "runtool_cancel",
        description: "Signal the recorded process group of a running run (TERM then KILL).",
        input: {
          type: "object",
          properties: { run_id: { type: "string" } },
          required: ["run_id"],
          additionalProperties: false,
        },
        execute: wrap(dir, (a) => ["cancel", str(a.run_id)]),
      })
    })
  },
})
