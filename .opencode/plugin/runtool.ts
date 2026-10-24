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
          "(defaults/bounds), and declared metrics. Call before submit.",
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
          '"steps":120,"warmup":20}.',
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
            note: {
              type: "string",
              description: "free-text note for the record",
              default: "",
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
          "--note",
          typeof a.note === "string" ? a.note : "",
        ]),
      })
      editor.add({
        name: "runtool_wait",
        description:
          "Block until the run exits / is cancelled / times out, then return its " +
          "status, exit code and parsed metrics. This replaces polling and " +
          "log-scraping; it watches the recorded pid. status is one of done, " +
          "failed, crash (exit code attached), cancelled, timeout.",
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
          "classification, latest structured progress event.",
        input: {
          type: "object",
          properties: { run_id: { type: "string" } },
          required: ["run_id"],
          additionalProperties: false,
        },
        execute: wrap(dir, (a) => ["status", str(a.run_id)]),
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
