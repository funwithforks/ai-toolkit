import { tool, type Plugin } from "@opencode-ai/plugin"

// Exposes the repo runtool (tools/runtool) as four custom tools.
// All real logic lives in the Python runner; these are thin argument
// shippers. No shell strings: every call is an argv array.
function runtoolCall(dir: string, args: string[], abort: AbortSignal): Promise<{ code: number; out: string; err: string }> {
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
  abort.addEventListener("abort", kill)
  return Promise.all([
    new Response(proc.stdout).text(),
    new Response(proc.stderr).text(),
    proc.exited,
  ]).then(([out, err, code]) => {
    abort.removeEventListener("abort", kill)
    return { code, out, err }
  })
}

const wrap =
  (build: (a: any) => string[]) =>
  async (a: any, ctx: any) => {
    const r = await runtoolCall(ctx.directory, build(a), ctx.abort)
    const text = r.out.trim() || r.err.trim() || `(no output, exit ${r.code})`
    return { output: text, metadata: { exit: r.code } }
  }

export default (async () => ({
  tool: {
    runtool_suites: tool({
      description:
        "List the runtool suite registry: suite ids, variants, typed params " +
        "(defaults/bounds), and declared metrics. Call before submit.",
      args: {},
      execute: wrap(() => ["suites"]),
    }),
    runtool_submit: tool({
      description:
        "Submit a registered suite run. Validates suite/variant/params BEFORE " +
        "launch (unknown anything = rejected, nothing starts). Returns a run_id " +
        "immediately; use runtool_wait to block until done. Params is a JSON " +
        'object string, e.g. {"dataset":"<key from datasets.json>",' +
        '"steps":120,"warmup":20}.',
      args: {
        suite: tool.schema.string().describe("suite id (see runtool_suites)"),
        variant: tool.schema.string().describe("variant within the suite"),
        params: tool.schema
          .string()
          .default("{}")
          .describe("JSON object of registered params only"),
        note: tool.schema.string().default("").describe("free-text note for the record"),
      },
      execute: wrap((a) => ["submit", "--suite", a.suite, "--variant", a.variant, "--params", a.params, "--note", a.note]),
    }),
    runtool_wait: tool({
      description:
        "Block until the run exits / is cancelled / times out, then return its " +
        "status, exit code and parsed metrics. This replaces polling and " +
        "log-scraping; it watches the recorded pid. status is one of done, " +
        "failed, crash (exit code attached), cancelled, timeout.",
      args: {
        run_id: tool.schema.string(),
        timeout_s: tool.schema.number().optional().describe("override the suite default wait timeout"),
      },
      execute: wrap((a) => ["wait", a.run_id, ...(a.timeout_s ? ["--timeout-s", String(a.timeout_s)] : [])]),
    }),
    runtool_status: tool({
      description:
        "One-shot live check of a run without waiting: recorded-pid liveness, " +
        "classification, latest structured progress event.",
      args: { run_id: tool.schema.string() },
      execute: wrap((a) => ["status", a.run_id]),
    }),
    runtool_cancel: tool({
      description: "Signal the recorded process group of a running run (TERM then KILL).",
      args: { run_id: tool.schema.string() },
      execute: wrap((a) => ["cancel", a.run_id]),
    }),
  },
})) satisfies Plugin
