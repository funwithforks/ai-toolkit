import { NextResponse } from 'next/server';
import { spawn } from 'child_process';
import { TOOLKIT_ROOT } from '@/paths';
import { resolvePythonPath } from '../../../../cron/pythonPath';

// runtool-over-HTTP: forwards one runtool op to this machine's CLI and
// returns its output verbatim. The single consumer is runtool itself
// (`--machine NAME`), which owns all formatting and error envelopes; this
// route adds nothing to the output and parses nothing but its own request
// body. Auth is the standard AI_TOOLKIT_AUTH middleware that covers
// every /api route.
//
// The route holds no state: each call spawns `python -m tools.runtool <op>`
// (argv array, no shell), captures stdout/stderr, and answers with
// {exit_code, stdout, stderr}. No route-level timeout — every call the
// client makes is short-lived (its remote wait is a status-poll loop), and
// when the client disconnects its own socket timeout, the child is killed
// via the request abort signal.

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

const OPS = new Set([
  'suites',
  'config-schema',
  'datasets',
  'submit',
  'wait',
  'status',
  'cancel',
  'inspect',
  'list',
  'search',
  'judge',
  'gc',
  'complain',
  'complaints',
  'recompute',
  'machines',
]);

export async function POST(request: Request) {
  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: 'body must be JSON' }, { status: 400 });
  }
  const argv = (body as { argv?: unknown })?.argv;
  if (
    !Array.isArray(argv) ||
    argv.length === 0 ||
    argv.length > 64 ||
    !argv.every(t => typeof t === 'string') ||
    !OPS.has(argv[0])
  ) {
    return NextResponse.json(
      {
        error:
          'argv must be a non-empty array of strings whose first element is a runtool op',
      },
      { status: 400 },
    );
  }

  const child = spawn(resolvePythonPath(), ['-m', 'tools.runtool', ...argv], {
    cwd: TOOLKIT_ROOT,
    stdio: ['ignore', 'pipe', 'pipe'],
    env: { ...process.env, PYTHONIOENCODING: 'utf-8' },
  });

  const abort = () => child.kill('SIGTERM');
  request.signal.addEventListener('abort', abort);

  let stdout = '';
  let stderr = '';
  child.stdout!.on('data', (c: Buffer) => (stdout += c.toString()));
  child.stderr!.on('data', (c: Buffer) => (stderr += c.toString()));

  return new Promise<Response>(resolve => {
    child.on('error', err => {
      request.signal.removeEventListener('abort', abort);
      resolve(
        NextResponse.json(
          {
            exit_code: null,
            stdout: '',
            stderr: `spawn failed: ${err.message}`,
          },
          { status: 200 },
        ),
      );
    });
    child.on('close', code => {
      request.signal.removeEventListener('abort', abort);
      resolve(
        NextResponse.json(
          { exit_code: code, stdout, stderr },
          { status: 200 },
        ),
      );
    });
  });
}
