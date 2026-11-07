/**
 * Opt-in client-side performance counters for diagnosing render and poll
 * load on heavy pages (e.g. /jobs with many rows).
 *
 * Enable in the browser console:
 *   localStorage.setItem('AI_TOOLKIT_PERF', '1')
 * then reload. Counters are dumped as one console line every 10 seconds
 * (counts per window, so each line reads as a rate) and on pagehide.
 *
 * Everything is a no-op until the flag is set; the flag is read once.
 */
const ENABLE_KEY = 'AI_TOOLKIT_PERF';
const DUMP_INTERVAL_MS = 10_000;

let enabled: boolean | null = null;
let dumpTimer: ReturnType<typeof setInterval> | null = null;
const counters: Record<string, number> = {};

function isEnabled(): boolean {
  if (enabled === null) {
    enabled = typeof window !== 'undefined' && window.localStorage.getItem(ENABLE_KEY) === '1';
    if (enabled && !dumpTimer) {
      dumpTimer = setInterval(() => dump('10s'), DUMP_INTERVAL_MS);
      window.addEventListener('pagehide', () => dump('pagehide'));
    }
  }
  return enabled;
}

/** Count one occurrence (or `n` occurrences) under `key`. */
export function perfCount(key: string, n = 1) {
  if (!isEnabled()) return;
  counters[key] = (counters[key] || 0) + n;
}

/** True when counters are being collected (for gating expensive measurement). */
export function perfEnabled(): boolean {
  return isEnabled();
}

/** Print and reset the counters. */
export function dump(label: string) {
  if (!isEnabled()) return;
  const keys = Object.keys(counters);
  if (keys.length === 0) return;
  const parts = keys.sort().map(k => `${k}=${counters[k]}`);
  console.log(`[perf ${label}] ${parts.join(' ')}`);
  for (const k of keys) delete counters[k];
}
