'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { Job } from '@prisma/client';
import { apiClient } from '@/utils/api';
import { perfCount, perfEnabled } from '@/utils/perfLog';

type UseJobsListProps = {
  onlyActive?: boolean;
  reloadInterval?: number | null;
  job_type?: string | null;
};

type Status = 'idle' | 'loading' | 'success' | 'error';

/**
 * Shared per-query job stores. Every hook instance asking for the same query
 * (same onlyActive/job_type) shares one poll loop and one payload, so e.g.
 * the jobs page and the sidebar widget no longer fetch and parse the same
 * data independently.
 *
 * Updates are identity-preserving: a poll that returns a job with unchanged
 * fields reuses the previous job object, so consumers can memo on object
 * identity, and a poll where nothing changed notifies nobody at all.
 */
interface JobsStore {
  params: Record<string, string>;
  reloadInterval: number | null;
  jobs: Job[];
  status: Status;
  listeners: Set<(jobs: Job[], status: Status) => void>;
  fetching: boolean;
  loopRunning: boolean;
  lastFetchedAt: number;
}

const stores = new Map<string, JobsStore>();

function storeKey(onlyActive: boolean, job_type: string | null) {
  return `${onlyActive ? 'active' : 'full'}|${job_type ?? ''}`;
}

function getStore(
  key: string,
  opts: { onlyActive: boolean; job_type: string | null; reloadInterval: number | null },
): JobsStore {  let store = stores.get(key);
  if (!store) {
    const params: Record<string, string> = {};
    if (opts.job_type) {
      params.job_type = opts.job_type;
    }
    if (opts.onlyActive) {
      params.only_active = 'true';
    }
    store = {
      params,
      reloadInterval: opts.reloadInterval,
      jobs: [],
      status: 'idle',
      listeners: new Set(),
      fetching: false,
      loopRunning: false,
      lastFetchedAt: 0,
    };
    stores.set(key, store);
  } else if (store.reloadInterval === null && opts.reloadInterval !== null) {
    store.reloadInterval = opts.reloadInterval;
  }
  return store;
}

function notify(store: JobsStore) {
  for (const listener of store.listeners) {
    listener(store.jobs, store.status);
  }
}

function jobFieldsEqual(a: Job, b: Job): boolean {
  for (const key of Object.keys(a) as (keyof Job)[]) {
    if (a[key] !== b[key]) return false;
  }
  return true;
}

/**
 * Merge a fresh poll result into the previous array, reusing previous job
 * objects wherever nothing changed so downstream memoization can rely on
 * object identity. Returns the previous array itself when nothing at all
 * changed (including order).
 */
function mergeJobs(prev: Job[], incoming: Job[]): { jobs: Job[]; changed: boolean } {
  if (prev.length === 0) {
    return { jobs: incoming, changed: incoming.length > 0 };
  }
  const prevById = new Map(prev.map(job => [job.id, job]));
  let changed = prev.length !== incoming.length;
  const next = incoming.map(job => {
    const prevJob = prevById.get(job.id);
    if (prevJob && jobFieldsEqual(prevJob, job)) {
      return prevJob;
    }
    changed = true;
    return job;
  });
  if (!changed) {
    for (let i = 0; i < next.length; i++) {
      if (next[i] !== prev[i]) {
        changed = true;
        break;
      }
    }
  }
  return { jobs: changed ? next : prev, changed };
}

async function fetchStore(store: JobsStore): Promise<void> {
  if (store.fetching) return;
  store.fetching = true;
  try {
    const res = await apiClient.get('/api/jobs', { params: store.params });
    const data = res.data;
    perfCount(`jobsList.fetch.${store.params.only_active === 'true' ? 'active' : 'full'}`);
    if (perfEnabled()) {
      perfCount('jobsList.fetch.bytes', JSON.stringify(data).length);
    }
    if (data.error) {
      console.log('Error fetching jobs:', data.error);
      store.status = 'error';
      notify(store);
    } else {
      const { jobs, changed } = mergeJobs(store.jobs, data.jobs);
      const statusChanged = store.status !== 'success';
      if (changed) {
        store.jobs = jobs;
      } else {
        perfCount('jobsList.fetch.unchanged');
      }
      if (changed || statusChanged) {
        store.status = 'success';
        notify(store);
      }
    }
  } catch (error) {
    console.error('Error fetching jobs:', error);
    store.status = 'error';
    notify(store);
  } finally {
    store.fetching = false;
    store.lastFetchedAt = Date.now();
  }
}

/**
 * Settle-then-wait loop (same contract as usePollLoop): the next poll is
 * scheduled only after the current one settles. Exits once the last
 * subscriber unsubscribes; a later subscribe restarts it.
 */
function ensureLoop(store: JobsStore) {
  if (store.loopRunning) return;
  store.loopRunning = true;
  void (async () => {
    for (;;) {
      await fetchStore(store);
      if (store.listeners.size === 0) {
        store.loopRunning = false;
        return;
      }
      await new Promise(resolve => setTimeout(resolve, store.reloadInterval ?? 0));
    }
  })();
}

/**
 * Existing store for a key, without creating one. Used to seed hook state
 * synchronously during the first render so a revisited page never paints an
 * empty frame while waiting for the mount effect to run.
 */
function peekStore(key: string): JobsStore | null {
  return stores.get(key) ?? null;
}

export default function useJobsList({
  onlyActive = false,
  reloadInterval = null,
  job_type = null,
}: UseJobsListProps = {}) {
  const key = storeKey(onlyActive, job_type);
  // Sync seed from the shared cache (survives client-side navigation).
  const [seed] = useState(() => peekStore(key));
  const [jobs, setJobs] = useState<Job[]>(seed?.jobs ?? []);
  const [status, setStatus] = useState<Status>(seed ? seed.status : 'idle');
  const storeRef = useRef<JobsStore | null>(null);

  useEffect(() => {
    const store = getStore(key, { onlyActive, job_type, reloadInterval });
    storeRef.current = store;
    const listener = (incomingJobs: Job[], incomingStatus: Status) => {
      setJobs(incomingJobs);
      setStatus(incomingStatus);
    };
    store.listeners.add(listener);
    // Seed from the shared cache so revisiting a page renders instantly.
    if (store.jobs.length > 0) {
      setJobs(store.jobs);
      setStatus(store.status);
    } else if (store.status === 'idle') {
      setStatus('loading');
    }
    ensureLoop(store);
    // Revisiting a page must not show stale data for up to a full interval:
    // if the shared cache is older than one poll, refresh right away. (The
    // fetching guard makes this safe when the loop is mid-request.)
    if (Date.now() - store.lastFetchedAt > (store.reloadInterval ?? 0)) {
      void fetchStore(store);
    }
    return () => {
      store.listeners.delete(listener);
    };
  }, [key, onlyActive, job_type, reloadInterval]);

  const refreshJobs = useCallback(() => {
    const store = storeRef.current;
    if (!store) return Promise.resolve();
    return fetchStore(store);
  }, []);

  return { jobs, setJobs, status, refreshJobs };
}
