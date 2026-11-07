'use client';

import { GpuInfo } from '@/types';
import { useEffect, useMemo, useRef, useState } from 'react';
import useMonitorStream, { getMonitorStreamState, subscribeMonitorStream } from '@/hooks/useMonitorStream';
import type { MonitorStreamState } from '@/hooks/useMonitorStream';

/**
 * GPU stats from the shared /api/monitor SSE stream. Data arrives live every
 * MONITOR_TICK_MS, so `reloadInterval` is accepted only for call-site
 * compatibility with the old polling implementation and is ignored.
 */
export default function useGPUInfo(gpuIds: null | number[] = null, reloadInterval: null | number = null) {
  void reloadInterval;
  const { gpu, connected } = useMonitorStream();

  // Key on contents, not identity — call sites often build gpuIds inline.
  const gpuIdsKey = gpuIds ? gpuIds.join(',') : null;
  const gpuList: GpuInfo[] = useMemo(() => {
    if (!gpu) return [];
    let gpus = [...gpu.gpus].sort((a, b) => a.index - b.index);
    if (gpuIdsKey !== null) {
      const ids = gpuIdsKey === '' ? [] : gpuIdsKey.split(',').map(Number);
      gpus = gpus.filter(g => ids.includes(g.index));
    }
    return gpus;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [gpu, gpuIdsKey]);

  const isGPUInfoLoaded = gpu !== null;
  const status: 'idle' | 'loading' | 'success' | 'error' = gpu !== null ? 'success' : connected ? 'loading' : 'idle';

  return {
    gpuList,
    isGPUInfoLoaded,
    status,
    // The stream is always live; nothing to refresh manually.
    refreshGpuInfo: async () => {},
  };
}

export interface GpuStructure {
  index: number;
  name: string;
}

function structureFromState(s: MonitorStreamState): { list: GpuStructure[]; loaded: boolean; sig: string } {
  const sig = s.gpu ? s.gpu.gpus.map(g => `${g.index}:${g.name}`).join('|') : '(unloaded)';
  const list = s.gpu
    ? [...s.gpu.gpus].sort((a, b) => a.index - b.index).map(g => ({ index: g.index, name: g.name }))
    : [];
  return { list, loaded: s.gpu !== null, sig };
}

/**
 * Structure-only view of the GPU list ({index, name}), for consumers that
 * group or select GPUs and must not re-render on every 500 ms monitor
 * sample. Shares the single SSE connection via subscribeMonitorStream but
 * only updates state when the set of GPUs (index + name) actually changes.
 *
 * Seeds synchronously from the shared stream's last-known state: the module
 * state survives client-side navigation, so revisiting a page renders with
 * the previous GPU list immediately instead of waiting for the stream to
 * re-establish (leader election + connect + init).
 */
export function useGPUStructure() {
  const [initial] = useState(() => structureFromState(getMonitorStreamState()));
  const [gpuList, setGpuList] = useState<GpuStructure[]>(initial.list);
  const [isLoaded, setIsLoaded] = useState(initial.loaded);
  const sigRef = useRef(initial.sig);

  useEffect(() => {
    return subscribeMonitorStream(s => {
      const { list, loaded, sig } = structureFromState(s);
      if (sig !== sigRef.current) {
        sigRef.current = sig;
        setGpuList(list);
        setIsLoaded(loaded);
      }
    });
  }, []);

  return { gpuList, isGPUInfoLoaded: isLoaded };
}
