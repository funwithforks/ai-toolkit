'use client';

import { GpuInfo } from '@/types';
import { useEffect, useMemo, useRef, useState } from 'react';
import useMonitorStream, { subscribeMonitorStream } from '@/hooks/useMonitorStream';

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

/**
 * Structure-only view of the GPU list ({index, name}), for consumers that
 * group or select GPUs and must not re-render on every 500 ms monitor
 * sample. Shares the single SSE connection via subscribeMonitorStream but
 * only updates state when the set of GPUs (index + name) actually changes.
 */
export function useGPUStructure() {
  const [gpuList, setGpuList] = useState<GpuStructure[]>([]);
  const [isLoaded, setIsLoaded] = useState(false);
  const sigRef = useRef<string>('(unloaded)');

  useEffect(() => {
    return subscribeMonitorStream(s => {
      const sig = s.gpu ? s.gpu.gpus.map(g => `${g.index}:${g.name}`).join('|') : '(unloaded)';
      if (sig !== sigRef.current) {
        sigRef.current = sig;
        setGpuList(
          s.gpu
            ? [...s.gpu.gpus].sort((a, b) => a.index - b.index).map(g => ({ index: g.index, name: g.name }))
            : [],
        );
        setIsLoaded(s.gpu !== null);
      }
    });
  }, []);

  return { gpuList, isGPUInfoLoaded: isLoaded };
}
