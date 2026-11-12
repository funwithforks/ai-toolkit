'use client';

import { useEffect, useState } from 'react';
import { apiClient } from '@/utils/api';

export interface DatasetCounts {
  image: number;
  video: number;
  audio: number;
  total: number;
}

export interface DatasetInfo {
  name: string;
  // null when the server could not count this dataset (e.g. unreadable folder)
  counts: DatasetCounts | null;
}

export default function useDatasetList() {
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');

  const refreshDatasets = () => {
    setStatus('loading');
    apiClient
      .get('/api/datasets/list')
      .then(res => res.data)
      .then(data => {
        console.log('Datasets:', data);
        // sort
        data.sort((a: DatasetInfo, b: DatasetInfo) => a.name.localeCompare(b.name));
        setDatasets(data);
        setStatus('success');
      })
      .catch(error => {
        console.error('Error fetching datasets:', error);
        setStatus('error');
      });
  };
  useEffect(() => {
    refreshDatasets();
  }, []);

  return { datasets, setDatasets, status, refreshDatasets };
}
