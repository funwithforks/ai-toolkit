import { JobConfig } from '@/types';
import { Job } from '@prisma/client';
import { apiClient } from '@/utils/api';
import { perfCount } from '@/utils/perfLog';

export const startJob = (jobID: string) => {
  return new Promise<void>((resolve, reject) => {
    apiClient
      .get(`/api/jobs/${jobID}/start`)
      .then(res => res.data)
      .then(data => {
        console.log('Job started:', data);
        resolve();
      })
      .catch(error => {
        console.error('Error starting job:', error);
        reject(error);
      });
  });
};

export const stopJob = (jobID: string) => {
  return new Promise<void>((resolve, reject) => {
    apiClient
      .get(`/api/jobs/${jobID}/stop`)
      .then(res => res.data)
      .then(data => {
        console.log('Job stopped:', data);
        resolve();
      })
      .catch(error => {
        console.error('Error stopping job:', error);
        reject(error);
      });
  });
};

export const deleteJob = (jobID: string) => {
  return new Promise<void>((resolve, reject) => {
    apiClient
      .get(`/api/jobs/${jobID}/delete`)
      .then(res => res.data)
      .then(data => {
        console.log('Job deleted:', data);
        resolve();
      })
      .catch(error => {
        console.error('Error deleting job:', error);
        reject(error);
      });
  });
};

export const saveJobNow = (jobID: string) => {
  return new Promise<void>((resolve, reject) => {
    apiClient
      .get(`/api/jobs/${jobID}/save_now`)
      .then(res => res.data)
      .then(data => {
        console.log('Job set to save on next step:', data);
        resolve();
      })
      .catch(error => {
        console.error('Error setting job to save on next step:', error);
        reject(error);
      });
  });
};

export const sampleJobNow = (jobID: string) => {
  return new Promise<void>((resolve, reject) => {
    apiClient
      .get(`/api/jobs/${jobID}/sample_now`)
      .then(res => res.data)
      .then(data => {
        console.log('Job set to sample on next step:', data);
        resolve();
      })
      .catch(error => {
        console.error('Error setting job to sample on next step:', error);
        reject(error);
      });
  });
};

export const markJobAsStopped = (jobID: string) => {
  return new Promise<void>((resolve, reject) => {
    apiClient
      .get(`/api/jobs/${jobID}/mark_stopped`)
      .then(res => res.data)
      .then(data => {
        console.log('Job marked as stopped:', data);
        resolve();
      })
      .catch(error => {
        console.error('Error marking job as stopped:', error);
        reject(error);
      });
  });
};

// Parsed job_config cache, keyed by job id + config string. A jobs-list
// render used to JSON.parse the same ~3 KB config twice per row per render;
// with the cache each distinct config is parsed once.
const jobConfigCache = new Map<string, JobConfig | null>();
const JOB_CONFIG_CACHE_MAX = 500;

export const getJobConfig = (job: Job): JobConfig | null => {
  const key = `${job.id}|${job.job_config}`;
  const hit = jobConfigCache.get(key);
  if (jobConfigCache.has(key)) {
    perfCount('jobConfig.cacheHit');
    return hit ?? null;
  }
  perfCount('jobConfig.parse');
  let parsed: JobConfig | null = null;
  try {
    parsed = JSON.parse(job.job_config) as JobConfig;
  } catch (e) {
    // A corrupt config must not crash a render; callers treat null as
    // "config unreadable" and degrade the affected feature only.
    console.error('Error parsing job config for job', job.id, e);
    parsed = null;
  }
  if (jobConfigCache.size >= JOB_CONFIG_CACHE_MAX) {
    jobConfigCache.clear();
  }
  jobConfigCache.set(key, parsed);
  return parsed;
};

export const getAvaliableJobActions = (job: Job) => {
  const jobConfig = getJobConfig(job);
  const isStopping = job.stop && job.status === 'running';
  const canDelete = ['queued', 'completed', 'stopped', 'error'].includes(job.status) && !isStopping;
  let canEdit = ['queued', 'completed', 'stopped', 'error'].includes(job.status) && !isStopping;
  const canRemoveFromQueue = job.status === 'queued';
  const canStop = job.status === 'running' && !isStopping;
  let canStart = ['stopped', 'error'].includes(job.status) && !isStopping;
  // can resume if more steps were added
  if (
    job.status === 'completed' &&
    (jobConfig?.config.process[0].train?.steps || 0) > job.step &&
    !isStopping
  ) {
    canStart = true;
  }
  return { canDelete, canEdit, canStop, canStart, canRemoveFromQueue };
};

export const getNumberOfSamples = (job: Job) => {
  const jobConfig = getJobConfig(job);
  return jobConfig?.config.process[0].sample?.prompts?.length || 0;
};

export const getTotalSteps = (job: Job) => {
  if (job.total_steps != null) {
    return job.total_steps;
  }
  const jobConfig = getJobConfig(job);
  return jobConfig?.config.process[0].train?.steps || 0;
};
