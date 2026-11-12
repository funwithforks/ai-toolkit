import fs from 'fs';
import path from 'path';

export interface DatasetCounts {
  image: number;
  video: number;
  audio: number;
  total: number;
}

// Media classification shared by the dataset list counts and the image
// listing route, so both always agree on what counts as a dataset item.
export const imageExtensions = ['.png', '.jpg', '.jpeg', '.webp'];
export const videoExtensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.m4v', '.flv'];
export const audioExtensions = ['.mp3', '.wav', '.flac', '.ogg'];
export const mediaExtensions = [...imageExtensions, ...videoExtensions, ...audioExtensions];

const ZERO_COUNTS: DatasetCounts = { image: 0, video: 0, audio: 0, total: 0 };

// Per-dataset counts cache. A dataset is recounted only when its root
// directory mtime changes or the entry ages out. The max-age also catches
// changes in nested subfolders, which don't move the root mtime.
const CACHE_MAX_AGE_MS = 60_000;

interface CacheEntry {
  counts: DatasetCounts;
  dirMtimeMs: number;
  countedAt: number;
}

const cache = new Map<string, CacheEntry>();
const inFlight = new Map<string, Promise<DatasetCounts>>();

async function countFilesRecursively(dir: string, counts: DatasetCounts): Promise<void> {
  // withFileTypes avoids a separate stat per entry — counting is a pure
  // readdir walk, so it can't disturb concurrent file transfers.
  const entries = await fs.promises.readdir(dir, { withFileTypes: true });

  const subdirs: string[] = [];
  for (const entry of entries) {
    const name = entry.name;
    if (name.startsWith('.')) continue;
    if (entry.isDirectory()) {
      if (name === '_controls') continue;
      subdirs.push(path.join(dir, name));
    } else if (entry.isFile()) {
      const ext = path.extname(name).toLowerCase();
      if (imageExtensions.includes(ext)) {
        counts.image++;
      } else if (videoExtensions.includes(ext)) {
        counts.video++;
      } else if (audioExtensions.includes(ext)) {
        counts.audio++;
      }
    }
  }

  // Recurse into subdirectories concurrently.
  await Promise.all(subdirs.map(subdir => countFilesRecursively(subdir, counts)));
}

export async function getDatasetCounts(datasetDir: string): Promise<DatasetCounts> {
  let stat: fs.Stats;
  try {
    stat = await fs.promises.stat(datasetDir);
  } catch {
    // Directory vanished (deleted or moved): nothing to count.
    return { ...ZERO_COUNTS };
  }
  if (!stat.isDirectory()) {
    return { ...ZERO_COUNTS };
  }

  const cached = cache.get(datasetDir);
  if (cached && cached.dirMtimeMs === stat.mtimeMs && Date.now() - cached.countedAt < CACHE_MAX_AGE_MS) {
    return cached.counts;
  }

  // Share a single walk between concurrent requests.
  const pending = inFlight.get(datasetDir);
  if (pending) {
    return pending;
  }

  const walk = (async () => {
    const counts: DatasetCounts = { image: 0, video: 0, audio: 0, total: 0 };
    await countFilesRecursively(datasetDir, counts);
    counts.total = counts.image + counts.video + counts.audio;
    cache.set(datasetDir, { counts, dirMtimeMs: stat.mtimeMs, countedAt: Date.now() });
    return counts;
  })().finally(() => {
    inFlight.delete(datasetDir);
  });
  inFlight.set(datasetDir, walk);
  return walk;
}
