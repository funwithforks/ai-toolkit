import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { getDatasetsRoot } from '@/server/settings';
import { getDatasetCounts } from '@/server/datasetCounts';

export async function GET() {
  try {
    let datasetsPath = await getDatasetsRoot();

    // if folder doesnt exist, create it
    try {
      await fs.promises.access(datasetsPath);
    } catch {
      await fs.promises.mkdir(datasetsPath);
    }

    // find all the folders in the datasets folder
    const folders = (await fs.promises.readdir(datasetsPath, { withFileTypes: true }))
      .filter(dirent => dirent.isDirectory())
      .filter(dirent => !dirent.name.startsWith('.'))
      .map(dirent => dirent.name);

    const datasets = await Promise.all(
      folders.map(async name => {
        try {
          const counts = await getDatasetCounts(path.join(datasetsPath, name));
          return { name, counts };
        } catch (error) {
          // One unreadable dataset shouldn't blank the whole page.
          console.error(`Failed to count items in dataset '${name}':`, error);
          return { name, counts: null };
        }
      }),
    );

    return NextResponse.json(datasets);
  } catch (error) {
    return NextResponse.json({ error: 'Failed to fetch datasets' }, { status: 500 });
  }
}
