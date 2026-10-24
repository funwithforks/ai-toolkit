import { NextRequest, NextResponse } from 'next/server';
import prisma from '@/server/prisma';
import path from 'path';
import fs from 'fs';
import { getTrainingFolder } from '@/server/settings';

// The comfy sampler journals the exact prompt strings it ships to the remote
// box (comfy_samples.json, rounds[].items[].prompt - inline DOP tokens
// already resolved). Items map 1:1 onto the job config's sample items by
// idx. Rounds repeat the same resolved prompts for a given config, so the
// newest round that carries items is the current truth. No journal (non-comfy
// job, or first round not queued yet) means prompts: null and the UI keeps
// showing the config text.
export async function GET(request: NextRequest, { params }: { params: { jobID: string } }) {
  const { jobID } = await params;

  const job = await prisma.job.findUnique({
    where: { id: jobID },
  });

  if (!job) {
    return NextResponse.json({ error: 'Job not found' }, { status: 404 });
  }

  const trainingFolder = await getTrainingFolder();
  const journalPath = path.join(trainingFolder, job.name, 'comfy_samples.json');

  let data: any;
  try {
    // the trainer writes this atomically (tmp + rename), a plain read is safe
    data = JSON.parse(await fs.promises.readFile(journalPath, 'utf-8'));
  } catch {
    return NextResponse.json({ prompts: null });
  }

  const rounds = Array.isArray(data?.rounds) ? data.rounds : [];
  for (let r = rounds.length - 1; r >= 0; r--) {
    const items = Array.isArray(rounds[r]?.items) ? rounds[r].items : [];
    if (items.length === 0) continue;
    const prompts: (string | null)[] = [];
    for (const item of items) {
      if (typeof item?.idx === 'number' && typeof item?.prompt === 'string') {
        prompts[item.idx] = item.prompt;
      }
    }
    if (prompts.some(p => typeof p === 'string')) {
      return NextResponse.json({ prompts });
    }
  }
  return NextResponse.json({ prompts: null });
}
