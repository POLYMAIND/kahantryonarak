/**
 * Katalógus elő-renderelő batch — keret × modell-archetípus.
 *
 *   still:  GPT Image 2 edit (portré + keret → try-on állókép)
 *   video:  fal-ai/minimax/hailuo-02-fast/image-to-video (512p, 6 mp)
 *   store:  saját S3-kompatibilis bucket, idempotens cache-kulcsokkal
 *
 * Futtatás:
 *   tsx scripts/prerender-catalog.ts                     # csak a hiányzókat gyártja le
 *   tsx scripts/prerender-catalog.ts --only-stills
 *   tsx scripts/prerender-catalog.ts --frames 5409B,5411A --concurrency 4
 *   tsx scripts/prerender-catalog.ts --force             # cache figyelmen kívül hagyása
 *   tsx scripts/prerender-catalog.ts --dry-run           # csak a terv kiírása
 *
 * Env:
 *   OPENAI_API_KEY, FAL_KEY
 *   S3_BUCKET, S3_REGION, S3_ENDPOINT?, S3_ACCESS_KEY_ID, S3_SECRET_ACCESS_KEY, CDN_BASE_URL
 *
 * Deps: openai @fal-ai/client @aws-sdk/client-s3 tsx
 */

import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import OpenAI from "openai";
import { fal } from "@fal-ai/client";
import {
  S3Client,
  HeadObjectCommand,
  PutObjectCommand,
  GetObjectCommand,
} from "@aws-sdk/client-s3";

/* ─── verziózott pipeline-paraméterek ──────────────────────────────────────
   Bármelyik érték változása új cache-kulcsot ad, tehát újragyártást vált ki.
   Ez a szerződés az idempotencia és a "mikor kell újrafuttatni" között.      */

const PIPELINE = {
  still: {
    v: 3,
    model: "gpt-image-2",
    size: "1024x1536" as const,
    quality: "high" as const,
    prompt:
      "Place the eyewear frame from the product image onto the person's face in the portrait. " +
      "Match head pose, perspective and scale precisely; keep the frame's exact shape, colour, " +
      "metal finish and rim thickness unchanged. Preserve the person's identity, skin texture, " +
      "hair and the original background and lighting. Add realistic lens reflections and a soft " +
      "contact shadow on the nose bridge and temples. Photographic, no illustration.",
  },
  video: {
    v: 2,
    model: "fal-ai/minimax/hailuo-02-fast/image-to-video",
    resolution: "512P" as const,
    duration: 6, // mp — a modell alsó határa
    prompt:
      "The person turns their head slowly and gently, a subtle natural micro-movement, eyes to camera. " +
      "The eyewear stays perfectly fixed on the face: its shape, colour and rim thickness must not change. " +
      "No zoom, no cuts, no camera motion, no change of identity or clothing.",
  },
} as const;

/* ─── input katalógus ─────────────────────────────────────────────────────── */

type Frame = { id: string; sku: string; name: string; imageUrl: string };
type Archetype = { id: string; label: string; portraitUrl: string };

type Pair = { frame: Frame; archetype: Archetype };

type Manifest = {
  frameId: string;
  archetypeId: string;
  still: { key: string; url: string; cacheKey: string; pipeline: number };
  video?: { key: string; url: string; cacheKey: string; pipeline: number; durationSec: number };
  generatedAt: string;
};

/* ─── env / kliensek ──────────────────────────────────────────────────────── */

const env = (k: string, required = true) => {
  const v = process.env[k];
  if (!v && required) throw new Error(`Missing env: ${k}`);
  return v ?? "";
};

const BUCKET = env("S3_BUCKET");
const CDN = env("CDN_BASE_URL").replace(/\/$/, "");

const openai = new OpenAI({ apiKey: env("OPENAI_API_KEY") });
fal.config({ credentials: env("FAL_KEY") });

const s3 = new S3Client({
  region: env("S3_REGION"),
  endpoint: process.env.S3_ENDPOINT || undefined,
  forcePathStyle: !!process.env.S3_ENDPOINT,
  credentials: {
    accessKeyId: env("S3_ACCESS_KEY_ID"),
    secretAccessKey: env("S3_SECRET_ACCESS_KEY"),
  },
});

/* ─── CLI ─────────────────────────────────────────────────────────────────── */

const argv = process.argv.slice(2);
const flag = (name: string) => argv.includes(`--${name}`);
const opt = (name: string) => {
  const i = argv.indexOf(`--${name}`);
  return i >= 0 ? argv[i + 1] : undefined;
};

const OPTS = {
  force: flag("force"),
  dryRun: flag("dry-run"),
  onlyStills: flag("only-stills"),
  concurrency: Number(opt("concurrency") ?? 3),
  frames: opt("frames")?.split(",").map((s) => s.trim()).filter(Boolean),
  archetypes: opt("archetypes")?.split(",").map((s) => s.trim()).filter(Boolean),
};

/* ─── cache-kulcs ─────────────────────────────────────────────────────────
   A kulcs a teljes bemenetet fedi: forrásképek tartalom-hash-e + prompt +
   modell + méret + pipeline-verzió. Így ha a keret fotóját kicserélik a
   katalógusban, az objektum-kulcs is megváltozik — nincs elavult cache.     */

const sha = (...parts: (string | Buffer)[]) => {
  const h = createHash("sha256");
  for (const p of parts) h.update(p);
  return h.digest("hex").slice(0, 16);
};

const stillCacheKey = (frameBytes: Buffer, portraitBytes: Buffer) =>
  sha(
    "still",
    String(PIPELINE.still.v),
    PIPELINE.still.model,
    PIPELINE.still.size,
    PIPELINE.still.quality,
    PIPELINE.still.prompt,
    frameBytes,
    portraitBytes,
  );

const videoCacheKey = (stillBytes: Buffer) =>
  sha(
    "video",
    String(PIPELINE.video.v),
    PIPELINE.video.model,
    PIPELINE.video.resolution,
    String(PIPELINE.video.duration),
    PIPELINE.video.prompt,
    stillBytes,
  );

const keys = (p: Pair, kind: "still" | "video" | "manifest", cacheKey?: string) => {
  const base = `catalog/${p.frame.id}/${p.archetype.id}`;
  if (kind === "still") return `${base}/still-${cacheKey}.png`;
  if (kind === "video") return `${base}/clip-${cacheKey}.mp4`;
  return `${base}/manifest.json`;
};

const cdnUrl = (key: string) => `${CDN}/${key}`;

/* ─── S3 segédek ──────────────────────────────────────────────────────────── */

async function head(key: string) {
  try {
    const r = await s3.send(new HeadObjectCommand({ Bucket: BUCKET, Key: key }));
    return { exists: true, size: Number(r.ContentLength ?? 0) };
  } catch (e: any) {
    if (e?.$metadata?.httpStatusCode === 404 || e?.name === "NotFound") return { exists: false, size: 0 };
    throw e;
  }
}

async function put(key: string, body: Buffer, contentType: string) {
  await s3.send(
    new PutObjectCommand({
      Bucket: BUCKET,
      Key: key,
      Body: body,
      ContentType: contentType,
      CacheControl: "public, max-age=31536000, immutable", // a kulcs tartalom-hash-elt
    }),
  );
  return cdnUrl(key);
}

async function getBuffer(key: string) {
  const r = await s3.send(new GetObjectCommand({ Bucket: BUCKET, Key: key }));
  return Buffer.from(await r.Body!.transformToByteArray());
}

/* ─── hálózat / retry ─────────────────────────────────────────────────────── */

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

async function withRetry<T>(label: string, fn: () => Promise<T>, tries = 4): Promise<T> {
  let lastErr: unknown;
  for (let i = 0; i < tries; i++) {
    try {
      return await fn();
    } catch (e: any) {
      lastErr = e;
      const status = e?.status ?? e?.response?.status;
      if (status && status >= 400 && status < 500 && status !== 429) throw e; // nem átmeneti
      const wait = Math.min(30_000, 1_500 * 2 ** i) + Math.random() * 500;
      console.warn(`  ↻ ${label} hiba (${status ?? e?.message}), újra ${Math.round(wait)}ms után`);
      await sleep(wait);
    }
  }
  throw lastErr;
}

const fetchBytes = (url: string) =>
  withRetry(`fetch ${url}`, async () => {
    const r = await fetch(url);
    if (!r.ok) throw Object.assign(new Error(`HTTP ${r.status} ${url}`), { status: r.status });
    return Buffer.from(await r.arrayBuffer());
  });

const toFile = (bytes: Buffer, name: string, type: string) =>
  new File([new Uint8Array(bytes)], name, { type });

/* ─── 1. lépés: try-on állókép (GPT Image 2 edit) ─────────────────────────── */

async function renderStill(p: Pair, portrait: Buffer, frameImg: Buffer, cacheKey: string) {
  const key = keys(p, "still", cacheKey);
  const hit = await head(key);
  if (hit.exists && !OPTS.force) {
    console.log(`  ✓ still cache-hit  ${key}`);
    return { key, url: cdnUrl(key), bytes: await getBuffer(key), cached: true };
  }

  const res = await withRetry(`gpt-image-2 ${p.frame.sku}/${p.archetype.id}`, () =>
    openai.images.edit({
      model: PIPELINE.still.model,
      // első kép a szerkesztendő jelenet (portré), második a referencia (keret)
      image: [
        toFile(portrait, "portrait.png", "image/png"),
        toFile(frameImg, "frame.png", "image/png"),
      ],
      prompt: PIPELINE.still.prompt,
      size: PIPELINE.still.size,
      quality: PIPELINE.still.quality,
      n: 1,
    }),
  );

  const b64 = res.data?.[0]?.b64_json;
  if (!b64) throw new Error("GPT Image 2: üres válasz");
  const bytes = Buffer.from(b64, "base64");
  const url = await put(key, bytes, "image/png");
  console.log(`  + still feltöltve  ${key} (${(bytes.length / 1024).toFixed(0)} kB)`);
  return { key, url, bytes, cached: false };
}

/* ─── 2. lépés: 6 mp-es klip (fal Hailuo-02 Fast, aszinkron queue) ───────── */

async function renderVideo(p: Pair, stillBytes: Buffer, cacheKey: string) {
  const key = keys(p, "video", cacheKey);
  const hit = await head(key);
  if (hit.exists && !OPTS.force) {
    console.log(`  ✓ video cache-hit  ${key}`);
    return { key, url: cdnUrl(key), cached: true };
  }

  // a still base64 data-URI-ként megy be, külön feltöltés nélkül
  const dataUri = `data:image/png;base64,${stillBytes.toString("base64")}`;

  const { request_id } = await withRetry(`fal submit ${p.frame.sku}/${p.archetype.id}`, () =>
    fal.queue.submit(PIPELINE.video.model, {
      input: {
        image_url: dataUri,
        prompt: PIPELINE.video.prompt,
        duration: PIPELINE.video.duration,
        resolution: PIPELINE.video.resolution,
        prompt_optimizer: false, // a prompt-kényszereket ne írja át
      },
    }),
  );

  // poll — a queue státuszt kérdezzük, nem streamelünk
  const deadline = Date.now() + 15 * 60_000;
  let status = "IN_QUEUE";
  while (status !== "COMPLETED") {
    if (Date.now() > deadline) throw new Error(`fal timeout: ${request_id}`);
    await sleep(5_000);
    const s = await withRetry(`fal status ${request_id}`, () =>
      fal.queue.status(PIPELINE.video.model, { requestId: request_id }),
    );
    status = s.status;
    if (status === "FAILED" as string) throw new Error(`fal FAILED: ${request_id}`);
  }

  const result: any = await fal.queue.result(PIPELINE.video.model, { requestId: request_id });
  const falUrl: string | undefined = result?.data?.video?.url ?? result?.video?.url;
  if (!falUrl) throw new Error(`fal: nincs videó URL (${request_id})`);

  // a fal CDN efemer — átmásoljuk a saját bucketbe
  const bytes = await fetchBytes(falUrl);
  const url = await put(key, bytes, "video/mp4");
  console.log(`  + klip feltöltve   ${key} (${(bytes.length / 1024 / 1024).toFixed(1)} MB)`);
  return { key, url, cached: false };
}

/* ─── egy pár teljes feldolgozása ─────────────────────────────────────────── */

async function processPair(p: Pair): Promise<Manifest> {
  const tag = `${p.frame.sku} × ${p.archetype.id}`;
  console.log(`▸ ${tag}`);

  const [frameImg, portrait] = await Promise.all([
    fetchBytes(p.frame.imageUrl),
    fetchBytes(p.archetype.portraitUrl),
  ]);

  const sKey = stillCacheKey(frameImg, portrait);
  const still = await renderStill(p, portrait, frameImg, sKey);

  let video: Manifest["video"];
  if (!OPTS.onlyStills) {
    const vKey = videoCacheKey(still.bytes);
    const v = await renderVideo(p, still.bytes, vKey);
    video = {
      key: v.key,
      url: v.url,
      cacheKey: vKey,
      pipeline: PIPELINE.video.v,
      durationSec: PIPELINE.video.duration,
    };
  }

  const manifest: Manifest = {
    frameId: p.frame.id,
    archetypeId: p.archetype.id,
    still: { key: still.key, url: still.url, cacheKey: sKey, pipeline: PIPELINE.still.v },
    video,
    generatedAt: new Date().toISOString(),
  };

  // a manifest mindig felülíródik: ez a "mi az aktuális" forrása a frontendnek
  await s3.send(
    new PutObjectCommand({
      Bucket: BUCKET,
      Key: keys(p, "manifest"),
      Body: JSON.stringify(manifest, null, 2),
      ContentType: "application/json",
      CacheControl: "public, max-age=60",
    }),
  );

  return manifest;
}

/* ─── futtató: korlátozott párhuzamosság, hibatűrés ───────────────────────── */

async function pool<T, R>(items: T[], limit: number, worker: (item: T) => Promise<R>) {
  const results: { item: T; ok: boolean; value?: R; error?: unknown }[] = [];
  let cursor = 0;
  await Promise.all(
    Array.from({ length: Math.min(limit, items.length) }, async () => {
      while (cursor < items.length) {
        const item = items[cursor++];
        try {
          results.push({ item, ok: true, value: await worker(item) });
        } catch (error) {
          results.push({ item, ok: false, error });
          console.error(`  ✗ hiba:`, (error as Error)?.message ?? error);
        }
      }
    }),
  );
  return results;
}

async function main() {
  const frames: Frame[] = JSON.parse(await readFile("data/frames.json", "utf8"));
  const archetypes: Archetype[] = JSON.parse(await readFile("data/archetypes.json", "utf8"));

  const fs_ = OPTS.frames ? frames.filter((f) => OPTS.frames!.includes(f.sku) || OPTS.frames!.includes(f.id)) : frames;
  const as_ = OPTS.archetypes ? archetypes.filter((a) => OPTS.archetypes!.includes(a.id)) : archetypes;

  const pairs: Pair[] = fs_.flatMap((frame) => as_.map((archetype) => ({ frame, archetype })));

  console.log(
    `${fs_.length} keret × ${as_.length} archetípus = ${pairs.length} pár · ` +
      `concurrency ${OPTS.concurrency}${OPTS.force ? " · FORCE" : ""}${OPTS.onlyStills ? " · csak still" : ""}`,
  );

  if (OPTS.dryRun) {
    for (const p of pairs) console.log(`  · ${p.frame.sku} × ${p.archetype.id}`);
    return;
  }

  const started = Date.now();
  const results = await pool(pairs, OPTS.concurrency, processPair);
  const failed = results.filter((r) => !r.ok);

  console.log(
    `\nkész: ${results.length - failed.length}/${pairs.length} pár, ` +
      `${((Date.now() - started) / 60_000).toFixed(1)} perc`,
  );
  if (failed.length) {
    console.log("sikertelen párok (újrafuttatható, a cache-hitek nem gyártódnak újra):");
    for (const f of failed) console.log(`  ${(f.item as Pair).frame.sku} × ${(f.item as Pair).archetype.id}`);
    process.exitCode = 1;
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
