import 'dotenv/config';

import vision, { protos } from '@google-cloud/vision';
import sharp from 'sharp';
import fs from 'fs';
import path from 'path';

type TextAnnotation = protos.google.cloud.vision.v1.IEntityAnnotation;
type Vertex = protos.google.cloud.vision.v1.IVertex;

// Creates a client
// Credentials are read automatically from GOOGLE_APPLICATION_CREDENTIALS env var.
const client = new vision.ImageAnnotatorClient();

interface WordBounds {
  left: number;
  top: number;
  bottom: number;
  midY: number;
  height: number;
}

interface LineAnnotation {
  description: string;
  boundingPoly: { vertices: Required<Vertex>[] };
}

function getWordBounds(word: TextAnnotation): WordBounds {
  const verts = word.boundingPoly?.vertices ?? [];
  const xs = verts.map((v) => v.x ?? 0);
  const ys = verts.map((v) => v.y ?? 0);
  const top = Math.min(...ys);
  const bottom = Math.max(...ys);
  return {
    left: Math.min(...xs),
    top,
    bottom,
    midY: (top + bottom) / 2,
    height: bottom - top,
  };
}

/**
 * Groups word annotations into lines (top-to-bottom), each line sorted L→R.
 */
function groupIntoLines(words: TextAnnotation[]): TextAnnotation[][] {
  // STUB: each word is its own "line"
  return words.map((w) => [w]);
}

/**
 * Merges all word bounding boxes in a line into one axis-aligned rectangle.
 */
function mergeLineBoundingBox(
  lineWords: TextAnnotation[]
): LineAnnotation['boundingPoly'] {
  // STUB: return the single word's own bounding poly
  const verts = lineWords[0]?.boundingPoly?.vertices ?? [];
  return {
    vertices: verts.map((v) => ({ x: v.x ?? 0, y: v.y ?? 0 })) as Required<Vertex>[],
  };
}

/**
 * Draws per-line bounding boxes and labels onto the image and saves it.
 */
async function saveAnnotatedImage(
  inputPath: string,
  lineAnnotations: LineAnnotation[]
): Promise<void> {
  const meta = await sharp(inputPath).metadata();
  const { width, height } = meta;

  const polygons = lineAnnotations
    .map(({ description, boundingPoly: { vertices } }) => {
      const pts = vertices.map((v) => `${v.x},${v.y}`).join(' ');
      const lx = vertices[0].x ?? 0;
      const ly = vertices[0].y ?? 0;
      const label = description
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
      return `
      <polygon points="${pts}"
        fill="none" stroke="#00FF00" stroke-width="2"/>
      <text x="${lx}" y="${Math.max(ly - 4, 12)}"
        font-family="monospace" font-size="13"
        fill="#00FF00" stroke="black" stroke-width="0.6"
        paint-order="stroke">${label}</text>`;
    })
    .join('\n');

  const svg = `<svg xmlns="http://www.w3.org/2000/svg"
    width="${width}" height="${height}">
    ${polygons}
  </svg>`;

  const ext = path.extname(inputPath);
  const base = path.basename(inputPath, ext);
  const dir = path.dirname(inputPath);
  const outPath = path.join(dir, `${base}_annotated${ext}`);

  await sharp(inputPath)
    .composite([{ input: Buffer.from(svg), top: 0, left: 0 }])
    .toFile(outPath);

  console.log(`\nAnnotated image saved to: ${outPath}`);
}

/**
 * Detects text in a local image file, logs lines L→R, and saves an annotated copy.
 */
async function detectTextLocal(filePath: string): Promise<void> {
  const resolvedPath = path.resolve(filePath);

  if (!fs.existsSync(resolvedPath)) {
    console.error(`File not found: ${resolvedPath}`);
    process.exit(1);
  }

  console.log(`\nDetecting text in local image: ${resolvedPath}\n`);

  const [result] = await client.textDetection(resolvedPath);
  const detections = result.textAnnotations;

  if (!detections || detections.length === 0) {
    console.log('No text detected.');
    return;
  }

  console.log('=== Full detected text ===');
  console.log(detections[0].description);

  // Group words into lines, merge bounding boxes
  const lines = groupIntoLines(detections.slice(1));
  const lineAnnotations: LineAnnotation[] = lines.map((lineWords) => ({
    description: lineWords.map((w) => w.description ?? '').join(' '),
    boundingPoly: mergeLineBoundingBox(lineWords),
  }));

  console.log('\n=== Lines (L→R, merged bounding box) ===');
  lineAnnotations.forEach(({ description, boundingPoly: { vertices: v } }, i) => {
    const box = `(${v[0].x},${v[0].y}) → (${v[1].x},${v[1].y}) → (${v[2].x},${v[2].y}) → (${v[3].x},${v[3].y})`;
    console.log(`Line ${i + 1}: "${description}"`);
    console.log(`         Box: [${box}]`);
  });

  await saveAnnotatedImage(resolvedPath, lineAnnotations);
}

/**
 * Detects text in a remote image (Cloud Storage URI or public HTTPS URL).
 */
async function detectTextRemote(imageUri: string): Promise<void> {
  console.log(`\nDetecting text in remote image: ${imageUri}\n`);

  const [result] = await client.textDetection({
    image: { source: { imageUri } },
  });
  const detections = result.textAnnotations;

  if (!detections || detections.length === 0) {
    console.log('No text detected.');
    return;
  }

  console.log('=== Full detected text ===');
  console.log(detections[0].description);

  const lines = groupIntoLines(detections.slice(1));
  const lineAnnotations: LineAnnotation[] = lines.map((lineWords) => ({
    description: lineWords.map((w) => w.description ?? '').join(' '),
    boundingPoly: mergeLineBoundingBox(lineWords),
  }));

  console.log('\n=== Lines (L→R, merged bounding box) ===');
  lineAnnotations.forEach(({ description, boundingPoly: { vertices: v } }, i) => {
    const box = `(${v[0].x},${v[0].y}) → (${v[1].x},${v[1].y}) → (${v[2].x},${v[2].y}) → (${v[3].x},${v[3].y})`;
    console.log(`Line ${i + 1}: "${description}"`);
    console.log(`         Box: [${box}]`);
  });
}

// ── Entry point ───────────────────────────────────────────────────────────────
(async () => {
  const mode = process.argv[2] ?? 'remote';

  if (mode === 'local') {
    // Usage: npx ts-node src/detect.ts local <path-to-image>
    const filePath = process.argv[3] ?? './sample.jpg';
    await detectTextLocal(filePath);
  } else {
    // Usage: npx ts-node src/detect.ts remote <image-uri>
    const imageUri =
      process.argv[3] ?? 'gs://cloud-samples-data/vision/ocr/sign.jpg';
    await detectTextRemote(imageUri);
  }
})().catch((err: Error) => {
  console.error('Error:', err.message ?? err);
  process.exit(1);
});
