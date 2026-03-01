import 'dotenv/config';

import vision, { protos } from '@google-cloud/vision';
import sharp from 'sharp';
import fs from 'fs';
import path from 'path';

import { reconstructLines } from './algorithm';
import type { ReceiptLine } from './utils';

type TextAnnotation = protos.google.cloud.vision.v1.IEntityAnnotation;
type Vertex = protos.google.cloud.vision.v1.IVertex;

// Creates a client
// Credentials are read automatically from GOOGLE_APPLICATION_CREDENTIALS env var.
const client = new vision.ImageAnnotatorClient();

interface LineAnnotation {
  description: string;
  boundingPoly: { vertices: Required<Vertex>[] };
}

/**
 * Merges all word bounding boxes in a line into one axis-aligned rectangle.
 */
function mergeLineBoundingBox(
  lineWords: TextAnnotation[],
): LineAnnotation['boundingPoly'] {
  const allVerts = lineWords.flatMap((w) => w.boundingPoly?.vertices ?? []);
  if (allVerts.length === 0) {
    const zero = { x: 0, y: 0 } as Required<Vertex>;
    return { vertices: [zero, zero, zero, zero] };
  }

  const xs = allVerts.map((v) => v.x ?? 0);
  const ys = allVerts.map((v) => v.y ?? 0);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);

  return {
    vertices: [
      { x: minX, y: minY },
      { x: maxX, y: minY },
      { x: maxX, y: maxY },
      { x: minX, y: maxY },
    ] as Required<Vertex>[],
  };
}

/**
 * Produce a short label for the annotated-image overlay.
 */
function formatLineLabel(line: ReceiptLine): string {
  if (line.lineType === 'wrapped') return `\u21a9 ${line.text}`;
  if (line.price) {
    if (line.lineType === 'item')
      return `${line.itemName ?? ''} \u2192 ${line.price}`;
    return `[${line.lineType.toUpperCase()}] ${line.itemName ?? ''} ${line.price}`;
  }
  return line.text;
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

  // Reconstruct receipt lines using the hybrid algorithm
  const receiptLines = reconstructLines(detections.slice(1));

  // Build line annotations for the annotated-image overlay
  const lineAnnotations: LineAnnotation[] = receiptLines.map((line) => ({
    description: formatLineLabel(line),
    boundingPoly: mergeLineBoundingBox(
      line.words.map((w) => w.original),
    ),
  }));

  console.log('\n=== Reconstructed Lines ===');
  lineAnnotations.forEach(
    ({ description, boundingPoly: { vertices: v } }, i) => {
      const box = `(${v[0].x},${v[0].y}) \u2192 (${v[1].x},${v[1].y}) \u2192 (${v[2].x},${v[2].y}) \u2192 (${v[3].x},${v[3].y})`;
      console.log(`Line ${i + 1}: "${description}"`);
      console.log(`         Box: [${box}]`);
    },
  );

  console.log('\n=== Parsed Receipt ===');
  for (const line of receiptLines) {
    if (line.lineType === 'wrapped') continue;
    const tag = `[${line.lineType.toUpperCase()}]`.padEnd(12);
    const name = (line.itemName ?? line.text).padEnd(35);
    const price = line.price ?? '';
    console.log(`  ${tag} ${name} ${price}`);
  }

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

  // Reconstruct receipt lines using the hybrid algorithm
  const receiptLines = reconstructLines(detections.slice(1));

  console.log('\n=== Reconstructed Lines ===');
  receiptLines.forEach((line, i) => {
    console.log(`Line ${i + 1}: "${line.text}"`);
  });

  console.log('\n=== Parsed Receipt ===');
  for (const line of receiptLines) {
    if (line.lineType === 'wrapped') continue;
    const tag = `[${line.lineType.toUpperCase()}]`.padEnd(12);
    const name = (line.itemName ?? line.text).padEnd(35);
    const price = line.price ?? '';
    console.log(`  ${tag} ${name} ${price}`);
  }
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
