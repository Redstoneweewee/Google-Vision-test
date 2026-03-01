import 'dotenv/config';

import vision, { protos } from '@google-cloud/vision';
import sharp from 'sharp';
import fs from 'fs';
import path from 'path';

import { reconstructLines } from './algorithm';
import type { ReconstructionResult } from './algorithm';
import type { ReceiptLine, Point } from './utils';
import { rotatePoint } from './utils';

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
 * Builds a rotated bounding box for a line:
 *   1. Collect all original vertices from every word in the line.
 *   2. Rotate them by −θ into the straightened coordinate system.
 *   3. Compute the axis-aligned bounding box in that space.
 *   4. Rotate the 4 AABB corners back by +θ into image space.
 *
 * The result is a tilted rectangle that follows the text baseline.
 */
function mergeRotatedBoundingBox(
  lineWords: TextAnnotation[],
  angle: number,
): LineAnnotation['boundingPoly'] {
  const allVerts = lineWords.flatMap((w) =>
    (w.boundingPoly?.vertices ?? []).map((v) => ({
      x: v.x ?? 0,
      y: v.y ?? 0,
    })),
  );

  if (allVerts.length === 0) {
    const zero = { x: 0, y: 0 } as Required<Vertex>;
    return { vertices: [zero, zero, zero, zero] };
  }

  // Rotate into straightened space
  const rotated = allVerts.map((v) => rotatePoint(v, -angle));
  const rxs = rotated.map((v) => v.x);
  const rys = rotated.map((v) => v.y);
  const minX = Math.min(...rxs);
  const maxX = Math.max(...rxs);
  const minY = Math.min(...rys);
  const maxY = Math.max(...rys);

  // AABB corners in rotated space (TL, TR, BR, BL)
  const corners: Point[] = [
    { x: minX, y: minY },
    { x: maxX, y: minY },
    { x: maxX, y: maxY },
    { x: minX, y: maxY },
  ];

  // Rotate back into image space
  const imageCorners = corners.map((c) => rotatePoint(c, angle));

  return {
    vertices: imageCorners.map((c) => ({
      x: Math.round(c.x),
      y: Math.round(c.y),
    })) as Required<Vertex>[],
  };
}

/**
 * Produce a short label for the annotated-image overlay.
 */
function formatLineLabel(line: ReceiptLine): string {
  if (line.lineType === 'wrapped') return `\u21a9 ${line.text}`;
  if (line.price) {
    if (line.lineType === 'untaxed_item' || line.lineType === 'taxed_item')
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
  const { lines: receiptLines, angle } = reconstructLines(detections.slice(1));

  console.log(`\nEstimated rotation: ${(angle * 180 / Math.PI).toFixed(2)}°\n`);

  // Build line annotations with per-line rotated bounding boxes
  const lineAnnotations: LineAnnotation[] = receiptLines.map((line) => ({
    description: formatLineLabel(line),
    boundingPoly: mergeRotatedBoundingBox(
      line.words.map((w) => w.original),
      line.angle,
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
  const { lines: receiptLines, angle } = reconstructLines(detections.slice(1));

  console.log(`\nEstimated rotation: ${(angle * 180 / Math.PI).toFixed(2)}°\n`);

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
