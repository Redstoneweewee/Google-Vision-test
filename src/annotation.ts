/**
 * Image annotation helpers — draws per-line bounding boxes onto
 * receipt images for visual debugging.
 */

import { protos } from '@google-cloud/vision';
import sharp from 'sharp';
import path from 'path';
import type { WordBox, ReceiptLine, LineType } from './types';

type Vertex = protos.google.cloud.vision.v1.IVertex;

export interface LineAnnotation {
  description: string;
  lineType: LineType;
  boundingPoly: { vertices: Required<Vertex>[] };
}

/**
 * Builds a polygon for a line by tracing the bounding boxes of each word:
 *   - Top edges from left to right (TL → TR for each word)
 *   - Bottom edges from right to left (BR → BL for each word)
 *
 * This produces a tight outline that follows each word's actual OCR box
 * rather than a single merged rectangle.
 */
export function buildLinePolygon(
  words: WordBox[],
): LineAnnotation['boundingPoly'] {
  if (words.length === 0) {
    const zero = { x: 0, y: 0 } as Required<Vertex>;
    return { vertices: [zero, zero, zero, zero] };
  }

  // Sort words left to right
  const sorted = [...words].sort((a, b) => a.left - b.left);

  const topPath: Required<Vertex>[] = [];
  const bottomReversed: Required<Vertex>[] = [];

  for (const word of sorted) {
    const verts = (word.original.boundingPoly?.vertices ?? []).map((v) => ({
      x: v.x ?? 0,
      y: v.y ?? 0,
    })) as Required<Vertex>[];

    if (verts.length >= 4) {
      // Vertices: [TL(0), TR(1), BR(2), BL(3)]
      topPath.push(verts[0], verts[1]);
      bottomReversed.push(verts[3], verts[2]);
    }
  }

  // Reverse the bottom path so it goes right → left
  bottomReversed.reverse();

  return { vertices: [...topPath, ...bottomReversed] };
}

/**
 * Produce a short label for the annotated-image overlay.
 */
export function formatLineLabel(line: ReceiptLine): string {
  if (line.lineType === 'wrapped') return `\u21a9 ${line.text}`;
  if (line.price) {
    if (line.lineType === 'untaxed_item' || line.lineType === 'taxed_item')
      return `${line.itemName ?? ''} \u2192 ${line.price}`;
    return `[${line.lineType.toUpperCase()}] ${line.itemName ?? ''} ${line.price}`;
  }
  return line.text;
}

/**
 * Draws per-line bounding boxes onto the image and saves it as
 * `{basename}_annotated{ext}` alongside the original.
 */
export async function saveAnnotatedImage(
  inputPath: string,
  imageBuffer: Buffer,
  lineAnnotations: LineAnnotation[]
): Promise<void> {
  const meta = await sharp(imageBuffer).metadata();
  const { width, height } = meta;

  const polygons = lineAnnotations
    .map(({ lineType, boundingPoly: { vertices } }) => {
      const pts = vertices.map((v) => `${v.x},${v.y}`).join(' ');
      const color = (lineType === 'info' || lineType === 'tender') ? '#888888' : '#00FF00';
      return `
      <polygon points="${pts}"
        fill="none" stroke="${color}" stroke-width="2"/>`;
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

  await sharp(imageBuffer)
    .composite([{ input: Buffer.from(svg), top: 0, left: 0 }])
    .toFile(outPath);

  console.log(`\nAnnotated image saved to: ${outPath}`);
}
