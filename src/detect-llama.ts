/**
 * detect-llama.ts — Receipt OCR + Llama 3.2 post-processing pipeline.
 *
 * Uses the same Google Cloud Vision OCR step as detect.ts, then runs
 * the existing algorithmic line-reconstruction pipeline to produce
 * structured, layout-aware line groups.  That structured intermediate
 * representation — NOT the raw text dump — is sent to Llama 3.2
 * (via Ollama) for refined item extraction.
 *
 * Usage:
 *   npx ts-node src/detect-llama.ts <image-path>
 *
 * Environment:
 *   GOOGLE_API_KEY   — Google Cloud Vision API key (required)
 *   OLLAMA_BASE_URL  — Ollama endpoint (default: http://localhost:11434)
 *   OLLAMA_MODEL     — Model name (default: llama3.2)
 */

import 'dotenv/config';

import vision from '@google-cloud/vision';
import sharp from 'sharp';
import fs from 'fs';
import path from 'path';

import type { DebugReceipt } from './types';
import { MAX_IMAGE_DIMENSION } from './constants';
import { reconstructLines } from './algorithm';
import { buildLlamaInput, processReceiptWithLlama } from './llama-pipeline';
import { printReceipt } from './printing';

const client = new vision.ImageAnnotatorClient({
  apiKey: process.env.GOOGLE_API_KEY,
});

/**
 * Detects text in a local image file using Google Cloud Vision, runs the
 * algorithmic line-reconstruction pipeline, builds a structured intermediate
 * representation, and sends it to Llama 3.2 for refined parsing.
 *
 * Returns the same DebugReceipt shape as detectTextLocal in detect.ts.
 */
export async function detectTextLocalLlama(filePath: string): Promise<DebugReceipt> {
  const start = Date.now();
  const resolvedPath = path.resolve(filePath);

  if (!fs.existsSync(resolvedPath)) {
    throw new Error(`File not found: ${resolvedPath}`);
  }

  // ── Resize image if needed ──────────────────────────────────────────
  const imageBuffer = await sharp(resolvedPath)
    .resize({
      width: MAX_IMAGE_DIMENSION,
      height: MAX_IMAGE_DIMENSION,
      fit: 'inside',
      withoutEnlargement: true,
    })
    .toBuffer();

  const end1 = Date.now();

  // ── Google Cloud Vision OCR ─────────────────────────────────────────
  const [result] = await client.documentTextDetection({ image: { content: imageBuffer } });

  const end2 = Date.now();

  const detections = result.textAnnotations;
  if (!detections || detections.length === 0) {
    throw new Error('No text detected in image.');
  }

  console.log(`Full text: ${result.fullTextAnnotation?.text}\n\n`);

  // ── Algorithmic line reconstruction (reuse existing pipeline) ───────
  const { lines: receiptLines, angle } = reconstructLines(detections.slice(1));

  const end3 = Date.now();

  // ── Build minimal input for Llama (items/discounts only) ───────────
  const llamaInput = buildLlamaInput(receiptLines);

  const lineCount = llamaInput.split('\n').length;
  console.log(`Sending ${llamaInput} to Llama 3.2...\n`);

  // ── Llama 3.2 post-processing ─────────────────────────────────────
  const { receipt, llamaElapsed } = await processReceiptWithLlama(
    llamaInput,
    receiptLines,
    angle,
  );

  const end4 = Date.now();

  //console.log(`receipt:`, receipt);

  return {
    ...receipt,
    times: [
      { type: 'Image loaded and resized time', elapsed: end1 - start },
      { type: 'OCR Text detection time', elapsed: end2 - end1 },
      { type: 'Algorithmic line reconstruction time', elapsed: end3 - end2 },
      { type: 'Llama 3.2 post-processing time', elapsed: llamaElapsed },
    ],
  };
}

// ── Entry point ───────────────────────────────────────────────────────────────
(async () => {
  const filePath = process.argv[2] ?? './sample.jpg';
  const receipt = await detectTextLocalLlama(filePath);
  printReceipt(receipt);
})().catch((err: Error) => {
  console.error('Error:', err.message ?? err);
  process.exit(1);
});
