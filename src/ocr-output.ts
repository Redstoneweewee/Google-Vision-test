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
export async function detectText(filePath: string) {
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

  const detections = result.textAnnotations;
  if (!detections || detections.length === 0) {
    throw new Error('No text detected in image.');
  }
  const { lines: receiptLines } = reconstructLines(detections.slice(1));

  const annotatedLines = receiptLines.map((line) => {
    const left   = Math.min(...line.words.map((w) => w.left));
    const top    = Math.min(...line.words.map((w) => w.top));
    const right  = Math.max(...line.words.map((w) => w.right));
    const bottom = Math.max(...line.words.map((w) => w.bottom));
    return {
      text: line.text,
      boundingBox: { left, top, right, bottom },
      center: { x: (left + right) / 2, y: (top + bottom) / 2 },
    };
  });

  console.log(`Reconstructed lines:\n${JSON.stringify(annotatedLines, null, 2)}\n\n`);
}


const filePath = process.argv[2] ?? './sample.jpg';
detectText(filePath);

