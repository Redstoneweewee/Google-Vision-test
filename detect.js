'use strict';

require('dotenv').config();

const vision = require('@google-cloud/vision');
const sharp = require('sharp');
const fs = require('fs');
const path = require('path');

// Creates a client
// If GOOGLE_APPLICATION_CREDENTIALS env var is set, it will be used automatically.
// Otherwise, you can pass keyFilename: process.env.KEY_FILE to the constructor.
const client = new vision.ImageAnnotatorClient();

/**
 * Draws bounding boxes and labels onto the image and saves it.
 * @param {string} inputPath   - Path to the source image.
 * @param {Array}  annotations - textAnnotations array from Vision API.
 */
async function saveAnnotatedImage(inputPath, annotations) {
  const meta = await sharp(inputPath).metadata();
  const { width, height } = meta;

  // Skip the first annotation (it is the full-text block)
  const words = annotations.slice(1);

  // Build SVG overlay: one polygon + label per word
  const polygons = words.map(({ description, boundingPoly }) => {
    const pts = boundingPoly.vertices
      .map(v => `${v.x || 0},${v.y || 0}`)
      .join(' ');
    const lx = boundingPoly.vertices[0].x || 0;
    const ly = boundingPoly.vertices[0].y || 0;
    // Escape XML special chars in the label
    const label = description
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
    return `
      <polygon points="${pts}"
        fill="none" stroke="#00FF00" stroke-width="2"/>
      <text x="${lx}" y="${Math.max(ly - 3, 10)}"
        font-family="monospace" font-size="11"
        fill="#00FF00" stroke="black" stroke-width="0.5"
        paint-order="stroke">${label}</text>`;
  }).join('\n');

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
 * Detects text in a local image file and saves an annotated copy.
 * @param {string} filePath - Path to the local image file.
 */
async function detectTextLocal(filePath) {
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

  // The first annotation contains the full detected text
  console.log('=== Full detected text ===');
  console.log(detections[0].description);

  console.log('\n=== Individual word detections ===');
  detections.slice(1).forEach((text, i) => {
    const vertices = text.boundingPoly.vertices
      .map(v => `(${v.x || 0}, ${v.y || 0})`)
      .join(', ');
    console.log(`Word ${i + 1}: "${text.description}" — Bounding box: [${vertices}]`);
  });

  await saveAnnotatedImage(resolvedPath, detections);
}

/**
 * Detects text in a remote image (Cloud Storage URI or public HTTPS URL).
 * @param {string} imageUri - gs://... or https://... URI of the image.
 */
async function detectTextRemote(imageUri) {
  console.log(`\nDetecting text in remote image: ${imageUri}\n`);

  const [result] = await client.textDetection({ image: { source: { imageUri } } });
  const detections = result.textAnnotations;

  if (!detections || detections.length === 0) {
    console.log('No text detected.');
    return;
  }

  console.log('=== Full detected text ===');
  console.log(detections[0].description);

  console.log('\n=== Individual word detections ===');
  detections.slice(1).forEach((text, i) => {
    const vertices = text.boundingPoly.vertices
      .map(v => `(${v.x || 0}, ${v.y || 0})`)
      .join(', ');
    console.log(`Word ${i + 1}: "${text.description}" — Bounding box: [${vertices}]`);
  });
}

// ── Entry point ──────────────────────────────────────────────────────────────
(async () => {
  const mode = process.argv[2] || 'remote';

  if (mode === 'local') {
    // Usage: node detect.js local <path-to-image>
    const filePath = process.argv[3] || './sample.jpg';
    await detectTextLocal(filePath);
  } else {
    // Usage: node detect.js remote <image-uri>
    // Defaults to the Google sample sign image.
    const imageUri =
      process.argv[3] || 'gs://cloud-samples-data/vision/ocr/sign.jpg';
    await detectTextRemote(imageUri);
  }
})().catch(err => {
  console.error('Error:', err.message || err);
  process.exit(1);
});
