# Google Vision API — Image Text Detection (Node.js)

Uses the [Google Cloud Vision API](https://cloud.google.com/vision/docs/ocr) to detect and extract text from images via OCR.

---

## Prerequisites

1. **Node.js** ≥ 18  
2. A **Google Cloud project** with the [Cloud Vision API enabled](https://console.cloud.google.com/apis/library/vision.googleapis.com)  
3. A **service account key** with the Vision API role (or use Application Default Credentials)

---

## Setup

### 1. Clone / open the repo

```bash
cd "Google Vision test"
```

### 2. Install dependencies

```bash
npm install
```

### 3. Configure credentials

Copy the example env file and fill in your key path:

```bash
copy .env.example .env
```

Then edit `.env`:

```
GOOGLE_APPLICATION_CREDENTIALS=./service-account-key.json
```

Place your downloaded service account JSON file at that path.  
**Never commit this file** — it is already listed in `.gitignore`.

Alternatively, set the env var in your shell before running:

```powershell
$env:GOOGLE_APPLICATION_CREDENTIALS = "C:\path\to\key.json"
```

---

## Usage

### Detect text in a remote image (default — uses Google sample image)

```bash
node detect.js
# or
node detect.js remote
```

### Detect text in a specific remote image (Cloud Storage URI or public URL)

```bash
node detect.js remote gs://cloud-samples-data/vision/ocr/sign.jpg
node detect.js remote https://example.com/image.jpg
```

### Detect text in a local image

```bash
node detect.js local ./sample.jpg
node detect.js local C:\path\to\image.png
```

---

## Project structure

```
.
├── detect.js           # Main OCR script
├── package.json
├── .env.example        # Template for credentials env var
├── .gitignore
└── README.md
```

---

## Output example

```
Detecting text in remote image: gs://cloud-samples-data/vision/ocr/sign.jpg

=== Full detected text ===
WAITING?
PLEASE TURN OFF
YOUR IDLE ENGINE

=== Individual word detections ===
Word 1: "WAITING?" — Bounding box: [(50, 30), (230, 30), (230, 70), (50, 70)]
...
```

---

## References

- [Vision API OCR docs](https://cloud.google.com/vision/docs/ocr)
- [@google-cloud/vision npm package](https://www.npmjs.com/package/@google-cloud/vision)
- [Authentication setup](https://cloud.google.com/docs/authentication/set-up-adc-local-dev-environment)
