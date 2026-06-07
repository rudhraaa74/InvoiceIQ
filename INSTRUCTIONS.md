# Invoice Data Extraction Pipeline
### Qwen3.5-9B · Kaggle GPU · Antigravity MCP

---

## Overview

### Workflow Rules
- **Constant Explanations:** I must explicitly explain what I am doing and *why* I am doing it at every step.
- **Interactive First:** All code updates must first be written locally to `dataextractor.ipynb`. The user will run the code interactively and verify it. I will ONLY push a new version to Kaggle when explicitly instructed to "push it".

This pipeline extracts structured header information from invoices (PDF + images) at scale using Qwen3.5-9B's vision-language capabilities, running on Kaggle's free T4 GPU compute, orchestrated from Antigravity via the Kaggle MCP.

```
Invoices (PDF + Images)
        │
        ▼
┌─────────────────┐
│  Stage 1        │  Environment Setup
│  Setup          │  Install deps, load model
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Stage 2        │  Pre-processing
│  Ingestion      │  PDF → images, normalize
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Stage 3        │  Extraction
│  Inference      │  Qwen3.5-9B vision pipeline
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Stage 4        │  Post-processing
│  Parsing        │  JSON cleanup + error handling
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Stage 5        │  Output
│  Validation     │  Field checks, JSONL output
└─────────────────┘
```

---

## Stage 1 — Environment Setup

### 1.1 Kaggle Notebook Configuration

Before running any code, configure your Kaggle notebook:

| Setting | Value |
|---|---|
| Accelerator | GPU T4 x2 |
| Internet | ON (required for model download) |
| Persistence | Files + Variables |

> **GPU Quota reminder:** The 30-hour weekly quota counts wall-clock time from the moment GPU is enabled, not just active computation. Switch accelerator back to **None** when writing/debugging code.

### 1.2 Install Dependencies

```python
# Cell 1 — run once per session
!pip install -q transformers accelerate pdf2image pillow
!apt-get install -q poppler-utils   # required by pdf2image for PDF rendering
```

### 1.3 Verify GPU

```python
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
```

Expected output:
```
CUDA available: True
GPU count: 2
  GPU 0: Tesla T4
  GPU 1: Tesla T4
```

### 1.4 Load Model Pipeline

```python
from transformers import pipeline

MODEL_ID = "Qwen/Qwen3.5-9B"

print("Loading Qwen3.5-9B...")
pipe = pipeline(
    "image-text-to-text",
    model=MODEL_ID,
    device_map="auto",      # auto-distributes across both T4s
    torch_dtype="auto"
)
print("Model ready.")
```

> `device_map="auto"` spreads the 9B model across both T4 GPUs (16 GB VRAM each = 32 GB total), which comfortably fits the model in float16.

---

## Stage 2 — File Ingestion & Pre-processing

### 2.1 Discover All Invoice Files

```python
import glob
from pathlib import Path

INPUT_DIR = "/kaggle/input/invoices"

all_files = (
    glob.glob(f"{INPUT_DIR}/**/*.pdf",  recursive=True) +
    glob.glob(f"{INPUT_DIR}/**/*.png",  recursive=True) +
    glob.glob(f"{INPUT_DIR}/**/*.jpg",  recursive=True) +
    glob.glob(f"{INPUT_DIR}/**/*.jpeg", recursive=True)
)

print(f"Found {len(all_files)} invoice files")

# Quick breakdown by type
from collections import Counter
type_counts = Counter(Path(f).suffix.lower() for f in all_files)
for ext, count in type_counts.items():
    print(f"  {ext}: {count} files")
```

### 2.2 PDF → Image Conversion

PDFs are rendered page-by-page into PIL images before being passed to the model. Each page is treated as a separate invoice candidate.

```python
from pdf2image import convert_from_path
from PIL import Image

def pdf_to_images(pdf_path: str, dpi: int = 200) -> list:
    """
    Render each page of a PDF to a PIL Image.
    DPI 200 gives a good balance of quality vs. memory.
    Higher DPI (300) improves accuracy on small text but increases VRAM use.
    """
    return convert_from_path(pdf_path, dpi=dpi)

def load_image(image_path: str) -> Image.Image:
    """Load and normalize an image file to RGB."""
    return Image.open(image_path).convert("RGB")
```

### 2.3 Image → Base64 URL

The `pipeline` format requires a URL in the `"url"` field. Since invoices are local files, convert each image to an inline base64 data URL.

```python
import base64
from io import BytesIO

def image_to_base64_url(img: Image.Image, quality: int = 90) -> str:
    """
    Convert a PIL image to a base64 data URL.
    Used because pipeline() expects a URL, not a file path.
    """
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"
```

---

## Stage 3 — Inference

### 3.1 Extraction Prompt

The prompt instructs the model to return only a JSON object with fixed keys. Strict JSON-only output minimises post-processing.

```python
HEADER_PROMPT = """Extract the following header fields from this invoice.
Return ONLY a valid JSON object with these exact keys, no extra text:
{
  "invoice_number": "",
  "invoice_date": "",
  "due_date": "",
  "vendor_name": "",
  "vendor_address": "",
  "bill_to_name": "",
  "bill_to_address": "",
  "total_amount": "",
  "currency": "",
  "po_number": ""
}
If a field is not present in the invoice, use null."""
```

**Fields extracted:**

| Field | Description |
|---|---|
| `invoice_number` | Unique invoice identifier |
| `invoice_date` | Date the invoice was issued |
| `due_date` | Payment due date |
| `vendor_name` | Name of the issuing company |
| `vendor_address` | Full address of the vendor |
| `bill_to_name` | Name of the recipient/customer |
| `bill_to_address` | Full address of the recipient |
| `total_amount` | Final total (numeric string) |
| `currency` | Currency code or symbol |
| `po_number` | Purchase order number if present |

### 3.2 Single Invoice Extraction

```python
def extract_fields(img: Image.Image, source_file: str, page: int) -> dict:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": image_to_base64_url(img)},
                {"type": "text",  "text": HEADER_PROMPT}
            ]
        }
    ]

    result = pipe(text=messages, max_new_tokens=512)

    # pipeline returns full conversation — grab last assistant turn
    raw = result[0]["generated_text"]
    if isinstance(raw, list):
        raw = raw[-1]["content"]

    return raw, source_file, page
```

---

## Stage 4 — Output Parsing & Error Handling

### 4.1 JSON Cleanup

Models sometimes wrap output in markdown fences (` ```json `) even when instructed not to. Strip these before parsing.

```python
import json

def parse_output(raw: str, source_file: str, page: int) -> dict:
    """
    Parse the model's raw text output into a structured dict.
    Handles markdown fences and JSON decode errors gracefully.
    """
    clean = raw.strip()

    # Strip markdown code fences if present
    for fence in ["```json", "```"]:
        clean = clean.removeprefix(fence).removesuffix("```").strip()

    try:
        data = json.loads(clean)
    except json.JSONDecodeError:
        # Keep the raw output for manual review instead of silently dropping it
        data = {
            "raw_output": raw,
            "parse_error": True
        }

    # Attach source metadata
    data["_source_file"] = source_file
    data["_page"]        = page
    return data
```

### 4.2 Error Categories

| Error type | How it's handled |
|---|---|
| JSON parse failure | Saved with `parse_error: true` + raw output for review |
| PDF render failure | Logged with `error` field, file skipped |
| Image load failure | Logged with `error` field, file skipped |
| Model OOM | Reduce `dpi` from 200 → 150, or process PDFs one page at a time |

---

## Stage 5 — Batch Loop, Resumability & Output

### 5.1 Resume from Checkpoint

For 500+ invoices across Kaggle's 9-hour session limit, the loop checks what's already written to the JSONL output and skips those files. If the session resets, just rerun the cell — it picks up exactly where it left off.

```python
OUTPUT_FILE = "/kaggle/working/extracted.jsonl"

def load_processed(output_file: str) -> set:
    """Return a set of (source_file, page) tuples already in the output."""
    processed = set()
    if not os.path.exists(output_file):
        return processed
    with open(output_file) as f:
        for line in f:
            try:
                rec = json.loads(line)
                processed.add((rec["_source_file"], rec["_page"]))
            except:
                pass
    return processed
```

### 5.2 Full Batch Loop

```python
import os
from tqdm import tqdm

processed = load_processed(OUTPUT_FILE)
print(f"Resuming — {len(processed)} records already done, {len(all_files)} total files")

with open(OUTPUT_FILE, "a") as out:
    for filepath in tqdm(all_files):
        ext = Path(filepath).suffix.lower()
        try:
            images = pdf_to_images(filepath) if ext == ".pdf" \
                     else [load_image(filepath)]

            for page_num, img in enumerate(images):
                if (filepath, page_num) in processed:
                    continue

                raw, src, pg = extract_fields(img, filepath, page_num)
                record = parse_output(raw, src, pg)

                out.write(json.dumps(record) + "\n")
                out.flush()   # write immediately — never lose a completed record

        except Exception as e:
            out.write(json.dumps({
                "_source_file": filepath,
                "_page": -1,
                "error": str(e)
            }) + "\n")
            out.flush()

print(f"Done. Output → {OUTPUT_FILE}")
```

### 5.3 Output Format

Output is written as **JSONL** (newline-delimited JSON) — one record per line. This is better than a single JSON array for large batches because:

- Records are appended incrementally (no need to load the full file)
- Partial output is always valid and readable
- Easy to stream into pandas, BigQuery, or any downstream tool

**Sample output record:**
```json
{
  "invoice_number": "INV-2024-00421",
  "invoice_date": "2024-03-15",
  "due_date": "2024-04-15",
  "vendor_name": "Acme Supplies Ltd.",
  "vendor_address": "12 Industrial Park, Mumbai 400001",
  "bill_to_name": "TechCorp India Pvt. Ltd.",
  "bill_to_address": "45 MG Road, Bengaluru 560001",
  "total_amount": "84500.00",
  "currency": "INR",
  "po_number": "PO-2024-1187",
  "_source_file": "/kaggle/input/invoices/march/INV-00421.pdf",
  "_page": 0
}
```

---

## Stage 6 — Validation (Post-run)

Run this after the batch loop completes to check output quality.

```python
import pandas as pd

# Load JSONL into DataFrame
records = []
with open(OUTPUT_FILE) as f:
    for line in f:
        records.append(json.loads(line))

df = pd.DataFrame(records)

print(f"Total records: {len(df)}")
print(f"Parse errors:  {df.get('parse_error', pd.Series()).sum()}")
print(f"Runtime errors:{df['error'].notna().sum() if 'error' in df.columns else 0}")
print()

# Field coverage — % of records where each field was found
core_fields = [
    "invoice_number", "invoice_date", "due_date",
    "vendor_name", "bill_to_name", "total_amount", "currency"
]
print("Field coverage:")
for field in core_fields:
    if field in df.columns:
        filled = df[field].notna() & (df[field] != "null") & (df[field] != "")
        print(f"  {field:20s}: {filled.mean()*100:.1f}%")
```

---

## Workflow: Antigravity + Kaggle MCP

Rather than manually copy-pasting code and errors, use Antigravity's agent to manage the loop:

```
Antigravity Agent
      │
      │  (Kaggle MCP)
      ├─── push notebook version ──────────► Kaggle
      │                                        │
      │                                        ▼
      │                                   runs on T4 GPU
      │                                        │
      └─── fetch output / error logs ◄─────────┘
      │
      ▼
  Agent reads errors, fixes code, pushes again
```

**Recommended agent prompt pattern in Antigravity:**
```
Run the invoice extraction notebook on Kaggle. 
If it errors, fetch the logs, fix the issue, and push a new version. 
Repeat until the run completes successfully and extracted.jsonl is non-empty.
```

---

## GPU Quota Tips

- Only enable GPU when actually running inference — disable during setup/debugging
- Stop the session explicitly before closing the browser tab (saves up to 60 min of idle quota)
- Quota resets every **Saturday at midnight UTC**
- Link a Colab Pro subscription to Kaggle for an additional 15–30 GPU hours/week

---

## Run the invoice extraction notebook on Kaggle. 
If it errors, fetch the logs, fix the issue, and push a new version. 
Repeat until the run completes successfully and extracted.jsonl is non-empty.

## File Reference

| File | Purpose |
|---|---|
| `invoice_extraction.py` | Main pipeline script |
| `/kaggle/input/invoices/` | Input directory (upload your dataset here) |
| `/kaggle/working/extracted.jsonl` | Output — one JSON record per invoice |