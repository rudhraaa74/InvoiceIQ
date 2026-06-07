# Invoice Extraction Pipeline: Complete Journey & Debugging Walkthrough

This document serves as a complete history of the Kaggle notebook pipeline construction for extracting structured data from invoices using the `Qwen3.5-9B` Vision-Language Model. It breaks down every error we encountered, why it happened, and how we solved it across all 10 notebook iterations.

## Part 1: Environment & Model Setup (Versions 1-4)

**The Goal:** Establish the environment on Kaggle, leverage 2x Tesla T4 GPUs, and load Qwen3.5-9B into VRAM to run a test inference.

> [!WARNING] 
> **Error 1: Missing Architecture (`KeyError: qwen2_5_vl`)**
> - **What happened:** Attempting to load the model threw an error stating the architecture was unrecognized. 
> - **Explanation:** `Qwen3.5-9B` is an extremely new model architecture. The stable release of the `transformers` library on PyPI did not yet support it.
> - **The Fix:** We updated Cell 1 to install `transformers` directly from the bleeding-edge GitHub source: `!pip install git+https://github.com/huggingface/transformers.git`.

> [!CAUTION]
> **Error 2: Catastrophic VRAM Overflow (`OutOfMemoryError`)**
> - **What happened:** Passing an original invoice/test image crashed the GPU immediately, attempting to allocate a massive 133 GiB for the attention matrix.
> - **Explanation:** Vision models scale their memory footprint quadratically with image resolution. High-resolution raw images produce thousands of visual tokens, exploding the VRAM requirements.
> - **The Fix:** We introduced a preprocessing step using PIL to cap image dimensions before sending them to the model: `image.thumbnail((1024, 1024))`.

> [!WARNING]
> **Error 3: Default Precision Overload**
> - **What happened:** Loading the 9B model natively exhausted VRAM.
> - **Explanation:** By default, models load in `float32` (4 bytes per parameter), making a 9B parameter model require 36GB of VRAM (more than a single T4's 16GB). 
> - **The Fix:** We added `device_map="auto"` to shard the model across both T4 GPUs, and forced half-precision loading using `dtype=torch.float16`.

**Result:** Version 4 successfully ran a test inference identifying an animal on a piece of candy.

---

## Part 2: Data Ingestion (Versions 5-7)

**The Goal:** Build Stage 2 to discover and ingest invoice files for processing.

- **Version 5:** Implemented local file discovery using `glob` to scan `/kaggle/input/invoices` and added `pdf2image` capabilities for multi-page documents.
- **The Pivot:** We decided to abandon local file scanning and instead pull data directly from the Hugging Face Hub (`amaye15/invoices-google-ocr`).

> [!NOTE]
> **Cosmetic Issue: Red PIP Conflicts**
> - **What happened:** Installing our dependencies caused Kaggle to throw massive red dependency conflict errors in the output block.
> - **Explanation:** Kaggle comes pre-installed with strict data science libraries (`dask-cuda`, `cudf`). Our newer packages updated base libraries that technically broke those pre-installed dependencies.
> - **The Fix (Version 6):** Since we don't use `dask` or `cudf`, this was harmless. We added the `%%capture` magic command to Cell 1 to completely suppress the scary output. We also updated a deprecation warning (`torch_dtype` -> `dtype`).

- **Version 7:** Fully integrated the Hugging Face `datasets` library, dynamically identifying the image column in the dataset and streaming PIL images directly into memory.

---

## Part 3: Inference Logic & Hallucination Taming (Versions 8-10)

**The Goal:** Feed the dataset images into the model and extract clean, predictable JSON data.

- **Version 8:** Added the `HEADER_PROMPT` targeting 10 specific invoice fields, and built the `extract_fields()` function.
- **Version 9:** Executed a full pipeline run against the first image in the Hugging Face dataset to test the extraction logic.

> [!IMPORTANT]
> **Error 4: Conversational Truncation (No JSON!)**
> - **What happened:** The model successfully processed the invoice but completely failed to output a JSON object. Instead, it started generating a long essay (Chain-of-Thought): *"Based on the visual content of the invoice, I need to locate and extract specific fields..."*. It eventually hit the `max_new_tokens=512` limit and got cut off mid-sentence.
> - **Explanation:** Large Language Models naturally want to explain their reasoning. If we don't strictly forbid it, they will waste their generation budget on text, and any downstream JSON parser (Stage 4) will crash with a `JSONDecodeError`.
> - **The Fix (Version 10):** 
>   1. We aggressively re-engineered the prompt: *"Do NOT include any explanations, reasoning, or conversational text. Start your response with { and end with }."*
>   2. We doubled the generation budget (`max_new_tokens=1024`) to ensure even large, complex JSON structures don't get truncated.

## Next Steps
With the core generation pipeline secured and generating strict JSON, the notebook is perfectly positioned for **Stage 4 & Stage 5**: deploying the batch processing loop over the dataset and writing the final results securely to `extracted.jsonl`.
