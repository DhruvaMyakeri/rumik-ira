# Ira - Hinglish Companion AI

Ira is a companion AI built on Gemma 3 12B, fine-tuned to have conversations the way young Indians actually text - natural Hinglish, warm but not soft, funny without trying too hard.

---

## What's in this repo

```
README.md
final_report.md
serve/
    serve.py          - FastAPI inference service (Modal.com)
    chat.py           - CLI chat client
training/
    train_sft_modal.py  - SFT training script
    data_prep.py        - Dataset preparation and formatting
preference/
    train_dpo_modal.py  - DPO training script
    dpo_pairs_v2.json   - 41 hand-curated preference pairs
    dpo_combined.jsonl  - 204 total preference pairs
evals/
    eval_suite.py       - Evaluation script (base Gemma vs SFT Ira)
    eval_results.json   - Full eval output
results/
    sft_training_logs.md
```

---

## Setup

You need:

- Modal.com account with GPU access
- Groq API key (for dataset generation)
- HuggingFace token (for Gemma access)

```bash
pip install modal
modal setup
modal secret create huggingface-secret HF_TOKEN=your_token_here
```

---

## Training

**SFT:**

```bash
# prepare dataset
python training/data_prep.py

# upload data and train
modal run training/train_sft_modal.py
```

**DPO (optional, currently degrades quality - see report):**

```bash
modal run preference/train_dpo_modal.py
```

---

## Inference Service

**Deploy:**

```bash
modal deploy serve/serve.py
```

**Chat:**

```bash
pip install requests
python serve/chat.py --url https://your-modal-url
```

**With memory context:**

```bash
python serve/chat.py --url https://your-modal-url --memory "user has exam tomorrow"
```

**Send an image:**

```
You: /image path/to/photo.jpg
[Image description: ...]
Image ready. Now type your message.
You: dekh yeh banaya maine aaj
Ira: yaar edges thodi burnt hain par honestly still looks good...
```

---

## API

`POST /chat`

```json
{
  "messages": [
    { "role": "user", "content": "yaar aaj kuch acha nahi lag raha" }
  ],
  "memory_context": "user mentioned exam stress last week",
  "image_description": "homemade chole bhature, slightly burnt edges",
  "max_new_tokens": 200,
  "temperature": 0.8
}
```

`GET /health` - GPU status, VRAM, load time

`POST /describe` - base64 image to description (see report for known issues)

---

## Evaluation

```bash
modal run evals/eval_suite.py
```

Runs 25 scenarios across 8 categories comparing base Gemma 3 12B vs SFT Ira. Results saved to `evals/eval_results.json`.

---

## Model

- Base: `unsloth/gemma-3-12b-it`
- Method: QLoRA, rank 16, 2 epochs
- Hardware: A100 40GB (Modal.com)
- Checkpoint: stored on Modal volume `ira-training-vol`

Full details in `final_report.md`.
