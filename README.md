# Ira — Hinglish Companion AI

Ira is a companion AI built on Gemma 3 12B, fine-tuned to have conversations the way young Indians actually text — natural Hinglish, warm but not soft, funny without trying too hard.

---

## What's in this repo

```
README.md
final_report.md          — full write-up: training, evals, decisions, failure modes
pipeline.html            — visual flowcharts of the full pipeline (open in browser)
serve/
    serve.py             — FastAPI inference service (Modal.com), multimodal
    chat.py              — CLI chat client
ira_async.html           — async web client (debounce/gather, image attach, buffer strip)
training/
    train_sft_modal.py   — SFT training script
    data_prep.py         — dataset preparation and formatting
preference/
    train_dpo_modal.py   — DPO training script
    dpo_pairs_v2.json    — 41 hand-curated preference pairs
    dpo_combined.jsonl   — 204 total preference pairs
evals/
    eval_suite.py        — evaluation script (base Gemma vs SFT Ira)
    eval_results.json    — full eval output
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
python training/data_prep.py
modal run training/train_sft_modal.py
```

**DPO (optional — currently degrades quality, see report):**

```bash
modal run preference/train_dpo_modal.py
```

---

## Inference Service

**Deploy:**

```bash
modal deploy serve/serve.py
```

Endpoint: `https://rumik-ai-2--ira-inference-service-api.modal.run`

Cold start: ~90s. Warm inference: 1–3s. VRAM: ~13GB on A100 40GB.

---

## API

**`POST /chat`** — text conversation, consecutive user messages supported

```json
{
  "history":        [{"role": "user", "content": "yaar"}],
  "new_messages":   ["kuch bata", "kya ho raha"],
  "memory_context": "user has exam tomorrow, mentioned anxiety last week",
  "temperature":    0.8,
  "max_new_tokens": 150
}
```

`new_messages` accepts a list — multiple messages sent before Ira replied are batched into one user turn on the backend.

**`POST /chat_image`** — send an image, Ira actually sees it

```json
{
  "history":      [],
  "message":      "dekh yeh banaya maine",
  "image_base64": "...",
  "media_type":   "image/jpeg"
}
```

The image is passed directly through the Gemma 3 vision processor — no description injection.

**`POST /initiate`** — Ira speaks first, or re-initiates after a pause

```json
{ "history": [] }
```

**`GET /health`** — GPU, VRAM, load time

---

## Web Client

Open `ira_async.html` in a browser and point it at the Modal endpoint. No build step, no server.

**Features:**

- **Consecutive multi-message batching** — messages sent in a burst are collected and sent together. Ira waits for a 4s typing pause, then gathers for up to 5s before replying.
- **Concurrent request guard** — messages typed while Ira is responding are held in a ghost buffer strip and sent immediately after. No parallel API calls, no server errors.
- **Buffer strip with cancel** — queued messages appear as faded chips above the input bar. Each chip has an × button to remove it before it's sent.
- **Image attach** — thumbnail preview strip before sending, optional caption, renders as an image card in chat.
- **Multi-bubble responses** — when Ira sends two bubbles (emotional trigger or image), they appear with a natural staggered delay.
- **Memory panel** — free-text context injected into every request.
- **Temperature slider** — 0.1–1.5, live.

---

## Evaluation

```bash
modal run evals/eval_suite.py
```

Runs 25 scenarios across 8 categories comparing base Gemma 3 12B vs SFT Ira. Results saved to `evals/eval_results.json`.

---

## Model

| | |
|---|---|
| Base | `google/gemma-3-12b-it` |
| Method | QLoRA · rank 16 · alpha 32 · 2 epochs |
| Hardware | A100 40GB (Modal.com) |
| Train loss | 0.11 |
| Checkpoint | Modal volume `ira-training-vol` |
| Final model | SFT checkpoint (DPO degraded quality — not used) |

Full details, failure analysis, and next steps in `final_report.md`.
Pipeline flowcharts in `pipeline.html` (open in browser).
