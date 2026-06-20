# Ira,Hinglish Companion AI

Ira is a companion AI fine-tuned on **Gemma 4 31B**, trained on an A100 40GB to talk the way young Indians actually text.

She code-switches mid-conversation without being told to - moving from English to Hinglish to heavy-Hindi depending on the emotional weight of what's being said.
She holds roleplay without breaking character. Ask her to be your coworker, your road trip co-pilot, your fake date, your college rival ,she slides in immediately and stays there. No "okay I'll pretend to be your coworker now." She just is.

She reads subtext. "I'm fine" typed a certain way is not fine. She notices, stays close, doesn't push. When someone actually says something hard, she drops everything else - no analysis, no questions, no advice. Just presence.

She has a personality outside of the conversation. She listens to the Rockstar OST when she's in a certain mood. She'll rewatch Jab We Met without warning. She has opinions about Penny vs Sheldon. She's been to Goa and Puducherry. These things come up naturally because they're part of who she is - not because she was asked.

And she sees images. Send her a photo and she reacts to what's actually in it - not a description, not a caption, a reaction. Built on Gemma 4's native vision, no description injection.

The dataset behind her was built with unusual care. Generated conversations span 16 scenario categories - emotional sensitivity, code-switching, late-night energy, ambiguous recovery, safety boundaries, roleplay, first-contact awkwardness - each written to feel like a different person on a different day, with natural imperfections and real emotional arcs. But the training data goes beyond generated conversations: it includes manually curated Reddit threads where real people talk the way Ira talks, and transcripts from the Ishmeet Khamba podcast - hours of natural Hinglish conversation used as a grounding signal for cadence, register, and how people actually sound when they're being real with each other. Not a chatbot dataset. Something closer to a friendship.

---

---

## What's in this repo

```
build_dataset.py              - builds ira_train.jsonl / ira_val.jsonl from SCENE format + JSONL sources
dataset_gen_prompt.md         - prompt used to generate gemma4_dataset.txt (395 conversations, 16 scenario tags)
dpo_gen_prompt.md             - prompt for DPO preference pair generation (390 pairs, 13 categories)
gemma4_dataset.txt            - SFT training conversations in SCENE format
gemma4_dpo.txt                - DPO preference pairs
dpo_combined.jsonl            - compiled DPO dataset
ira_train.jsonl / ira_val.jsonl - final train/val splits (90/10)

serve/
    serve2.py                 - FastAPI inference server (Modal.com), Gemma 4 31B + LoRA, multimodal
    serve.py                  - earlier version (Gemma 3 12B)
    chat.py                   - CLI chat client

training/
    train_sft_modal.py        - SFT training script (QLoRA 4-bit NF4, Gemma 4 31B)
    test_inference_modal.py   - 17 inference test cases with probe annotations

preference/
    train_dpo_modal.py        - DPO training script

evals/
    eval_suite.py             - 37 scenarios across 12 categories, base vs SFT, auto-scoring
    eval_results.json         - latest eval output (one line per scenario)

eval_results_v2.json          - v2 SFT eval
eval_results_v3.json          - v3 SFT eval

ira_async.html                - async web client (multi-bubble, image attach, memory panel)
pipeline.html                 - visual pipeline flowcharts (open in browser)
final_report.md               - full write-up: decisions, training runs, failure analysis
```

---

## Model

|              |                                                       |
| ------------ | ----------------------------------------------------- |
| Base         | `google/gemma-4-31B-it`                               |
| Method       | QLoRA · 4-bit NF4 · rank 16 · alpha 32                |
| Hardware     | A100 40GB (Modal.com)                                 |
| SFT versions | v1, v2, v3 (see eval results)                         |
| DPO          | Attempted - degraded quality on v1, not retried on v3 |
| Checkpoint   | Modal volume `ira-training-vol`                       |

---

## Training progression

**v1 SFT** - first run on Gemma 4 31B. Question-ending rate 44% (too high), multi-bubble 51% (too low). Base Gemma outperformed SFT on formatting - training data caused regression.

**v2 SFT** - fixed question-ending (44% → 30%), multi-bubble still stuck at 47%. Root cause: 45% of A: turns in training data were single-bubble.

**v3 SFT** - focused on emotional understanding, persona consistency, language mirroring. Question-ending 33%, multi-bubble 47%, forbidden phrases near zero. Emotional training landed well.

---

## Setup

```bash
pip install modal
modal setup
modal secret create huggingface-secret HF_TOKEN=your_token_here
```

---

## Build dataset

```bash
python build_dataset.py
```

Reads `gemma4_dataset.txt` + `ismeet_khamba.jsonl` + `reddit.jsonl`, deduplicates, applies bubble post-processing, outputs `ira_train.jsonl` and `ira_val.jsonl`.

---

## Training

```bash
modal run training/train_sft_modal.py
```

```bash
modal run preference/train_dpo_modal.py
```

---

## Inference

```bash
modal deploy serve/serve2.py
```

Endpoint: `https://rumik-ai-2--ira-serve-ira-api.modal.run`

Cold start: ~90s. Warm inference: 1–3s.

**`POST /chat`**

```json
{
  "history": [{ "role": "user", "content": "yaar" }],
  "message": "kuch feel nahi ho raha",
  "memory_context": "user has exam tomorrow",
  "temperature": 0.8,
  "max_new_tokens": 150
}
```

**`POST /chat_image`** - send image as base64, Ira sees it via Gemma 4 vision processor.

---

## Eval

```bash
modal run evals/eval_suite.py
```

37 scenarios across: Emotional Sensitivity, Warmth, Playfulness, Safety/Boundaries, Multilingual, Multi-turn Memory, Introductory, Persona Stress, Multimodal, Ambiguous Recovery, Roleplay.

Auto-scores: bubble count, question-ending rate, forbidden phrase hits, multi-bubble rate.

Results saved as JSONL - one line per scenario.

---

## Web client

Open `ira_async.html` in a browser. No build step.

- Multi-bubble responses with staggered delay
- Consecutive message batching (4s pause → gather → send)
- Ghost buffer strip for messages typed while Ira is responding
- Image attach with caption
- Memory panel + temperature slider

---

## Key design decisions

- **\n as bubble separator** - each \n in Ira's response = she hit send and typed again. Rendered as separate chat bubbles in the UI.
- **Gemma 4 31B over 12B** - significantly better Hinglish coherence and emotional depth.
- **DPO not used in final** - 13 categories × 30 pairs generated, training attempted, quality degraded. SFT-only checkpoint is the final model.
- **Devanagari stripped** - 5 contaminated training lines caused Hindi script bleed. Fixed in `_clean()` at inference and cleaned from source data.
- **Two-pass image inference removed** - attempted base model captioning + LoRA text pass, removed in favour of direct single-pass multimodal.
