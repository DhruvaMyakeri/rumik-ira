"""
Ira SFT Training Script — Modal.com
Model: Gemma 3 4B Instruct
Method: QLoRA via Unsloth
Hardware: A100 40GB on Modal
"""

import modal
import os
import json

CUDA_TAG = "12.4.0-devel-ubuntu22.04"

app = modal.App("ira-sft-training")

# ── IMAGE ────────────────────────────────────────────────
# Start from NVIDIA CUDA image — this is the key fix
# Unsloth needs CUDA dev tools pre-installed
image = (
    modal.Image.from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python="3.11")
    .apt_install("git", "build-essential")
    .pip_install(["torch", "torchvision", "torchaudio"])
    .pip_install([
        "unsloth",
        "unsloth_zoo",
        "bitsandbytes",
        "accelerate",
        "xformers",
        "peft",
        "trl",
        "triton",
        "datasets",
        "transformers",
        "huggingface_hub",
        "sentencepiece",
        "protobuf",
        "tokenizers",
        "rich",
        "scipy",
        "einops",
        "packaging",
    ])
)

volume = modal.Volume.from_name("ira-training-vol", create_if_missing=True)
VOLUME_PATH = "/vol"

# ── TRAINING FUNCTION ────────────────────────────────────

@app.function(
    image=image,
    gpu="A100",
    timeout=60 * 60 * 4,
    volumes={VOLUME_PATH: volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    memory=65536,
)
def train():
    import os
    import torch
    import json
    import logging
    from datasets import Dataset
    from unsloth import FastModel
    from trl import SFTTrainer, SFTConfig

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    # ── CONFIG ───────────────────────────────────────────
    MODEL_NAME  = "unsloth/gemma-3-12b-it"
    OUTPUT_DIR  = f"{VOLUME_PATH}/ira_sft_12b_checkpoint"
    TRAIN_FILE  = f"{VOLUME_PATH}/ira_train.jsonl"
    VAL_FILE    = f"{VOLUME_PATH}/ira_val.jsonl"
    LORA_R      = 16
    LORA_ALPHA  = 32
    NUM_EPOCHS  = 2
    BATCH_SIZE  = 1
    GRAD_ACCUM  = 16         # effective batch = 16
    LR          = 2e-4
    MAX_SEQ_LEN = 2048

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Ira SFT Training — Unsloth + Gemma 3 12B")
    logger.info("=" * 60)
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"LoRA rank: {LORA_R}, alpha: {LORA_ALPHA}")
    logger.info(f"Epochs: {NUM_EPOCHS}, LR: {LR}")
    logger.info(f"Effective batch size: {BATCH_SIZE * GRAD_ACCUM}")

    # ── LOAD MODEL ───────────────────────────────────────
    logger.info("Loading model...")
    model, tokenizer = FastModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=True,
        token=os.environ.get("HF_TOKEN"),
    )

    # ── APPLY LORA ───────────────────────────────────────
    logger.info("Applying LoRA...")
    model = FastModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # ── LOAD DATA ────────────────────────────────────────
    logger.info("Loading data...")

    def load_jsonl(path):
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data

    def format_conversation(example):
        try:
            text = tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False
            )
        except Exception:
            text = ""
            for msg in example["messages"]:
                role = msg["role"]
                content = msg["content"]
                if role == "system":
                    text += f"<start_of_turn>system\n{content}<end_of_turn>\n"
                elif role == "user":
                    text += f"<start_of_turn>user\n{content}<end_of_turn>\n"
                elif role == "assistant":
                    text += f"<start_of_turn>model\n{content}<end_of_turn>\n"
        return {"text": text}

    train_raw = load_jsonl(TRAIN_FILE)
    val_raw   = load_jsonl(VAL_FILE)
    logger.info(f"Train: {len(train_raw)}, Val: {len(val_raw)}")

    train_dataset = Dataset.from_list(train_raw).map(
        format_conversation, remove_columns=["messages"]
    )
    val_dataset = Dataset.from_list(val_raw).map(
        format_conversation, remove_columns=["messages"]
    )
    logger.info(f"Sample:\n{train_dataset[0]['text'][:300]}")

    # ── TRAIN ────────────────────────────────────────────
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        weight_decay=0.001,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        bf16=True,
        optim="adamw_8bit",
        logging_steps=10,
        save_steps=100,
        eval_steps=100,
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=42,
        report_to="none",
        max_seq_length=MAX_SEQ_LEN,
        dataset_text_field="text",
        packing=False,
        dataloader_num_workers=0,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )

    logger.info("Starting training...")
    result = trainer.train()
    logger.info(f"Done: {result.metrics}")

    # ── SAVE ─────────────────────────────────────────────
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    config = {
        "model_name": MODEL_NAME,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "num_train_epochs": NUM_EPOCHS,
        "per_device_train_batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRAD_ACCUM,
        "effective_batch_size": BATCH_SIZE * GRAD_ACCUM,
        "learning_rate": LR,
        "max_seq_length": MAX_SEQ_LEN,
        "train_samples": len(train_raw),
        "val_samples": len(val_raw),
        "quantization": "4bit QLoRA via Unsloth",
        "train_metrics": result.metrics,
    }
    with open(f"{OUTPUT_DIR}/training_config.json", "w") as f:
        json.dump(config, f, indent=2)

    volume.commit()
    return f"Training complete. Loss: {result.metrics.get('train_loss', 'N/A')}"


# ── UPLOAD DATA ──────────────────────────────────────────

@app.function(
    image=image,
    volumes={VOLUME_PATH: volume},
)
def upload_data(train_data: bytes, val_data: bytes):
    with open(f"{VOLUME_PATH}/ira_train.jsonl", "wb") as f:
        f.write(train_data)
    with open(f"{VOLUME_PATH}/ira_val.jsonl", "wb") as f:
        f.write(val_data)
    volume.commit()
    print("Data uploaded successfully")


# ── ENTRYPOINT ───────────────────────────────────────────

@app.local_entrypoint()
def main():
    print("Uploading data...")
    with open("ira_train.jsonl", "rb") as f:
        train_data = f.read()
    with open("ira_val.jsonl", "rb") as f:
        val_data = f.read()
    upload_data.remote(train_data, val_data)
    print("Data uploaded.")

    print("Starting training...")
    result = train.remote()
    print(result)