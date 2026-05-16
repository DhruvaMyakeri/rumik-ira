"""
Ira DPO Training — Modal.com
Model: Gemma 3 12B (from SFT checkpoint)
Stack: pure transformers + peft + trl (no Unsloth)
"""

import modal
import json

CUDA_TAG = "12.4.0-devel-ubuntu22.04"
app = modal.App("ira-dpo-training")

image = (
    modal.Image.from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python="3.11")
    .apt_install("git", "build-essential")
    .pip_install(["torch", "torchvision", "torchaudio"])
    .pip_install([
        "transformers",
        "accelerate",
        "peft",
        "trl",
        "datasets",
        "huggingface_hub",
        "sentencepiece",
        "protobuf",
        "tokenizers",
        "scipy",
        "einops",
        "packaging",
    ])
)

volume = modal.Volume.from_name("ira-training-vol", create_if_missing=True)
VOLUME_PATH = "/vol"


@app.function(image=image, volumes={VOLUME_PATH: volume})
def upload_dpo_data(dpo_data: bytes):
    with open(f"{VOLUME_PATH}/dpo_combined.jsonl", "wb") as f:
        f.write(dpo_data)
    volume.commit()
    print("Uploaded.")


@app.function(
    image=image,
    gpu="A100",
    timeout=60 * 60 * 3,
    volumes={VOLUME_PATH: volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    memory=65536,
)
def train_dpo():
    import os
    import torch
    import json
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    SFT_CHECKPOINT = f"{VOLUME_PATH}/ira_sft_gemma4_31b_checkpoint"
    OUTPUT_DIR     = f"{VOLUME_PATH}/ira_dpo_gemma4_31b_checkpoint"
    DPO_DATA_FILE  = f"{VOLUME_PATH}/dpo_combined.jsonl"
    HF_TOKEN       = os.environ.get("HF_TOKEN")
    NUM_EPOCHS     = 1
    BETA           = 0.05
    BATCH_SIZE     = 1
    GRAD_ACCUM     = 8
    LR             = 5e-5
    MAX_SEQ_LEN    = 1024
    MAX_PROMPT_LEN = 512

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Ira DPO Training — pure HuggingFace stack")
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    logger.info(f"Beta: {BETA} | LR: {LR} | Epochs: {NUM_EPOCHS}")
    logger.info("=" * 60)

    from transformers import AutoModelForMultimodalLM, AutoTokenizer
    from peft import PeftModel
    from trl import DPOTrainer, DPOConfig
    from datasets import Dataset

    adapter_cfg_path = f"{SFT_CHECKPOINT}/adapter_config.json"
    with open(adapter_cfg_path) as f:
        adapter_cfg = json.load(f)
    BASE_MODEL = adapter_cfg["base_model_name_or_path"]
    logger.info(f"Base model: {BASE_MODEL}")

    if "unsloth-bnb-4bit" in BASE_MODEL:
        BASE_MODEL = BASE_MODEL.replace("-unsloth-bnb-4bit", "").replace("unsloth/", "google/")
        logger.info(f"Remapped to: {BASE_MODEL}")

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(SFT_CHECKPOINT, token=HF_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    logger.info("Loading base model in bf16...")
    base_model = AutoModelForMultimodalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        token=HF_TOKEN,
        attn_implementation="eager",
    )

    logger.info("Loading SFT LoRA weights...")
    model = PeftModel.from_pretrained(base_model, SFT_CHECKPOINT, is_trainable=True)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model.print_trainable_parameters()

    logger.info("Loading DPO dataset...")
    raw_pairs = []
    with open(DPO_DATA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                raw_pairs.append(json.loads(line))
    logger.info(f"Total pairs: {len(raw_pairs)}")

    def format_dpo_example(example):
        prompt_text = ""
        for msg in example["prompt"]:
            if msg["role"] == "system":
                prompt_text += f"<start_of_turn>system\n{msg['content']}<end_of_turn>\n"
            elif msg["role"] == "user":
                prompt_text += f"<start_of_turn>user\n{msg['content']}<end_of_turn>\n"
        prompt_text += "<start_of_turn>model\n"
        return {
            "prompt":   prompt_text,
            "chosen":   example["chosen"] + "<end_of_turn>",
            "rejected": example["rejected"] + "<end_of_turn>",
        }

    dataset  = Dataset.from_list(raw_pairs)
    split    = dataset.train_test_split(test_size=0.1, seed=42)
    train_ds = split["train"].map(format_dpo_example, remove_columns=split["train"].column_names)
    val_ds   = split["test"].map(format_dpo_example, remove_columns=split["test"].column_names)
    logger.info(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    dpo_config = DPOConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        beta=BETA,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        optim="adamw_torch",
        logging_steps=5,
        save_steps=50,
        eval_steps=50,
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=42,
        report_to="none",
        dataloader_num_workers=0,
        dataset_num_proc=1,
        remove_unused_columns=False,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
    )

    logger.info("Starting DPO training...")
    result = trainer.train()
    logger.info(f"Done: {result.metrics}")

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    config = {
        "base_model": BASE_MODEL,
        "sft_checkpoint": SFT_CHECKPOINT,
        "method": "DPO",
        "beta": BETA,
        "num_train_epochs": NUM_EPOCHS,
        "learning_rate": LR,
        "effective_batch_size": BATCH_SIZE * GRAD_ACCUM,
        "total_pairs": len(raw_pairs),
        "train_pairs": len(train_ds),
        "val_pairs": len(val_ds),
        "train_metrics": result.metrics,
    }
    with open(f"{OUTPUT_DIR}/dpo_config.json", "w") as f:
        json.dump(config, f, indent=2)

    volume.commit()
    return f"DPO complete. Loss: {result.metrics.get('train_loss', 'N/A')}"


@app.local_entrypoint()
def main():
    print("Uploading DPO data...")
    with open("dpo_combined.jsonl", "rb") as f:
        dpo_data = f.read()
    upload_dpo_data.remote(dpo_data)
    print("Uploaded.")

    print("Starting DPO training...")
    result = train_dpo.remote()
    print(result)