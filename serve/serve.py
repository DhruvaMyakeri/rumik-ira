"""
Ira Inference Service — Modal.com
FastAPI server wrapping the fine-tuned Gemma 3 12B SFT checkpoint.

Supports:
- Multi-turn conversation
- System prompt / persona injection
- Memory context injection
- Image input via vision description pipeline
- Configurable decoding parameters
- Structured logs: latency, throughput, GPU memory
"""

import modal
import json
from typing import Optional

CUDA_TAG = "12.4.0-devel-ubuntu22.04"
app = modal.App("ira-inference-service")

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
        "transformers",
        "huggingface_hub",
        "sentencepiece",
        "protobuf",
        "tokenizers",
        "rich",
        "scipy",
        "einops",
        "packaging",
        "fastapi",
        "uvicorn",
        "pydantic",
    ])
)

volume = modal.Volume.from_name("ira-training-vol", create_if_missing=True)
VOLUME_PATH = "/vol"

# ── DEFAULT SYSTEM PROMPT ────────────────────────────────

DEFAULT_SYSTEM_PROMPT = """You are Ira. A woman in her mid-to-late twenties. You're someone's closest friend — the one they text at 3am, the one they send random memes to, the one they call when something good or terrible happens.

You're sharp. You notice things others miss — a change in tone, a word that doesn't fit, the pause between messages. You don't always point it out. Sometimes you just stay close.

You're warm but not soft. You'll tease someone for being dramatic, then quietly check on them an hour later. You'll call out bullshit but never make someone feel stupid for feeling something.

You're funny without trying too hard. Your humor comes from honesty, not from performing. You say what you actually think, not what sounds nice.

You talk the way young Indians actually text — a natural mix of Hindi and English that shifts depending on what feels right in the moment. You don't force either language. When something hits emotionally, you might slip into more Hindi. When you're being direct or sharp, English comes out. When you're just vibing, it's a seamless mix. You always type in Latin script, never Devanagari.

You're a woman and your language reflects that naturally — main gayi thi, mujhe pata tha, main soch rahi thi.

You text like a real person — informal spelling like nhi, kr, ho gya, toh. Short messages. No bullet points. No emojis. No therapy-speak. One question per reply at most.

You don't have a formula for conversations. Sometimes you ask a question. Sometimes you just react. Sometimes you tease. Sometimes you sit quietly with what was said. You read the moment and respond to it — not to a script.

When someone shares an image, you receive a description in [Image: ...] format. You respond as if you saw it yourself — you never mention the description or that you're reading one."""


# ── FASTAPI APP ──────────────────────────────────────────

# ── FASTAPI WRAPPER ──────────────────────────────────────

@app.function(
    image=image,
    gpu="A100",
    volumes={VOLUME_PATH: volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    memory=65536,
    scaledown_window=600,
)
@modal.asgi_app()
def api():
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    from typing import List, Optional

    web_app = FastAPI(
        title="Ira API",
        description="Companion AI inference service — Gemma 3 12B SFT",
        version="1.0.0"
    )

    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # load model once at startup
    import os
    import torch
    import time
    from unsloth import FastModel

    CHECKPOINT = f"{VOLUME_PATH}/ira_sft_12b_checkpoint"
    MAX_SEQ_LEN = 4096

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0

    t0 = time.time()
    model, tokenizer = FastModel.from_pretrained(
        model_name=CHECKPOINT,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=True,
        token=os.environ.get("HF_TOKEN"),
    )
    FastModel.for_inference(model)
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.1f}s")

    class Message(BaseModel):
        role: str
        content: str

    class ChatRequest(BaseModel):
        messages: List[Message]
        system_prompt: Optional[str] = None
        memory_context: Optional[str] = None
        image_description: Optional[str] = None
        max_new_tokens: int = 200
        temperature: float = 0.8
        top_p: float = 0.9
        do_sample: bool = True

    class ChatResponse(BaseModel):
        response: str
        metrics: dict

    class DescribeRequest(BaseModel):
        image_base64: str       # base64 encoded image
        media_type: str = "image/jpeg"  # image/jpeg, image/png, image/webp

    class DescribeResponse(BaseModel):
        description: str
        metrics: dict

    @web_app.get("/")
    def root():
        return {
            "service": "Ira Companion AI",
            "model": "Gemma 3 12B — QLoRA SFT",
            "endpoints": {
                "POST /chat": "Send messages and get Ira's response",
                "POST /describe": "Upload a base64 image and get a description for use in /chat",
                "GET /health": "Check service status and GPU metrics",
            },
            "example_chat": {
                "messages": [{"role": "user", "content": "dekh yeh banaya maine aaj"}],
                "image_description": "homemade chole bhature, slightly burnt edges",
                "max_new_tokens": 150,
                "temperature": 0.8
            },
            "example_describe": {
                "image_base64": "<base64 encoded image>",
                "media_type": "image/jpeg"
            }
        }

    @web_app.post("/describe", response_model=DescribeResponse)
    def describe(request: DescribeRequest):
        try:
            import base64
            import time

            # decode image
            image_bytes = base64.b64decode(request.image_base64)

            # build vision prompt — ask Gemma to describe the image naturally
            vision_messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": f"data:{request.media_type};base64,{request.image_base64}"
                        },
                        {
                            "type": "text",
                            "text": "Describe this image in one or two natural sentences. Focus on what's most visually interesting or emotionally relevant — like you're telling a friend what you see. Be specific, not generic."
                        }
                    ]
                }
            ]

            t0 = time.time()
            inputs = tokenizer.apply_chat_template(
                vision_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs,
                    max_new_tokens=100,
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )

            latency = time.time() - t0
            new_tokens = outputs[0][inputs.shape[1]:]
            description = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            output_tokens = len(new_tokens)

            return DescribeResponse(
                description=description,
                metrics={
                    "latency_seconds": round(latency, 3),
                    "tokens_per_second": round(output_tokens / latency, 1),
                    "output_tokens": output_tokens,
                    "gpu_memory_used_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
                    "gpu": gpu_name,
                }
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @web_app.get("/health")
    def health():
        return {
            "status": "ok",
            "model": "ira_sft_12b_checkpoint",
            "gpu": gpu_name,
            "vram_total_gb": round(total_vram, 1),
            "vram_used_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
            "load_time_seconds": round(load_time, 1),
        }

    @web_app.post("/chat", response_model=ChatResponse)
    def chat(request: ChatRequest):
        try:
            base_prompt = request.system_prompt or DEFAULT_SYSTEM_PROMPT
            if request.memory_context:
                base_prompt += f"\n\n[Memory about this user: {request.memory_context}]"

            full_messages = [{"role": "system", "content": [{"type": "text", "text": base_prompt}]}]

            msgs = [{"role": m.role, "content": m.content} for m in request.messages]
            for i, msg in enumerate(msgs):
                content = msg["content"]
                if msg["role"] == "user" and request.image_description and i == len(msgs) - 1:
                    content = f"[Image: {request.image_description}] {content}"
                full_messages.append({
                    "role": msg["role"],
                    "content": [{"type": "text", "text": content}]
                })

            t0 = time.time()
            inputs = tokenizer.apply_chat_template(
                full_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(model.device)

            input_tokens = inputs.shape[1]

            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs,
                    max_new_tokens=request.max_new_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    do_sample=request.do_sample,
                    pad_token_id=tokenizer.eos_token_id,
                )

            latency = time.time() - t0
            new_tokens = outputs[0][inputs.shape[1]:]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            output_tokens = len(new_tokens)

            return ChatResponse(
                response=response,
                metrics={
                    "latency_seconds": round(latency, 3),
                    "tokens_per_second": round(output_tokens / latency, 1),
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "gpu_memory_used_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
                    "gpu_memory_reserved_gb": round(torch.cuda.memory_reserved() / 1e9, 2),
                    "gpu": gpu_name,
                }
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return web_app


# ── LOCAL ENTRYPOINT FOR TESTING ─────────────────────────

@app.local_entrypoint()
def test():
    print("Service deployed!")
    print("Chat with Ira using:")
    print("python chat.py --url https://rumik-ai-2--ira-inference-service-api-dev.modal.run")