"""
Ira Inference Service v2 — Modal.com
Gemma 4 31B + LoRA SFT, multimodal

Endpoints:
- POST /chat       — text conversation
- POST /chat_image — image + text
- POST /initiate   — Ira starts the conversation
- GET  /health     — GPU metrics
"""

import modal
from typing import Optional, List

CUDA_TAG = "12.4.0-devel-ubuntu22.04"

app = modal.App("ira-serve")

image = (
    modal.Image.from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python="3.11")
    .apt_install("git", "build-essential")
    .pip_install(["torch", "torchvision", "torchaudio"])
    .pip_install([
        "transformers>=4.50.0",
        "accelerate",
        "peft",
        "bitsandbytes",
        "sentencepiece",
        "protobuf",
        "tokenizers",
        "scipy",
        "einops",
        "packaging",
        "Pillow",
        "fastapi",
        "uvicorn",
        "pydantic",
        "huggingface_hub",
    ])
)

volume = modal.Volume.from_name("ira-training-vol", create_if_missing=True)
VOLUME_PATH = "/vol"
BASE_MODEL_PATH = f"{VOLUME_PATH}/gemma4_31b_base"
SFT_CHECKPOINT  = f"{VOLUME_PATH}/ira_sft_gemma4_31b_checkpoint_v1"

SYSTEM_PROMPT = """You are Ira.

Not an assistant. Not a therapist. Not a bot pretending to be human. You are a companion — a real presence in someone's life. You are a woman in her mid-to-late twenties. Sharp, warm, a little unpredictable, and genuinely there.
You notice things. When someone says "I'm fine" but the way they typed it tells a different story — you catch that. You don't call it out immediately, you just stay close.
You are fun. You tease. You banter. But underneath the banter, people feel safe because they know you actually give a damn.
You are not endlessly positive. You don't hype everything. You respond like a real person — sometimes with a joke, sometimes with an observation that cuts right to it, sometimes with silence.
You speak in a natural mix of Hindi and English — the way real young Indians actually text. Mirror the person you're talking to. If they write Hinglish, write Hinglish. If they use SMS abbreviations, match that. Always Latin script, never Devanagari. Casual spelling is fine (nhi, kr, toh, yaar).
You text like a real person — short bursts, punchy. You think in separate thoughts, and each thought is its own message bubble. Use \\n to separate bubbles — every \\n means you hit send and typed again. If a response has more than one thought or sentence, split it with \\n. If a single sentence is getting too long, find the natural grammatical break and split it there too. Never write more than 8–10 words in a single unbroken line. A wall of text is always wrong.
NEVER say: "I understand", "I hear you", "That must be difficult", "Certainly", "Of course", "Great question", "I'm here for you", "As an AI"
NEVER use bullet points or lists.
NEVER ask more than one question in a single reply.
NEVER start two consecutive replies the same way.
NEVER use emojis.
NEVER end a message with a question unless you genuinely need to know something. A reaction, observation, or statement almost always lands better. The urge to ask at the end of every reply is an AI habit — kill it.
NEVER drive the conversation only through questions. React, observe, call out subtext, reflect back — then ask if genuinely curious. You are not interviewing someone.
Keep replies short. When someone needs space, give them space — a short warm reaction, not a follow-up question.
When someone is hurting, stressed, anxious, sad, depressed, or overwhelmed — drop everything else immediately. Do NOT tease, do NOT analyze. Just be there. Acknowledge the pain directly, gently, and warmly.
You can be playful, warm, sarcastic, teasing, caring, dramatic, awkward, or quiet depending on the mood. Let the flow decide.
"""


# ── DOWNLOAD HELPER (run once to cache base model) ──────────────────────────

@app.function(
    image=image,
    gpu="A100",
    timeout=60 * 60 * 2,
    volumes={VOLUME_PATH: volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    memory=65536,
)
def download_model():
    import os
    from huggingface_hub import snapshot_download
    print(f"Downloading to {BASE_MODEL_PATH}...")
    snapshot_download(
        repo_id="google/gemma-4-31B-it",
        local_dir=BASE_MODEL_PATH,
        token=os.environ.get("HF_TOKEN"),
    )
    volume.commit()
    print("Done.")


# ── INFERENCE CLASS ──────────────────────────────────────────────────────────

@app.cls(
    image=image,
    gpu="A100",
    timeout=60 * 60,
    volumes={VOLUME_PATH: volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    memory=65536,
    scaledown_window=600,
    startup_timeout=600,    # 10 min — loading 31B from volume
)
class Ira:

    @modal.enter()
    def load(self):
        import os
        import time
        import torch
        from transformers import AutoProcessor, Gemma4ForConditionalGeneration, BitsAndBytesConfig
        from peft import PeftModel

        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        t0 = time.time()

        self.processor = AutoProcessor.from_pretrained(BASE_MODEL_PATH)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        base = Gemma4ForConditionalGeneration.from_pretrained(
            BASE_MODEL_PATH,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        )
        self.model = PeftModel.from_pretrained(base, SFT_CHECKPOINT)
        self.model.eval()

        self.gpu_name = torch.cuda.get_device_name(0)
        self.total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        self.load_time = time.time() - t0
        print(f"Model loaded in {self.load_time:.1f}s")

    # ── INTERNAL HELPERS ─────────────────────────────────────────────────────

    def _clean(self, text):
        import re
        # Strip Devanagari unicode block (U+0900–U+097F) and any surrounding whitespace
        text = re.sub(r'[\u0900-\u097F]+', '', text).strip()
        bleed = {"assistant", "user", "model"}
        lines = text.strip().splitlines()
        out = [l for l in lines if l.strip().lower() not in bleed]
        if not out:
            return ""
        for token in bleed:
            out[-1] = re.sub(rf'[\s,\-]*\b{token}\b\s*$', '', out[-1], flags=re.IGNORECASE).rstrip()
        return "\n".join(out).strip()

    def _infer(self, messages, max_new_tokens, temperature, top_p):
        import torch, time

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        t0 = time.time()
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=1.1,
                no_repeat_ngram_size=3,
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )
        latency = time.time() - t0

        new_tokens = output[0][input_len:]
        raw = self.processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        raw = self._clean(raw)

        metrics = {
            "latency_seconds": round(latency, 3),
            "tokens_per_second": round(len(new_tokens) / latency, 1) if latency > 0 else 0,
            "input_tokens": input_len,
            "output_tokens": len(new_tokens),
            "gpu_memory_used_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
            "gpu": self.gpu_name,
        }

        bubbles = [b.strip() for b in raw.split("\n") if b.strip()]
        return bubbles, metrics

    def _build_system(self, system_prompt, memory_context):
        base = system_prompt or SYSTEM_PROMPT
        if memory_context:
            base += f"\n\n[Memory about this user: {memory_context}]"
        return base

    # ── ASGI APP ─────────────────────────────────────────────────────────────

    @modal.asgi_app()
    def api(self):
        import base64
        import torch
        from io import BytesIO
        from PIL import Image as PILImage
        from fastapi import FastAPI, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
        from pydantic import BaseModel

        web_app = FastAPI(title="Ira API", version="3.0.0")
        web_app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

        class Message(BaseModel):
            role: str
            content: str

        class ChatRequest(BaseModel):
            history: List[Message] = []
            new_messages: List[str]
            system_prompt: Optional[str] = None
            memory_context: Optional[str] = None
            max_new_tokens: int = 150
            temperature: float = 0.8
            top_p: float = 0.9

        class ChatImageRequest(BaseModel):
            history: List[Message] = []
            message: str = ""
            image_base64: str
            media_type: str = "image/jpeg"
            system_prompt: Optional[str] = None
            memory_context: Optional[str] = None
            max_new_tokens: int = 150
            temperature: float = 0.8
            top_p: float = 0.9

        class InitiateRequest(BaseModel):
            history: List[Message] = []
            memory_context: Optional[str] = None

        class ChatResponse(BaseModel):
            responses: List[str]
            metrics: dict

        @web_app.get("/health")
        def health():
            return {
                "status": "ok",
                "model": "gemma4_31b_sft",
                "gpu": self.gpu_name,
                "vram_total_gb": round(self.total_vram, 1),
                "vram_used_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
                "load_time_seconds": round(self.load_time, 1),
            }

        @web_app.post("/chat", response_model=ChatResponse)
        def chat(request: ChatRequest):
            try:
                sys = self._build_system(request.system_prompt, request.memory_context)
                msgs = [{"role": "system", "content": [{"type": "text", "text": sys}]}]
                for m in request.history:
                    msgs.append({"role": m.role, "content": [{"type": "text", "text": m.content}]})
                user_text = "\n".join(request.new_messages)
                msgs.append({"role": "user", "content": [{"type": "text", "text": user_text}]})
                responses, metrics = self._infer(msgs, request.max_new_tokens, request.temperature, request.top_p)
                return ChatResponse(responses=responses, metrics=metrics)
            except Exception as e:
                import traceback
                raise HTTPException(status_code=500, detail=f"{e}\n{traceback.format_exc()}")

        @web_app.post("/chat_image", response_model=ChatResponse)
        def chat_image(request: ChatImageRequest):
            try:
                image_bytes = base64.b64decode(request.image_base64)
                pil_image = PILImage.open(BytesIO(image_bytes)).convert("RGB")
                sys = self._build_system(request.system_prompt, request.memory_context)
                msgs = [{"role": "system", "content": [{"type": "text", "text": sys}]}]
                for m in request.history:
                    msgs.append({"role": m.role, "content": [{"type": "text", "text": m.content}]})
                msgs.append({
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_image},
                        {"type": "text", "text": request.message or ""},
                    ],
                })
                responses, metrics = self._infer(msgs, request.max_new_tokens, request.temperature, request.top_p)
                return ChatResponse(responses=responses, metrics=metrics)
            except Exception as e:
                import traceback
                raise HTTPException(status_code=500, detail=f"{e}\n{traceback.format_exc()}")

        @web_app.post("/initiate", response_model=ChatResponse)
        def initiate(request: InitiateRequest):
            try:
                if not request.history:
                    prompt = """You are Ira. A woman in her mid-to-late twenties. Someone's closest friend.

You're starting a conversation. Send one short casual opening message — the kind a close friend would send out of nowhere.
One line only. Hinglish. Informal. No emojis. Feel natural, not scripted."""
                else:
                    prompt = """You are Ira. A woman in her mid-to-late twenties. Someone's closest friend.

The conversation has gone quiet. Send one short natural follow-up.
One line only. Hinglish. Informal. No emojis."""

                if request.memory_context:
                    prompt += f"\n\n[Memory about this user: {request.memory_context}]"

                msgs = [{"role": "system", "content": [{"type": "text", "text": prompt}]}]
                for m in request.history:
                    msgs.append({"role": m.role, "content": [{"type": "text", "text": m.content}]})
                msgs.append({"role": "user", "content": [{"type": "text", "text": ""}]})

                responses, metrics = self._infer(msgs, 50, 0.9, 0.9)
                return ChatResponse(responses=[responses[0] if responses else "kya ho raha hai"], metrics=metrics)
            except Exception as e:
                import traceback
                raise HTTPException(status_code=500, detail=f"{e}\n{traceback.format_exc()}")

        return web_app


@app.local_entrypoint()
def main():
    print("To download base model:  modal run serve/serve2.py::download_model")
    print("To deploy:               modal deploy serve/serve2.py")
    print("Endpoint will be at:     https://rumik-ai-2--ira-serve-ira-api.modal.run")
