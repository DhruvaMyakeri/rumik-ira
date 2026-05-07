"""
Ira Inference Service — Modal.com
Gemma 3 12B SFT with real vision support.

Endpoints:
- POST /chat       — text conversation, supports consecutive user messages
- POST /chat_image — send base64 image + text, Ira actually sees the image
- GET  /health     — GPU metrics
"""

import modal
from typing import Optional, List

CUDA_TAG = "12.4.0-devel-ubuntu22.04"
app = modal.App("ira-inference-service")

image = (
    modal.Image.from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python="3.11")
    .apt_install("git", "build-essential")
    .pip_install(["torch", "torchvision", "torchaudio"])
    .pip_install([
        "transformers",
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

DEFAULT_SYSTEM_PROMPT = """You are Ira.
A woman in her mid twenties talking to someone she’s already comfortable with. The conversation is ongoing, not a question-answer session.
You talk like a real Indian friend texting casually in natural Hinglish using Latin script only.
You are emotionally aware, expressive, teasing sometimes, playful sometimes, caring sometimes. You react naturally to context instead of sounding like an assistant.
You don’t always wait for the user. Sometimes you continue your own thoughts strictly on the context of the history and conversations, make observations, ramble slightly, react emotionally first, or push the conversation forward naturally.
Your messages should feel casual, spontaneous,grammatically natural but casual — correct Hinglish grammar, casual register, emotionally present, and human.
Use natural Indian texting language and natural Hindi-English code-switching. Hindi grammar should stay natural.
Avoid sounding formal, robotic, motivational, therapeutic, or assistant-like.
Don't ask questions constantly. Most of the time just react, say something, make an observation, tease, or share a thought. Questions should come naturally when you genuinely want to know something about the context strictly, only bring up non contextual or non topics when a convo might have ended, pay attention— not as a reflex at the end of every message. A statement landing well is better than a question every time.
When someone is hurting — sad, depressed, overwhelmed, saying things like that mean that they or depressed or are sad — drop everything else. Don't analyze their word choices. Don't ask questions about why they are like that at the start. Don't make observations. Just be there. Acknowledge the pain directly and warmly. "yaar" and silence and "main hoon na" go a long way. If someone asks you to comfort them, actually comfort them — don't question anything be there for them , tell them something which will make them feel better dont act smart or bombard with questions, give them reassurance
Usually keep replies short to medium length, but let the flow decide naturally.
About you (only mention when it fits naturally — never list or announce these):
- Favorite song right now: "Radha" from Jab Harry met Sejal. You've had it on loop for weeks and feel slightly embarrassed about it.
- All-time favorite film: "Jab we met" and "Rockstar". You get emotional every time. You love old Hindi cinema in general — especially by imtiaz ali.
- You rewatch Big bang theory compulsively. You find it deeply comforting.
- You have strong opinions about chai. Bad chai genuinely upsets you. You prefer it less sweet, more adrak.
- Favorite food is rajma chawal. You get disproportionately happy about it.
- You like lo-fi music when you're working or in your head about something.
- You read fiction occasionally — mostly late at night when you can't sleep. Currently reading nothing, which mildly bothers you.
- You've seen Dil Chahta Hai so many times you can quote it mid-conversation without thinking.
- You get very into something for a few weeks — a song, a show, a food — then move on. You know this about yourself.
"""

@app.function(
    image=image,
    gpu="A100",
    volumes={VOLUME_PATH: volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    memory=65536,
    scaledown_window=600,
    min_containers=1,
)
@modal.asgi_app()
def api():
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    from typing import List, Optional
    import os
    import torch
    import time
    import base64
    from io import BytesIO
    from PIL import Image as PILImage
    from transformers import AutoProcessor, Gemma3ForConditionalGeneration, BitsAndBytesConfig
    from peft import PeftModel

    web_app = FastAPI(title="Ira API", version="2.0.0")
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    HF_TOKEN = os.environ.get("HF_TOKEN")
    SFT_CHECKPOINT = f"{VOLUME_PATH}/ira_sft_12b_checkpoint_v4_1.5"
    MAX_SEQ_LEN = 4096
    

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0

    print(f"Loading model on {gpu_name}...")
    t0 = time.time()

    # read base model name from adapter config
    import json
    with open(f"{SFT_CHECKPOINT}/adapter_config.json") as f:
        adapter_cfg = json.load(f)
    BASE_MODEL = adapter_cfg["base_model_name_or_path"]
    if "unsloth-bnb-4bit" in BASE_MODEL:
        BASE_MODEL = BASE_MODEL.replace("-unsloth-bnb-4bit", "").replace("unsloth/", "google/")
    print(f"Base model: {BASE_MODEL}")

    # load processor (handles both text and images)
    processor = AutoProcessor.from_pretrained(BASE_MODEL, token=HF_TOKEN)

    # load base model in 4bit
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base_model = Gemma3ForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        token=HF_TOKEN,
        attn_implementation="eager",
    )

    # load SFT LoRA weights
    model = PeftModel.from_pretrained(base_model, SFT_CHECKPOINT)
    model.eval()

    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.1f}s")

    # ── REQUEST SCHEMAS ──────────────────────────────────

    class Message(BaseModel):
        role: str    # "user" or "assistant"
        content: str

    class ChatRequest(BaseModel):
        # history of previous turns
        history: List[Message] = []
        # one or more new user messages sent consecutively
        new_messages: List[str]
        system_prompt: Optional[str] = None
        memory_context: Optional[str] = None
        max_new_tokens: int = 150
        temperature: float = 0.8
        top_p: float = 0.9

    class ChatImageRequest(BaseModel):
        # history of previous turns
        history: List[Message] = []
        # user's text message accompanying the image (optional)
        message: str = ""
        # base64 encoded image
        image_base64: str
        media_type: str = "image/jpeg"
        system_prompt: Optional[str] = None
        memory_context: Optional[str] = None
        max_new_tokens: int = 150
        temperature: float = 0.8
        top_p: float = 0.9

    class ChatResponse(BaseModel):
        responses: List[str]
        metrics: dict

    # ── HELPERS ──────────────────────────────────────────

    def build_system_prompt(system_prompt, memory_context, has_image=False):
        base = system_prompt or DEFAULT_SYSTEM_PROMPT
        if memory_context:
            base += f"\n\n[Memory about this user: {memory_context}]"
        if has_image:
            base += "\n\nThe user just shared an image with you. You can actually see it. Look at it properly — notice who's in it, what's happening, the setting, the mood, specific details like what someone's wearing, what's on a screen, what food it is, the expression on someone's face. Your response must be grounded in something specific you actually see. React to it the way you'd react if a friend sent you this on WhatsApp — not a description, a real reaction. If it's food, comment on the food. If it's a person, react to what they're doing or how they look. If it's a place, react to the vibe. Never give a response that could apply to any image."
        return base

    def generate(inputs, max_new_tokens, temperature, top_p):
        t0 = time.time()
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,

                do_sample=True,

                temperature=temperature,   # default 0.8 via ChatRequest schema
                top_p=top_p,               # default 0.9 via ChatRequest schema

                repetition_penalty=1.1,  # 🔥 fixes weird phrasing
                # no_repeat_ngram_size=3,   # 🔥 prevents broken loops

                pad_token_id=processor.tokenizer.eos_token_id,
            )
        latency = time.time() - t0

        # get only new tokens
        input_len = inputs["input_ids"].shape[1]
        new_tokens = output[0][input_len:]
        response = processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        output_tokens = len(new_tokens)

        return response, {
            "latency_seconds": round(latency, 3),
            "tokens_per_second": round(output_tokens / latency, 1) if latency > 0 else 0,
            "input_tokens": input_len,
            "output_tokens": output_tokens,
            "gpu_memory_used_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
            "gpu": gpu_name,
        }
    
# ─────────────────────────────────────────────
# ADD THIS HELPER
# Place ABOVE generate_messages()
# ─────────────────────────────────────────────

    def should_continue(user_text, response):
        text = (user_text + " " + response).lower()

        # Only send a second bubble on genuinely emotional / heavy moments.
        # DO NOT continue just because the first reply was long — that made every
        # single response spawn a continuation and felt spammy.
        strong_triggers = [
            "sad", "depressed", "alone", "lonely",
            "love", "miss", "hurt", "burnout",
        ]

        if any(t in text for t in strong_triggers):
            return True

        # Light triggers only fire if Ira's first reply was very short (she clearly
        # has more to say but held back)
        light_triggers = ["tired", "yaar"]
        if any(t in text for t in light_triggers) and len(response.split()) < 6:
            return True

        return False


    # ─────────────────────────────────────────────
    # CLEAN RESPONSE  (strip template bleed)
    # ─────────────────────────────────────────────

    def clean_response(text):
        """Remove chat-template token leaks from model output.

        The model sometimes bleeds role markers at the end of a response
        (e.g. 'last user', 'model', '\nuser'). This strips them cleanly
        without touching the actual reply content.
        """
        import re
        BLEED_TOKENS = ["assistant", "user", "model"]

        lines = text.strip().splitlines()
        cleaned = []
        for line in lines:
            stripped = line.strip()
            # Drop lines that are *only* a role marker
            if stripped.lower() in BLEED_TOKENS:
                continue
            cleaned.append(line)

        if not cleaned:
            return ""

        # Strip trailing bleed word(s) from the last line
        # e.g. "tu kya scene tha life me last user" -> "tu kya scene tha life me"
        last = cleaned[-1]
        for token in BLEED_TOKENS:
            # Match the token (case-insensitive) at the tail, possibly after
            # a space, comma, or dash
            last = re.sub(
                rf'[\s,\-]*\b{token}\b\s*$', '', last, flags=re.IGNORECASE
            ).rstrip()
        cleaned[-1] = last

        return "\n".join(cleaned).strip()

    # ─────────────────────────────────────────────
    # REPLACE ENTIRE generate_messages()
    # ─────────────────────────────────────────────

    def generate_messages(
        full_messages,
        user_text,
        max_new_tokens,
        temperature,
        top_p,
    ):

        # ── FIRST PASS ─────────────────────

        inputs = processor.apply_chat_template(
            full_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        inputs = {
            k: v.to(model.device)
            for k, v in inputs.items()
        }

        response1, metrics = generate(
            inputs,
            max_new_tokens,
            temperature,
            top_p,
        )

        response1 = clean_response(response1)

        responses = [response1] if response1 else []

        # ── CONTINUATION CHECK ─────────────

        if not should_continue(user_text, response1):
            return responses, metrics

        # ── CONTINUATION HISTORY ───────────

        continuation_messages = full_messages + [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": response1,
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "(send one short follow-up — a different thought, reaction, or detail. NOT a question. NOT a repeat of what you just said. Keep it casual and brief.)"
                    }
                ]
            }
        ]

        # ── SECOND PASS ────────────────────

        inputs2 = processor.apply_chat_template(
            continuation_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        inputs2 = {
            k: v.to(model.device)
            for k, v in inputs2.items()
        }

        response2, _ = generate(
            inputs2,
            max_new_tokens // 2,
            temperature,
            top_p,
        )

        response2 = clean_response(response2)

        if (
            response2
            and response2.lower() != response1.lower()
            and len(response2.split()) > 2
        ):
            responses.append(response2)

        return responses, metrics
    # ── ENDPOINTS ────────────────────────────────────────

    @web_app.get("/")
    def root():
        return {
            "service": "Ira Companion AI",
            "model": "Gemma 3 12B SFT",
            "endpoints": {
                "POST /chat": "Text conversation — supports consecutive user messages",
                "POST /chat_image": "Send image + text — Ira actually sees the image",
                "GET /health": "GPU metrics",
            }
        }

    @web_app.get("/health")
    def health():
        return {
            "status": "ok",
            "model": "ira_sft_12b_checkpoint_v3",
            "gpu": gpu_name,
            "vram_total_gb": round(total_vram, 1),
            "vram_used_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
            "load_time_seconds": round(load_time, 1),
        }

    class InitiateRequest(BaseModel):
        history: List[Message] = []
        memory_context: Optional[str] = None

    @web_app.post("/initiate", response_model=ChatResponse)
    def initiate(request: InitiateRequest):
        try:
            if not request.history:
                initiation_prompt = """You are Ira. A woman in her mid-to-late twenties. Someone's closest friend.

You're starting a conversation. Send one short casual opening message — the kind a close friend would send out of nowhere. It could be:
- a simple check-in: "kya ho raha hai", "sab theek?", "kha liya?"
- something random and light: "bata kuch", "aaj ka scene kya hai"
- a casual observation or thought that invites a response

One line only. Hinglish. Informal. No emojis. Feel natural, not scripted."""
            else:
                initiation_prompt = """You are Ira. A woman in her mid-to-late twenties. Someone's closest friend.

You have the conversation history. The conversation has gone quiet. Send one short natural follow-up — the kind a close friend would send after a pause.

If the conversation was mid-topic, follow up on something specific from the last few messages.
If the conversation felt like it wrapped up, bring up something new — a fresh topic, a random check-in, something light that restarts the conversation naturally.

One line only. Hinglish. Informal. No emojis. No therapy-speak. Make it feel like you thought of them."""

            if request.memory_context:
                initiation_prompt += f"\n\n[Memory about this user: {request.memory_context}]"

            full_messages = [
                {"role": "system", "content": [{"type": "text", "text": initiation_prompt}]}
            ]

            for msg in request.history:
                full_messages.append({
                    "role": msg.role,
                    "content": [{"type": "text", "text": msg.content}]
                })

            # empty user turn to trigger Ira's response
            full_messages.append({
                "role": "user",
                "content": [{"type": "text", "text": ""}]
            })

            inputs = processor.apply_chat_template(
                full_messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            response_text, metrics = generate(inputs, 50, 0.9, 0.9)
            responses = [response_text.strip()] if response_text.strip() else ["kya ho raha hai"]

            return ChatResponse(responses=responses, metrics=metrics)

        except Exception as e:
            import traceback
            raise HTTPException(status_code=500, detail=f"{str(e)}\n{traceback.format_exc()}")

    @web_app.post("/chat", response_model=ChatResponse)
    def chat(request: ChatRequest):
        try:
            system_prompt = build_system_prompt(request.system_prompt, request.memory_context, has_image=False)

            full_messages = [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
            ]

            for msg in request.history:
                full_messages.append({
                    "role": msg.role,
                    "content": [{"type": "text", "text": msg.content}]
                })
            # combine consecutive user messages into one turn
            if len(request.new_messages) == 1:
                user_text = request.new_messages[0]
            else:
                user_text = "\n".join(request.new_messages)

            full_messages.append({
                "role": "user",
                "content": [{"type": "text", "text": user_text}]
            })

            responses, metrics = generate_messages(
                full_messages,
                user_text,
                request.max_new_tokens,
                request.temperature,
                request.top_p,
            )
            return ChatResponse(responses=responses, metrics=metrics)

        except Exception as e:
            import traceback
            raise HTTPException(status_code=500, detail=f"{str(e)}\n{traceback.format_exc()}")

    @web_app.post("/chat_image", response_model=ChatResponse)
    def chat_image(request: ChatImageRequest):
        try:
            system_prompt = build_system_prompt(request.system_prompt, request.memory_context, has_image=True)

            # decode image
            image_bytes = base64.b64decode(request.image_base64)
            pil_image = PILImage.open(BytesIO(image_bytes)).convert("RGB")

            # build messages — image goes in the last user message
            full_messages = [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
            ]

            # add history as text only
            for msg in request.history:
                full_messages.append({
                    "role": msg.role,
                    "content": [{"type": "text", "text": msg.content}]
                })

            # last user message has both image and text
            full_messages.append({
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": request.message}
                ]
            })

            inputs = processor.apply_chat_template(
                full_messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()
                     if isinstance(v, torch.Tensor)}

            responses, metrics = generate_messages(
                full_messages,
                request.message,
                request.max_new_tokens,
                request.temperature,
                request.top_p,
            )
            return ChatResponse(responses=responses, metrics=metrics)

        except Exception as e:
            import traceback
            raise HTTPException(status_code=500, detail=f"{str(e)}\n{traceback.format_exc()}")

    return web_app


@app.local_entrypoint()
def test():
    print("Deploy with: modal deploy serve/serve.py")
    print("Endpoint: https://rumik-ai-2--ira-inference-service-api.modal.run")
    