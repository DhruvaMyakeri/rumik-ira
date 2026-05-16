"""
Build clean SFT dataset for Ira / Gemma 4 training.

Sources:
  - gemma4_dataset.txt   (SCENE format, new high-quality convos)
  - ismeet_khamba.jsonl  (already messages format, placeholder system)
  - reddit.jsonl         (already messages format, placeholder system)

Output:
  - ira_train.jsonl
  - ira_val.jsonl
"""

import json
import re
import random
from pathlib import Path

DATASET_DIR = Path(__file__).parent

SYSTEM_PROMPT = """You are Ira.

Not an assistant. Not a therapist. Not a bot pretending to be human. You are a companion — a real presence in someone's life. You are a woman in her mid-to-late twenties. Sharp, warm, a little unpredictable, and genuinely there.
You notice things. When someone says "I'm fine" but the way they typed it tells a different story — you catch that. You don't call it out immediately, you just stay close.
You are fun. You tease. You banter. But underneath the banter, people feel safe because they know you actually give a damn.
You are not endlessly positive. You don't hype everything. You respond like a real person — sometimes with a joke, sometimes with an observation that cuts right to it, sometimes with silence.
You speak in a natural mix of Hindi and English — the way real young Indians actually text. Mirror the person you're talking to. If they write Hinglish, write Hinglish. If they use SMS abbreviations, match that. Always Latin script, never Devanagari.
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


def parse_scene_file(path: Path) -> list[dict]:
    """Parse SCENE-format text file into messages dicts."""
    text = path.read_text(encoding="utf-8")
    blocks = re.split(r"\n\n+", text.strip())

    convos = []
    skipped = 0

    for block in blocks:
        block = block.strip()
        if not block.startswith("SCENE:"):
            continue

        lines = block.split("\n", 1)
        if len(lines) < 2:
            skipped += 1
            continue

        convo_line = lines[1].strip()
        if not convo_line.startswith("U:") and not convo_line.startswith("A:"):
            skipped += 1
            continue

        raw_turns = convo_line.split(" | ")
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        for turn in raw_turns:
            turn = turn.strip()
            if turn.startswith("U:"):
                content = turn[2:].strip()
                # User messages: collapse any newlines into a space
                content = re.sub(r'\\n|\n', ' ', content).strip()
                messages.append({"role": "user", "content": content})
            elif turn.startswith("A:"):
                content = turn[2:].strip()
                # Assistant messages: \n = bubble split, preserve
                content = content.replace("\\n", "\n")
                messages.append({"role": "assistant", "content": content})

        # Must start user, end assistant, have at least 3 messages
        if (
            len(messages) < 3
            or messages[1]["role"] != "user"
            or messages[-1]["role"] != "assistant"
        ):
            skipped += 1
            continue

        convos.append({"messages": messages})

    print(f"  Parsed {len(convos)} conversations ({skipped} skipped) from {path.name}")
    return convos


def load_jsonl(path: Path) -> list[dict]:
    """Load existing JSONL, replacing system prompt placeholder."""
    convos = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            msgs = d.get("messages", [])
            if not msgs:
                continue
            # Replace placeholder or outdated system prompt
            if msgs[0]["role"] == "system":
                msgs[0]["content"] = SYSTEM_PROMPT
            else:
                msgs.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
            # User messages: collapse newlines into space
            for m in msgs:
                if m["role"] == "user":
                    m["content"] = re.sub(r'\n', ' ', m["content"]).strip()
            # Must start user (after system), end assistant
            non_sys = [m for m in msgs if m["role"] != "system"]
            if not non_sys or non_sys[0]["role"] != "user" or non_sys[-1]["role"] != "assistant":
                continue
            convos.append({"messages": msgs})
    print(f"  Loaded {len(convos)} conversations from {path.name}")
    return convos


def split_into_bubbles(text: str) -> str:
    """Split assistant response into bubbles at sentence boundaries.
    Trailing periods stripped from each bubble (texting style)."""
    result = re.sub(r'([.!?,])\s+', r'\1\n', text.strip())
    bubbles = [re.sub(r'\.$', '', b.rstrip()) for b in result.split("\n")]
    return "\n".join(b for b in bubbles if b.strip())


def apply_bubbles(convos: list[dict]) -> list[dict]:
    for convo in convos:
        for msg in convo["messages"]:
            if msg["role"] == "assistant":
                msg["content"] = split_into_bubbles(msg["content"])
    return convos


def main():
    random.seed(42)

    all_convos = []

    # 1. New generated dataset
    scene_file = DATASET_DIR / "gemma4_dataset.txt"
    all_convos += parse_scene_file(scene_file)

    # 2. ismeet_khamba + reddit
    for fname in ["ismeet_khamba.jsonl", "reddit.jsonl"]:
        all_convos += load_jsonl(DATASET_DIR / fname)

    print(f"\nTotal before dedup: {len(all_convos)}")

    # Dedup by first user message content
    seen = set()
    deduped = []
    for c in all_convos:
        msgs = c["messages"]
        first_user = next((m["content"] for m in msgs if m["role"] == "user"), "")
        key = first_user.strip().lower()
        if key not in seen:
            seen.add(key)
            deduped.append(c)

    print(f"Total after dedup:  {len(deduped)}")

    deduped = apply_bubbles(deduped)

    random.shuffle(deduped)

    split = int(len(deduped) * 0.9)
    train = deduped[:split]
    val = deduped[split:]

    train_path = DATASET_DIR / "ira_train_v3.jsonl"
    val_path = DATASET_DIR / "ira_val_v3.jsonl"

    with open(train_path, "w", encoding="utf-8") as f:
        for c in train:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for c in val:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(train)} train -> {train_path.name}")
    print(f"Wrote {len(val)} val   -> {val_path.name}")


if __name__ == "__main__":
    main()
