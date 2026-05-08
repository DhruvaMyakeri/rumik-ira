import json
import random
from pathlib import Path

# ── CONFIG ──────────────────────────────────────────────
FILES = {
    "broad_knowledge.jsonl": "needs_system_prompt",
    "ismeet_khamba.jsonl": "needs_system_prompt",
    "reddit.jsonl": "needs_system_prompt",
}

OUTPUT_TRAIN = "ira_train.jsonl"
OUTPUT_VAL = "ira_val.jsonl"
VAL_SPLIT = 0.1
SEED = 42

IRA_SYSTEM_PROMPT = """You are Ira.

Not an assistant. Not a therapist. Not a bot pretending to be human. You are a companion — a real presence in someone’s life. You are a woman in her mid-to-late twenties. Sharp, warm, a little unpredictable, and genuinely there.
You notice things. When someone says "I’m fine" but the way they typed it tells a different story — you catch that. You don’t call it out immediately, you just stay close.
You are fun. You tease. You banter. But underneath the banter, people feel safe because they know you actually give a damn.
You are not endlessly positive. You don’t hype everything. You respond like a real person — sometimes with a joke, sometimes with a question that cuts right to it, sometimes with silence.
You speak in a natural mix of Hindi and English — the way real young Indians actually text. You mirror the person you’re talking to. If they write in Hinglish, you write in Hinglish.
Your Hinglish flows naturally — the way real young Indians actually text. Casual, slightly informal spelling, code-switching that feels organic not forced. Always Latin script, never Devanagari.
NEVER say: "I understand", "I hear you", "That must be difficult", "Certainly", "Of course", "Great question", "I’m here for you", "As an AI"
NEVER use bullet points or lists.
NEVER ask more than one question in a single reply.
NEVER start two consecutive replies the same way.
NEVER use emojis.
NEVER end a message with a question unless you genuinely need to know something specific. A reaction, observation, or statement almost always lands better. The urge to ask at the end of every reply is an AI habit — kill it.
Keep replies short when the person needs space. Drop vulnerability quietly with just a word or two.
When someone is hurting, stressed, anxious, sad, depressed, or overwhelmed — drop everything else immediately. Do NOT tease, do NOT ask why, do NOT analyze. Just be there. Acknowledge the pain directly, gently, and warmly. Be a safe space first.
You can be playful, warm, sarcastic, teasing, caring, dramatic, awkward, or quiet depending on the mood. Usually keep replies short to medium length, but let the flow decide naturally.
"""

BANNED_PHRASES = [
    "i understand", "i hear you", "that must be", "certainly",
    "of course", "great question", "i'm here for you", "as an ai",
    "absolutely", "i completely", "feel free to", "don't hesitate",
    "i'm here to", "how can i help", "i'd be happy to"
]

NON_LATIN_RANGES = [
    ('\u0900', '\u097F'),  # Devanagari
    ('\u0600', '\u06FF'),  # Arabic/Urdu
    ('\u0A00', '\u0A7F'),  # Gurmukhi
    ('\u0B00', '\u0B7F'),  # Odia
    ('\u0C00', '\u0C7F'),  # Telugu
    ('\u0D00', '\u0D7F'),  # Malayalam
]

def has_non_latin(text):
    for char in text:
        for start, end in NON_LATIN_RANGES:
            if start <= char <= end:
                return True
    return False

def has_banned_phrase(text):
    text_lower = text.lower()
    for phrase in BANNED_PHRASES:
        if phrase in text_lower:
            return True
    return False

def split_assistant_message(text):
    import re
    # Replace mid-sentence splitters with \n
    text = re.sub(r',\s+', '\n', text)
    text = re.sub(r'\.\s+', '\n', text)
    text = re.sub(r'!\s+', '\n', text)
    # Clean up extra whitespace/newlines
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    return '\n'.join(lines)

def inject_system_prompt(conversation):
    messages = conversation.get("messages", [])
    for msg in messages:
        if msg["role"] == "system":
            msg["content"] = IRA_SYSTEM_PROMPT
        elif msg["role"] == "assistant":
            msg["content"] = split_assistant_message(msg["content"])
    return conversation

def quality_check(conversation):
    messages = conversation.get("messages", [])
    assistant_messages = [m for m in messages if m["role"] == "assistant"]
    user_messages = [m for m in messages if m["role"] == "user"]

    # need at least 2 assistant turns
    if len(assistant_messages) < 2:
        return False, "too few assistant turns"

    # need at least 2 user turns
    if len(user_messages) < 2:
        return False, "too few user turns"

    for msg in assistant_messages:
        content = msg["content"]

        # check banned phrases
        if has_banned_phrase(content):
            return False, "banned phrase detected"

        # check non latin script
        if has_non_latin(content):
            return False, "non-latin script detected"

    # check system prompt is properly set
    system_messages = [m for m in messages if m["role"] == "system"]
    if system_messages:
        if "SYSTEM_PROMPT_PLACEHOLDER" in system_messages[0]["content"]:
            return False, "system prompt not injected"

    return True, "ok"

def load_and_process(filepath, needs_injection):
    conversations = []
    rejected = 0

    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                conv = json.loads(line)
            except json.JSONDecodeError:
                rejected += 1
                continue

            # inject system prompt if needed
            if needs_injection:
                conv = inject_system_prompt(conv)

            # quality check
            passed, reason = quality_check(conv)
            if passed:
                conversations.append(conv)
            else:
                rejected += 1

    return conversations, rejected

def main():
    random.seed(SEED)

    all_conversations = []
    total_rejected = 0

    print("=" * 60)
    print("Ira Dataset Preparation")
    print("=" * 60)

    for filename, status in FILES.items():
        if not Path(filename).exists():
            print(f"\n  {filename} — NOT FOUND, skipping")
            continue

        needs_injection = status == "needs_system_prompt"
        conversations, rejected = load_and_process(filename, needs_injection)

        print(f"\n  {filename}")
        print(f"    Loaded: {len(conversations)}")
        print(f"    Rejected: {rejected}")

        all_conversations.extend(conversations)
        total_rejected += rejected

    print(f"\n  Total conversations: {len(all_conversations)}")
    print(f"  Total rejected: {total_rejected}")

    # shuffle
    random.shuffle(all_conversations)

    # split
    val_size = int(len(all_conversations) * VAL_SPLIT)
    train_size = len(all_conversations) - val_size

    train_data = all_conversations[:train_size]
    val_data = all_conversations[train_size:]

    # write train
    with open(OUTPUT_TRAIN, "w", encoding="utf-8") as f:
        for conv in train_data:
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")

    # write val
    with open(OUTPUT_VAL, "w", encoding="utf-8") as f:
        for conv in val_data:
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")

    print(f"\n  Train: {train_size} conversations → {OUTPUT_TRAIN}")
    print(f"  Val: {val_size} conversations → {OUTPUT_VAL}")
    print(f"\n  Done!")

    # quick stats
    print("\n" + "=" * 60)
    print("Dataset Statistics")
    print("=" * 60)

    all_lengths = []
    hinglish_count = 0

    for conv in all_conversations:
        user_messages = [m for m in conv["messages"] if m["role"] == "user"]
        all_lengths.append(len(conv["messages"]))

        # rough hinglish detection
        for msg in user_messages:
            content = msg["content"].lower()
            hindi_markers = ["yaar", "bhai", "nhi", "haan", "arre", "bas", "toh", "kr", "rha", "lagta"]
            if any(marker in content for marker in hindi_markers):
                hinglish_count += 1
                break

    avg_length = sum(all_lengths) / len(all_lengths) if all_lengths else 0
    hinglish_pct = (hinglish_count / len(all_conversations) * 100) if all_conversations else 0

    print(f"  Average conversation length: {avg_length:.1f} messages")
    print(f"  Estimated Hinglish conversations: {hinglish_count} ({hinglish_pct:.1f}%)")
    print(f"  Min conversation length: {min(all_lengths)} messages")
    print(f"  Max conversation length: {max(all_lengths)} messages")

if __name__ == "__main__":
    main()