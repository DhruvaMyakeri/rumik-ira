import json
import random

random.seed(42)
all_convs = []

with open("ira_pairs_sft.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            all_convs.append(json.loads(line))

print(f"Total: {len(all_convs)}")
random.shuffle(all_convs)

split = int(len(all_convs) * 0.9)
train = all_convs[:split]
val = all_convs[split:]

with open("ira_train_v5.jsonl", "w", encoding="utf-8") as f:
    for conv in train:
        f.write(json.dumps(conv, ensure_ascii=False) + "\n")

with open("ira_val_v5.jsonl", "w", encoding="utf-8") as f:
    for conv in val:
        f.write(json.dumps(conv, ensure_ascii=False) + "\n")

print(f"Train: {len(train)} | Val: {len(val)}")