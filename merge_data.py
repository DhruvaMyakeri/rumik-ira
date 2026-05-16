"""
Merge all SFT data sources, shuffle, and split 90/10.

Sources:
- ira_train.jsonl (old training data)
- ira_val.jsonl (old validation data)
- ira_initiation_conversations.json (20 curated initiation)
- generated_filtered.json (filtered generated initiation conversations)
"""

import json
import random

random.seed(42)

all_conversations = []

# load old training data
print("Loading old ira_train.jsonl...")
with open("ira_train_v3.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            all_conversations.append(json.loads(line))
print(f"Old train: {len(all_conversations)}")

# load old validation data
old_count = len(all_conversations)
print("Loading old ira_val.jsonl...")
with open("ira_train.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            all_conversations.append(json.loads(line))
print(f"Old val: {len(all_conversations) - old_count}")

count_before = len(all_conversations)
print("Loading generated_filtered.json...")
with open("generated_filtered.json", "r", encoding="utf-8") as f:
    generated = json.load(f)
all_conversations.extend(generated)
print(f"Generated initiation: {len(all_conversations) - count_before}")

print(f"\nTotal before split: {len(all_conversations)}")

# shuffle
random.shuffle(all_conversations)

# split 90/10
split_idx = int(len(all_conversations) * 0.9)
train = all_conversations[:split_idx]
val = all_conversations[split_idx:]

print(f"Train: {len(train)}")
print(f"Val: {len(val)}")

# save
with open("ira_train_v5.jsonl", "w", encoding="utf-8") as f:
    for conv in train:
        f.write(json.dumps(conv, ensure_ascii=False) + "\n")

with open("ira_val_v5.jsonl", "w", encoding="utf-8") as f:
    for conv in val:
        f.write(json.dumps(conv, ensure_ascii=False) + "\n")

print("\nSaved:")
print(f"ira_train_v4.jsonl — {len(train)} conversations")
print(f"ira_val_v4.jsonl — {len(val)} conversations")