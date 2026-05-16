import json

with open("generated_200.json") as f:
    data = json.load(f)

print(f"Total conversations: {len(data)}")


# check first message of each to see variety
for i, conv in enumerate(data):
    print(f"{i+1}. {conv['messages'][1]['content'][:50]}")