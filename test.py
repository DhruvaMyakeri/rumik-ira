# import requests

# res = requests.post(
#     "https://rumik-ai-2--ira-inference-service-api.modal.run/chat",
#     json={"new_messages": ["hi", "kaisi ho", "kuch bata"], "history": []}
# )
# print(res.json())

import requests

res = requests.post(
    "https://rumik-ai-2--ira-inference-service-api.modal.run/chat",
    json={
        "new_messages": ["promotion mil gayi yaar finally"],
        "history": []
    }
)
print(res.json())