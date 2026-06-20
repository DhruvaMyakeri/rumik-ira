import base64
import requests

with open("image1.jpg", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

res = requests.post(
    "https://rumik-ai-2--ira-inference-service-api.modal.run/chat_image",
    json={
        "message": "",
        "image_base64": img_b64,
        "history": []
    }
)
print(res.json())