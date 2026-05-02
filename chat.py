"""
Ira Chat Client
Usage: python chat.py --url https://your-modal-url
       python chat.py --url https://your-modal-url --memory "user has exam tomorrow"

Commands during chat:
  /image <path>  — upload an image file and send it with your next message
  reset          — clear conversation history
  quit           — exit
"""

import requests
import argparse
import base64
import os


def describe_image(url: str, image_path: str) -> str:
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    ext = os.path.splitext(image_path)[1].lower()
    media_type_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }
    media_type = media_type_map.get(ext, "image/jpeg")
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    print("Describing image...")
    response = requests.post(
        f"{url}/describe",
        json={"image_base64": image_b64, "media_type": media_type},
        timeout=60
    )
    response.raise_for_status()
    data = response.json()
    print(f"[Image description: {data['description']}]")
    return data["description"]


def chat_with_ira(url: str, memory_context: str = None):
    history = []
    pending_image = None

    print("\nChatting with Ira (type 'quit' to exit, 'reset' to clear history)")
    print("─" * 50)

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "reset":
            history = []
            pending_image = None
            print("[Conversation reset]\n")
            continue

        if user_input.startswith("/image "):
            image_path = user_input[7:].strip()
            if not os.path.exists(image_path):
                print(f"[File not found: {image_path}]\n")
                continue
            try:
                pending_image = describe_image(url, image_path)
                print("Image ready. Now type your message to send with it.")
                continue
            except Exception as e:
                print(f"[Image error: {e}]\n")
                continue

        history.append({"role": "user", "content": user_input})

        payload = {
            "messages": history,
            "temperature": 0.8,
            "max_new_tokens": 150,
        }
        if memory_context:
            payload["memory_context"] = memory_context
        if pending_image:
            payload["image_description"] = pending_image
            pending_image = None

        try:
            response = requests.post(f"{url}/chat", json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()

            ira_response = data["response"]
            metrics = data["metrics"]

            print(f"Ira: {ira_response}")
            print(f"     [{metrics['latency_seconds']}s | {metrics['tokens_per_second']} tok/s | {metrics['gpu_memory_used_gb']}GB VRAM]\n")

            history.append({"role": "assistant", "content": ira_response})

        except requests.exceptions.Timeout:
            print("[Request timed out — model might be cold starting, try again]\n")
            history.pop()
        except Exception as e:
            print(f"[Error: {e}]\n")
            history.pop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="Modal deployment URL")
    parser.add_argument("--memory", default=None, help="Memory context for Ira")
    args = parser.parse_args()

    try:
        health = requests.get(f"{args.url}/health", timeout=30).json()
        print(f"Service: {health['status']} | GPU: {health['gpu']} | VRAM: {health['vram_total_gb']}GB")
    except:
        print("Warning: Could not reach health endpoint. Proceeding anyway...")

    chat_with_ira(args.url, args.memory)