"""
Ira Chat Client
Usage: python chat.py --url https://rumik-ai-2--ira-inference-service-api.modal.run

Commands:
  /image <path>  — attach image to next message
  /reset         — clear conversation
  quit           — exit
"""

import requests
import argparse
import base64
import os
import time

URL = None
history = []

def post(endpoint, payload, timeout=180):
    try:
        res = requests.post(f"{URL}/{endpoint}", json=payload, timeout=timeout)
        res.raise_for_status()
        return res.json()
    except requests.exceptions.Timeout:
        print("\n[timed out — try again]")
        return None
    except Exception as e:
        print(f"\n[error: {e}]")
        return None

def print_ira(responses, metrics=None):
    for i, msg in enumerate(responses):
        if i > 0:
            time.sleep(0.8)
        print(f"Ira: {msg}")
    if metrics:
        print(f"     [{metrics['latency_seconds']}s | {metrics['tokens_per_second']} tok/s]\n")
    else:
        print()

def initiate(fresh=True):
    """Call /initiate — fresh=True for empty history, False for mid-conversation"""
    payload = {"history": [] if fresh else history}
    result = post("initiate", payload, timeout=60)
    if result and result.get("responses"):
        msg = result["responses"][0]
        if msg:
            print_ira([msg])
            history.append({"role": "assistant", "content": msg})

def send_chat(messages, image_path=None):
    global history

    if image_path:
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()
        ext = os.path.splitext(image_path)[1].lower().lstrip(".")
        media_type = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}.get(ext, "image/jpeg")
        user_text = " ".join(messages) if messages else ""

        payload = {
            "history": history,
            "message": user_text,
            "image_base64": img_b64,
            "media_type": media_type,
        }
        result = post("chat_image", payload)
        if result:
            history.append({"role": "user", "content": user_text or "[image]"})
            history.append({"role": "assistant", "content": "\n".join(result["responses"])})
            print_ira(result["responses"], result.get("metrics"))
    else:
        payload = {
            "history": history,
            "new_messages": messages,
        }
        result = post("chat", payload)
        if result:
            history.append({"role": "user", "content": "\n".join(messages)})
            history.append({"role": "assistant", "content": "\n".join(result["responses"])})
            print_ira(result["responses"], result.get("metrics"))

def main():
    global URL, history

    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--no-initiate", action="store_true", help="Skip opening initiation")
    args = parser.parse_args()
    URL = args.url.rstrip("/")

    # health check
    try:
        h = requests.get(f"{URL}/health", timeout=30).json()
        print(f"[{h['gpu']} | {h['vram_total_gb']}GB VRAM]\n")
    except:
        print("[could not reach health endpoint]\n")

    print("Chat with Ira — /image <path>, /reset, quit\n" + "─" * 50)

    # Ira speaks first
    if not args.no_initiate:
        initiate(fresh=True)

    pending_image = None

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue

        if user_input.lower() == "quit":
            break

        if user_input.lower() == "/reset":
            history = []
            pending_image = None
            print("[reset]\n")
            initiate(fresh=True)
            continue

        if user_input.startswith("/image "):
            path = user_input[7:].strip()
            if not os.path.exists(path):
                print(f"[file not found: {path}]\n")
                continue
            pending_image = path
            print(f"[image attached: {os.path.basename(path)}]")
            print("Now type your message (or press Enter to send just the image):")
            continue

        send_chat([user_input], pending_image)
        pending_image = None

if __name__ == "__main__":
    main()