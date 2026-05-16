# Ira Pipeline Diagrams

---

## 1. Full Training & Serving Pipeline

```mermaid
flowchart TD
    subgraph DATA["Data Collection · 732 total"]
        D1["Ismeet/Khamba Podcasts\n147 convos"]
        D2["Reddit Posts\nr/India · r/bangalore\n111 convos"]
        D3["Synthetic via Groq\nllama-3.3-70b-versatile\n425 convos"]
        D4["Manual Curated\n49 convos"]
    end

    DATA --> PREP["Dataset Prep\n• remove Devanagari in assistant turns\n• remove banned phrases\n• max 1 question per reply\n• shuffle\n────────────────\n677 train / 68 val"]

    PREP --> SFT["QLoRA SFT\nGemma 3 12B Instruct\nrank 16 · alpha 32 · 2 epochs\nLR 2e-4 · effective batch 16\nA100 40GB · ~16 min"]

    SFT --> SFTLOSS["train loss 0.11\nval loss 0.55"]
    SFTLOSS --> CKPT["SFT Checkpoint v5\nira_sft_12b_checkpoint_v5_original"]

    CKPT --> DPO["DPO Attempt\n204 preference pairs\nbeta 0.05 · 1 epoch · LR 5e-5\nHuggingFace TRL stack"]
    DPO --> DPOFAIL["❌ Collapsed\nnonsensical one-liners\nnot used as final model"]

    CKPT --> SERVE

    subgraph SERVE["Modal.com Inference Service"]
        direction TB
        LOAD["Base: google/gemma-3-12b-it\n4-bit NF4 via BitsAndBytes\n+ PeftModel LoRA adapter\n~13GB VRAM · A100 40GB"]
        LOAD --> EP1["POST /chat\nnew_messages list\nconsecutive user turns"]
        LOAD --> EP2["POST /chat_image\nPIL image in processor\nnative multimodal"]
        LOAD --> EP3["POST /initiate\nIra speaks first\nor re-initiates after pause"]
        LOAD --> EP4["GET /health\nGPU · VRAM · latency"]
    end

    SERVE --> CLIENT["Web Client\nira_async.html"]
```

---

## 2. Backend Generation Flow

```mermaid
flowchart TD
    REQ["Incoming request\n/chat or /chat_image"]

    REQ --> BSP["build_system_prompt()\n• base Ira system prompt\n• append memory_context if set\n• append image instruction if /chat_image"]

    BSP --> BUILD["Build full_messages list\nsystem turn\n+ history turns\n+ new user turn\n  (text or image + text)"]

    BUILD --> PASS1["First pass — generate()\napply_chat_template → tokenize\nmodel.generate()\n  do_sample · temp · top_p\n  repetition_penalty 1.1\n  no_repeat_ngram_size 3"]

    PASS1 --> CLEAN1["clean_response()\nstrip role marker bleed\n(assistant / user / model)"]

    CLEAN1 --> CHECK{"should_continue()?\nor force_continue=True\n(image requests)"}

    CHECK -->|"No — short/neutral"| RETURN["Return [response1]"]

    CHECK -->|"Yes — emotional trigger\nor image"| CONT["Append response1 + continuation prompt\nto message history\n'send one short follow-up\n— different angle, not a question'"]

    CONT --> PASS2["Second pass — generate()\nmax_new_tokens // 2"]

    PASS2 --> CLEAN2["clean_response()"]

    CLEAN2 --> DEDUP{"response2 valid?\n≠ response1\nlen > 2 words"}

    DEDUP -->|Yes| RETURN2["Return [response1, response2]"]
    DEDUP -->|No| RETURN
```

---

## 3. Client Message Flow

```mermaid
flowchart TD
    INPUT["User presses Enter\nsendMessage()"]

    INPUT --> IMGCHECK{"selectedImage\nattached?"}

    IMGCHECK -->|Yes| IMGPATH["handleImageSend(text, pendingBatch)"]
    IMGCHECK -->|No| INFLIGHTCHECK{"flushInProgress?"}

    INFLIGHTCHECK -->|Yes| BUFFER["addToBuffer(text)\nshow ghost chip in buffer strip\n+ cancel button per chip"]
    INFLIGHTCHECK -->|No| CHAT["addMessage('user', text)\nshow bubble in chat"]

    BUFFER --> PENDING["pendingMessages.push(text)"]
    CHAT --> PENDING

    PENDING --> GATHER{"gatherStarted?"}

    GATHER -->|No - debounce phase| DEBOUNCE["clearTimeout(debounceTimer)\nsetTimeout(4000ms)\nwaiting for typing pause"]
    DEBOUNCE -->|pause detected| GATHERSTART["gatherStarted = true\ngatherStart = Date.now()\nsetTimeout(5000ms)"]

    GATHER -->|Yes - gather phase| GATHERRESET["reset gather window\nrespect 5s cap from gatherStart\nfire immediately if cap hit"]

    GATHERSTART --> FIREGATHER["fireGather()\nbatch = [...pendingMessages]\npendingMessages = []"]
    GATHERRESET --> FIREGATHER

    FIREGATHER --> FLUSH

    subgraph FLUSH["flushPending(batch)"]
        direction TB
        F1{"flushInProgress?"}
        F1 -->|Yes| QUEUE["queuedBatch = [...queuedBatch, ...batch]\nhold — drain after current call"]
        F1 -->|No| F2["flushInProgress = true\ndrainMsgBuffer()\naddTyping()"]
        F2 --> APICHAT["POST /chat\nhistory + new_messages\ntemperature from slider\nmemory_context if set"]
        APICHAT --> RESP["removeTyping()\nhistory.push user turn\nawait showLines(responses)\nhistory.push assistant turn"]
        RESP --> F3["flushInProgress = false"]
        F3 --> DRAIN{"queuedBatch?"}
        DRAIN -->|Yes| DRAINNEXT["next = queuedBatch\nqueuedBatch = null\nawait flushPending(next)"]
        DRAIN -->|No| DONE["done"]
    end

    subgraph IMGPATH["handleImageSend()"]
        direction TB
        I1["flushInProgress = true\nfileToBase64(imageFile)\nrender img-card in chat\nclearImage() pending strip\naddTyping()"]
        I1 --> I2{"pendingBefore\nmessages?"}
        I2 -->|Yes| I3["await flushPending(pendingBefore)\n(flush text context first)"]
        I2 -->|No| I4["POST /chat_image\nhistory + message\nimage_base64 + media_type"]
        I3 --> I4
        I4 --> I5["removeTyping()\nhistory.push user turn\nawait showLines(responses)\nhistory.push assistant turn\nflushInProgress = false"]
        I5 --> I6{"queuedBatch?"}
        I6 -->|Yes| I7["drain queuedBatch\nvia flushPending"]
        I6 -->|No| I8["done"]
    end
```

---

## 4. showLines — Staggered Bubble Reveal

```mermaid
flowchart LR
    R["responses array\nfrom API"]
    R --> FLAT["flatMap split on newline\ntrim · filter empty"]
    FLAT --> L1["line 0\naddMessage instantly"]
    L1 --> W1["wait 500–900ms\nrandom"]
    W1 --> L2["line 1\naddMessage"]
    L2 --> W2["wait 500–900ms"]
    W2 --> LN["line N\naddMessage"]
```

---

## 5. Consecutive Message Timing

```
User:  ──●────●──────────────────●──────────────●───────────────────────────────────▶
          msg1 msg2               msg3           msg4
                  └──4s debounce──┘
                                   └─gather starts─┤
                                                    └──5s gather window──┘
                                                                          └─► API call
                                                                              new_messages:
                                                                              [msg1, msg2, msg3, msg4]
```
