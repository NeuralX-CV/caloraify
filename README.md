# 🍽 CaLoRAify — AI Food Calorie Tracker

<div align="center">

![CaLoRAify Banner](https://img.shields.io/badge/CaLoRAify-AI%20Nutrition%20Tracker-orange?style=for-the-badge&logo=telegram)


**Send a food photo → Get instant calories, macros & meal tracking**

*Built with SmolVLM2-500M fine-tuned via LoRA on 2000 real food images*

</div>

---

## 📸 Demo

```
User sends photo of banana
         ↓
🍽 Meal Analysis
🥦 Ingredients: banana
⚖️ Portions: Portion estimated from image.
🔥 Calories: 89 kcal
💪 Protein:  1.1 g
🌾 Carbs:    23.0 g
🥑 Fat:      0.3 g
🌿 Fibre:    2.6 g
Meal ID #1 — logged ✅
📊 Today so far: 89 / 2000 kcal
█░░░░░░░░░░░ 4%
```

---

## 🏗 System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERFACES                          │
│                                                                 │
│   📱 Telegram App          🌐 Any HTTP Client                   │
│   (send food photo)        (REST API calls)                     │
└──────────────┬──────────────────────────┬───────────────────────┘
               │                          │
               ▼                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                     TELEGRAM BOT LAYER                          │
│                    telegram_bot.py                              │
│                                                                 │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────────┐ │
│  │ Photo Handler│  │ CMD Handlers │  │   Callback Buttons     │ │
│  │ /analyze    │  │ /today /week │  │ [Today] [Week] [Delete]│ │
│  │             │  │ /streak /goal│  │                        │ │
│  └──────┬──────┘  └──────┬───────┘  └──────────┬─────────────┘ │
│         │                │                      │               │
│         ▼                ▼                      ▼               │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    SQLite Database                          ││
│  │     meals table          user_settings table               ││
│  │  (id, user_id, date,   (user_id, daily_goal,              ││
│  │   calories, macros)     timezone)                          ││
│  └─────────────────────────────────────────────────────────────┘│
└──────────────┬──────────────────────────────────────────────────┘
               │  POST /analyze
               │  {image_b64, max_new_tokens}
               │  Header: x-api-key
               ▼
┌─────────────────────────────────────────────────────────────────┐
│              HUGGINGFACE SPACE — FastAPI Server                 │
│                      space_app.py                               │
│                                                                 │
│  ┌──────────────┐    ┌─────────────────┐    ┌───────────────┐  │
│  │  /health     │    │   /analyze      │    │    /debug     │  │
│  │  GET         │    │   POST          │    │    POST       │  │
│  └──────────────┘    └────────┬────────┘    └───────────────┘  │
│                               │                                 │
│                               ▼                                 │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              _run_inference()                               ││
│  │                                                             ││
│  │  base64 image → PIL.Image → processor → model → text       ││
│  └──────────────────────┬──────────────────────────────────────┘│
│                         │                                       │
│                         ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │           SmolVLM2-500M + LoRA Adapter                      ││
│  │                                                             ││
│  │  • 4-bit NF4 quantisation (bitsandbytes)                    ││
│  │  • LoRA rank=16, alpha=32                                   ││
│  │  • Target: q_proj, v_proj, k_proj                           ││
│  │  • Fine-tuned on 2000 real food images                      ││
│  └──────────────────────┬──────────────────────────────────────┘│
│                         │                                       │
│                         ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              _parse_response()                              ││
│  │                                                             ││
│  │  Strategy 1: CaLoRAify structured format                    ││
│  │  Strategy 2: Food keyword scanning                          ││
│  │  Strategy 3: Open Food Facts API fallback                   ││
│  └──────────────────────┬──────────────────────────────────────┘│
└─────────────────────────┼───────────────────────────────────────┘
                          │
                          ▼
              JSON Response to Bot
         {ingredients, calories, macros}
```

---

## 🔄 Request Flow (Step by Step)

```
STEP 1: User sends food photo on Telegram
        │
        ▼
STEP 2: Bot downloads photo (highest resolution)
        │
        ▼
STEP 3: Bot sends "🔍 Analysing your meal..." message
        │
        ▼
STEP 4: Bot encodes image to base64
        │
        ▼
STEP 5: POST https://unnatrathi-caloraify.hf.space/analyze
        Body: { image_b64: "...", max_new_tokens: 300 }
        Header: x-api-key: caloraify2024
        │
        ▼
STEP 6: FastAPI decodes base64 → PIL Image (384×384)
        │
        ▼
STEP 7: SmolVLM2 processes image + prompt
        Prompt: "What food is in this image?
                 Reply: Ingredients detected: [list]"
        │
        ▼
STEP 8: Model outputs text
        e.g. "Ingredients detected: banana, rice"
        │
        ▼
STEP 9: Parser extracts ingredients
        → Looks up nutrition table / Open Food Facts API
        → Returns { calories: 89, protein: 1.1, ... }
        │
        ▼
STEP 10: Response JSON sent back to bot
         { ingredients, calories, protein_g, carbs_g, fat_g }
         latency_ms: 35000
        │
        ▼
STEP 11: Bot logs meal to SQLite
         INSERT INTO meals (user_id, calories, ...)
        │
        ▼
STEP 12: Bot calculates today's total
         SELECT SUM(calories) WHERE log_date = today
        │
        ▼
STEP 13: Bot sends formatted reply card with:
         • Ingredients list
         • Full macro breakdown
         • Daily progress bar
         • Inline buttons [Today] [Week] [Delete]
        │
        ▼
STEP 14: User sees result ✅
```

---

## 🧠 Model Training Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE                            │
│                 caloraify_finetune_v4.py                        │
└─────────────────────────────────────────────────────────────────┘

  1. DATASET
     ┌─────────────────────────────────────┐
     │ Codatta/MM-Food-100K (HuggingFace)  │
     │ 2000 real food images               │
     │ columns: image_url, ingredients,    │
     │          portion_size,              │
     │          nutritional_profile        │
     └──────────────┬──────────────────────┘
                    │
                    ▼
  2. FORMAT (CaLoRAify Reasoning Loop)
     ┌─────────────────────────────────────┐
     │ USER:                               │
     │   <image>                           │
     │   "Analyze this food image..."      │
     │                                     │
     │ ASSISTANT:                          │
     │   "Ingredients detected: chicken,   │
     │    rice, broccoli.                  │
     │    Portion Analysis: ~300g plate.   │
     │    JSON Summary: {calories: 520,    │
     │    protein_g: 42, ...}"             │
     └──────────────┬──────────────────────┘
                    │
                    ▼
  3. MODEL LOADING
     ┌─────────────────────────────────────┐
     │ SmolVLM2-500M-Instruct              │
     │ • 4-bit NF4 quantisation            │
     │ • bfloat16 compute dtype            │
     │ • double quantisation               │
     │ • device_map="auto" (T4 GPU)        │
     └──────────────┬──────────────────────┘
                    │
                    ▼
  4. LORA CONFIG
     ┌─────────────────────────────────────┐
     │ rank (r)      = 16                  │
     │ alpha         = 32                  │
     │ dropout       = 0.05                │
     │ target_modules: q_proj, v_proj,     │
     │                 k_proj              │
     │ trainable params: 3,178,496 (0.62%) │
     └──────────────┬──────────────────────┘
                    │
                    ▼
  5. TRAINING (SFTTrainer)
     ┌─────────────────────────────────────┐
     │ batch_size    = 2                   │
     │ grad_accum    = 4 (effective: 8)    │
     │ learning_rate = 2e-4                │
     │ epochs        = 5                   │
     │ optimizer     = paged_adamw_8bit    │
     │ scheduler     = cosine              │
     │ grad_checkpt  = True                │
     └──────────────┬──────────────────────┘
                    │
                    ▼
  6. OUTPUT
     ┌─────────────────────────────────────┐
     │ adapter_model.safetensors (~40 MB)  │
     │ adapter_config.json                 │
     │ tokenizer files                     │
     │ → Uploaded to HuggingFace Hub       │
     └─────────────────────────────────────┘
```

---

## 📁 Project Structure

```
caloraify/
│
├── 📓 caloraify_finetune_v4.py     # Google Colab training notebook
│   ├── Block 1  — Install deps
│   ├── Block 2  — Imports & checks
│   ├── Block 3  — GPU memory reporter
│   ├── Block 4  — Constants (LoRA, training config)
│   ├── Block 5  — BitsAndBytes 4-bit config
│   ├── Block 6  — Load model + registry patch
│   ├── Block 7  — dtype alignment (bf16 fix)
│   ├── Block 8  — prepare_model + LoRA wrapping
│   ├── Block 9  — Codatta dataset (2000 food images)
│   ├── Block 10 — Chat template formatting
│   ├── Block 11 — VLMCollator
│   ├── Block 12 — SFTConfig
│   ├── Block 13 — SFTTrainer init
│   ├── Block 14 — Training
│   ├── Block 15 — Save adapter
│   └── Block 16 — Inference smoke test
│
├── 🚀 space_app.py                 # HuggingFace Space FastAPI server
│   ├── lifespan()      — model loading on startup
│   ├── /health         — GET endpoint for uptime check
│   ├── /analyze        — POST endpoint (main inference)
│   ├── /debug          — POST endpoint (raw model output)
│   ├── _run_inference()— SmolVLM2 forward pass
│   ├── _parse_response()— extract ingredients + nutrition
│   └── _get_nutrition_from_api() — Open Food Facts lookup
│
├── 🤖 telegram_bot.py              # Telegram bot
│   ├── init_db()       — SQLite setup
│   ├── log_meal()      — save meal to DB
│   ├── get_daily_summary() — today's totals
│   ├── get_streak()    — consecutive logging days
│   ├── get_weekly_summary() — 7-day history
│   ├── /start /help    — welcome message
│   ├── /today          — daily calorie summary
│   ├── /week           — 7-day bar chart
│   ├── /streak         — logging streak
│   ├── /goal [cal]     — set daily target
│   ├── /delete [id]    — remove a meal
│   ├── handle_photo()  — main photo handler
│   └── handle_callback()— inline button handler
│
├── 🐳 Dockerfile                   # HF Space container config
├── 📦 requirements_space.txt       # Space dependencies
├── 📦 requirements_bot.txt         # Bot dependencies
└── 📖 README.md                    # This file
```

---

## ⚙️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Base Model | SmolVLM2-500M-Instruct | Vision-language understanding |
| Fine-tuning | LoRA (PEFT) | Parameter-efficient adaptation |
| Quantisation | 4-bit NF4 (bitsandbytes) | Fit model in 15GB VRAM |
| Training | SFTTrainer (TRL) | Supervised fine-tuning |
| Dataset | Codatta/MM-Food-100K | 100K real food images |
| Inference Server | FastAPI + Uvicorn | REST API on HF Spaces |
| Hosting | HuggingFace Spaces (Docker) | Free cloud inference |
| Bot Framework | python-telegram-bot 21.3 | Telegram integration |
| HTTP Client | httpx (async) | Bot → Space API calls |
| Database | SQLite | Meal logs & user settings |
| Nutrition API | Open Food Facts | Calorie lookup fallback |

---

## 🚀 Deployment Guide

### Prerequisites
- Google Colab account (free T4 GPU)
- HuggingFace account (free)
- Telegram account
- GitHub account

---

### Step 1 — Train the Model (Google Colab)

```bash
# Open caloraify_finetune_v4.py in Google Colab
# Runtime → Change runtime type → T4 GPU
# Run all blocks sequentially (takes ~2-3 hours for 2000 samples)
```

### Step 2 — Upload LoRA Adapter to HuggingFace Hub

```python
from huggingface_hub import HfApi, login

login()  # enter your HF token

api = HfApi()
api.create_repo("caloraify-lora-adapter", repo_type="model", exist_ok=True)
api.upload_folder(
    folder_path="./caloraify-smolvlm2-lora",
    repo_id="YOUR_USERNAME/caloraify-lora-adapter",
    repo_type="model",
)
```

### Step 3 — Deploy to HuggingFace Spaces

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Name: `caloraify` | SDK: **Docker** | Template: **Blank**
3. Hardware: **CPU Basic** (free)
4. Upload files:

```
space_app.py        → rename to app.py
Dockerfile
requirements_space.txt
```

5. Add secrets in **Settings → Variables and Secrets**:

```
HF_MODEL_ID  = HuggingFaceTB/SmolVLM2-500M-Instruct
LORA_REPO_ID = YOUR_USERNAME/caloraify-lora-adapter
API_KEY      = your_secret_key
```

6. Verify deployment:
```bash
curl https://YOUR_USERNAME-caloraify.hf.space/health
# Expected: {"status":"ok","model_loaded":true,"cuda":false}
```

### Step 4 — Create Telegram Bot

1. Message [@BotFather](https://t.me/BotFather) on Telegram
2. Send `/newbot` → follow prompts → copy token

### Step 5 — Run the Bot

```bash
pip install -r requirements_bot.txt

export TELEGRAM_BOT_TOKEN="your_bot_token"
export CALORAIFY_API_URL="https://your_username-caloraify.hf.space"
export CALORAIFY_API_KEY="your_secret_key"

python telegram_bot.py
```

---

## 🤖 Bot Commands

| Command | Description | Example |
|---------|-------------|---------|
| *(send a photo)* | Analyse meal → log calories | — |
| `/start` | Welcome message + command list | — |
| `/today` | Today's calorie & macro summary | — |
| `/week` | 7-day calorie bar chart | — |
| `/streak` | Current logging streak | — |
| `/goal [calories]` | Set daily calorie target | `/goal 1800` |
| `/delete [id]` | Remove a logged meal | `/delete 5` |
| `/help` | Show all commands | — |

---

## 🗄 Database Schema

```sql
CREATE TABLE meals (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL,
    username    TEXT,
    logged_at   TEXT NOT NULL,     -- ISO datetime
    log_date    TEXT NOT NULL,     -- YYYY-MM-DD
    ingredients TEXT,
    portions    TEXT,
    calories    REAL,
    protein_g   REAL,
    carbs_g     REAL,
    fat_g       REAL,
    fibre_g     REAL,
    raw_text    TEXT               -- raw model output
);

CREATE TABLE user_settings (
    user_id     INTEGER PRIMARY KEY,
    username    TEXT,
    daily_goal  INTEGER DEFAULT 2000,
    timezone    TEXT DEFAULT 'UTC'
);
```

---

## 🐛 Known Issues & Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `SmolVLMConfig unrecognized` | Not in AutoModel registry (4.51.3) | `AutoModelForVision2Seq.register(..., exist_ok=True)` |
| `<image> tokens not divisible` | Truncation cuts patch blocks | `truncation=False` + `skip_prepare_dataset=True` |
| `dtype mismatch BFloat16/Float` | Vision encoder outputs float32 | Cast non-quantised params + forward hook |
| `requires_grad error` | Gradient checkpointing before LoRA | `prepare_model_for_kbit_training` before `get_peft_model` |
| `num2words ImportError` | Missing dependency | `pip install num2words` |

---

## 📊 Training Results

| Metric | Value |
|--------|-------|
| Base model params | 510,660,800 |
| Trainable LoRA params | 3,178,496 (0.62%) |
| Training samples | 2,000 |
| Epochs | 5 |
| Final training loss | ~2.5 |
| Adapter size | ~40 MB |
| Inference time (CPU) | 30–90 seconds |

---

## 🔮 Future Improvements

- [ ] Train on larger dataset (10K+ samples)
- [ ] Add Nutritionix API for accurate portion-based calories
- [ ] WhatsApp integration (Twilio)
- [ ] Weekly PDF nutrition report export
- [ ] Upgrade to GPU Space for faster inference
- [ ] Add food photo history gallery
- [ ] Multi-language support

---

## 📄 License

MIT License — feel free to use, modify and distribute.

---

## 🙏 Acknowledgements

- [HuggingFaceTB/SmolVLM2](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Instruct) — base vision-language model
- [Codatta/MM-Food-100K](https://huggingface.co/datasets/Codatta/MM-Food-100K) — training dataset
- [Open Food Facts](https://world.openfoodfacts.org) — nutrition database
- [TRL Library](https://github.com/huggingface/trl) — SFTTrainer
- [PEFT Library](https://github.com/huggingface/peft) — LoRA implementation

---

<div align="center">
Built with ❤️ using HuggingFace, FastAPI and python-telegram-bot
</div>
