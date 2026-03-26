
import os
import io
import json
import base64
import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import re
import requests as req_lib

# ── Lazy imports for heavy ML deps ──────────────────────────────────────────
_processor = None
_model      = None

logger = logging.getLogger("caloraify")
logging.basicConfig(level=logging.INFO)


# ── Startup / shutdown ───────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _processor, _model

    from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
    from transformers.models.smolvlm.configuration_smolvlm import SmolVLMConfig
    from transformers.models.smolvlm.modeling_smolvlm import SmolVLMForConditionalGeneration
    from bitsandbytes.nn import Linear4bit
    from peft import PeftModel

    MODEL_ID    = os.environ.get("HF_MODEL_ID",  "HuggingFaceTB/SmolVLM2-500M-Instruct")
    LORA_REPO   = os.environ.get("LORA_REPO_ID", "")

    # Registry patch (same fix as training notebook)
    AutoModelForVision2Seq.register(SmolVLMConfig, SmolVLMForConditionalGeneration, exist_ok=True)

    logger.info("Loading processor …")
    _processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    # Use 4-bit if CUDA available, else fp32 on CPU
    if torch.cuda.is_available():
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        dtype = torch.bfloat16
        logger.info("Loading model in 4-bit NF4 on GPU …")
    else:
        bnb   = None
        dtype = torch.float32
        logger.info("No GPU — loading model in fp32 on CPU (slower) …")

    base = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        quantization_config=bnb,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
        _fast_init=False,
    )

    # Cast vision encoder to bfloat16 (same fix as training notebook)
    if torch.cuda.is_available():
        for module in base.modules():
            if isinstance(module, Linear4bit):
                continue
            for p in module.parameters(recurse=False):
                if p.dtype == torch.float32:
                    p.data = p.data.to(torch.bfloat16)

    # Load LoRA adapter if provided
    if LORA_REPO:
        logger.info(f"Loading LoRA adapter from {LORA_REPO} …")
        _model = PeftModel.from_pretrained(base, LORA_REPO)
    else:
        logger.warning("No LORA_REPO_ID set — running base model without fine-tuning")
        _model = base

    _model.eval()
    _model.config.use_cache = True
    logger.info("Model ready ✅")

    yield  

    logger.info("Shutting down …")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="CaLoRAify Inference API",
    description="Food photo → ingredient list + calorie JSON",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


# ── Auth dependency ───────────────────────────────────────────────────────────
def verify_api_key(x_api_key: Optional[str] = Header(default=None)):
    expected = os.environ.get("API_KEY", "")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")


# ── Request / response models ─────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    image_b64: str          # base64-encoded JPEG/PNG
    max_new_tokens: int = 300

class NutritionResponse(BaseModel):
    ingredients: str
    portion_notes: str
    calories:    Optional[float] = None
    protein_g:   Optional[float] = None
    carbs_g:     Optional[float] = None
    fat_g:       Optional[float] = None
    fibre_g:     Optional[float] = None
    raw_text:    str            # full model output for debugging
    latency_ms:  int


# ── Inference helper ──────────────────────────────────────────────────────────
def _decode_image(b64: str) -> Image.Image:
    try:
        data = base64.b64decode(b64)
        img  = Image.open(io.BytesIO(data)).convert("RGB")
        img  = img.resize((384, 384))
        return img
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")


def _run_inference(image: Image.Image, max_new_tokens: int) -> str:
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": (
                        "What food are in this image? "
                        "Reply in exactly this format:\n"
                        "Ingredients detected: [food name and ingredients separated by commas]\n"
                        "Example: Ingredients detected: banana, rice, chicken, broccoli"
                    ),
                },
            ],
        }
    ]
    prompt = _processor.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )
    inputs = _processor(
        images=[[image]],
        text=[prompt],
        return_tensors="pt",
        truncation=False,
    )
    device = next(_model.parameters()).device
    if "pixel_values" in inputs and inputs["pixel_values"] is not None:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.float32)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        out_ids = _model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
            temperature=1.0,
            repetition_penalty=1.3,
            no_repeat_ngram_size=3,
        )

    new_tokens = out_ids[:, inputs["input_ids"].shape[-1]:]
    return _processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()


def _get_nutrition_from_api(ingredients_text: str) -> dict:
    NUTRITION_TABLE = {
        "banana":     {"calories": 89,  "protein_g": 1.1, "carbs_g": 23.0, "fat_g": 0.3, "fibre_g": 2.6},
        "apple":      {"calories": 72,  "protein_g": 0.4, "carbs_g": 19.0, "fat_g": 0.2, "fibre_g": 2.4},
        "orange":     {"calories": 62,  "protein_g": 1.2, "carbs_g": 15.4, "fat_g": 0.2, "fibre_g": 3.1},
        "mango":      {"calories": 99,  "protein_g": 1.4, "carbs_g": 25.0, "fat_g": 0.6, "fibre_g": 2.6},
        "grape":      {"calories": 69,  "protein_g": 0.7, "carbs_g": 18.1, "fat_g": 0.2, "fibre_g": 0.9},
        "strawberry": {"calories": 32,  "protein_g": 0.7, "carbs_g": 7.7,  "fat_g": 0.3, "fibre_g": 2.0},
        "watermelon": {"calories": 30,  "protein_g": 0.6, "carbs_g": 7.6,  "fat_g": 0.2, "fibre_g": 0.4},
        "pineapple":  {"calories": 50,  "protein_g": 0.5, "carbs_g": 13.0, "fat_g": 0.1, "fibre_g": 1.4},
        "coconut":    {"calories": 354, "protein_g": 3.3, "carbs_g": 15.0, "fat_g": 33.0,"fibre_g": 9.0},
        "lettuce":    {"calories": 15,  "protein_g": 1.4, "carbs_g": 2.9,  "fat_g": 0.2, "fibre_g": 1.3},
        "salad":      {"calories": 20,  "protein_g": 1.8, "carbs_g": 3.6,  "fat_g": 0.3, "fibre_g": 2.0},
        "carrot":     {"calories": 41,  "protein_g": 0.9, "carbs_g": 10.0, "fat_g": 0.2, "fibre_g": 2.8},
        "onion":      {"calories": 40,  "protein_g": 1.1, "carbs_g": 9.3,  "fat_g": 0.1, "fibre_g": 1.7},
        "rice":       {"calories": 206, "protein_g": 4.3, "carbs_g": 45.0, "fat_g": 0.4, "fibre_g": 0.6},
        "chicken":    {"calories": 239, "protein_g": 27.0,"carbs_g": 0.0,  "fat_g": 14.0,"fibre_g": 0.0},
        "egg":        {"calories": 155, "protein_g": 13.0,"carbs_g": 1.1,  "fat_g": 11.0,"fibre_g": 0.0},
        "bread":      {"calories": 265, "protein_g": 9.0, "carbs_g": 49.0, "fat_g": 3.2, "fibre_g": 2.7},
        "milk":       {"calories": 61,  "protein_g": 3.2, "carbs_g": 4.8,  "fat_g": 3.3, "fibre_g": 0.0},
        "cheese":     {"calories": 402, "protein_g": 25.0,"carbs_g": 1.3,  "fat_g": 33.0,"fibre_g": 0.0},
        "pizza":      {"calories": 266, "protein_g": 11.0,"carbs_g": 33.0, "fat_g": 10.0,"fibre_g": 2.3},
        "burger":     {"calories": 295, "protein_g": 17.0,"carbs_g": 24.0, "fat_g": 14.0,"fibre_g": 1.3},
        "pasta":      {"calories": 220, "protein_g": 8.1, "carbs_g": 43.0, "fat_g": 1.3, "fibre_g": 2.5},
        "fish":       {"calories": 136, "protein_g": 20.0,"carbs_g": 0.0,  "fat_g": 6.0, "fibre_g": 0.0},
        "potato":     {"calories": 77,  "protein_g": 2.0, "carbs_g": 17.0, "fat_g": 0.1, "fibre_g": 2.2},
        "broccoli":   {"calories": 34,  "protein_g": 2.8, "carbs_g": 6.6,  "fat_g": 0.4, "fibre_g": 2.6},
        "tomato":     {"calories": 18,  "protein_g": 0.9, "carbs_g": 3.9,  "fat_g": 0.2, "fibre_g": 1.2},
        "sandwich":   {"calories": 250, "protein_g": 12.0,"carbs_g": 33.0, "fat_g": 7.0, "fibre_g": 2.5},
        "soup":       {"calories": 71,  "protein_g": 3.8, "carbs_g": 8.0,  "fat_g": 2.0, "fibre_g": 1.5},
        "chocolate":  {"calories": 546, "protein_g": 5.0, "carbs_g": 60.0, "fat_g": 31.0,"fibre_g": 7.0},
        "cake":       {"calories": 347, "protein_g": 5.0, "carbs_g": 55.0, "fat_g": 12.0,"fibre_g": 1.0},
        "dal":        {"calories": 116, "protein_g": 9.0, "carbs_g": 20.0, "fat_g": 0.4, "fibre_g": 8.0},
        "roti":       {"calories": 297, "protein_g": 9.9, "carbs_g": 61.0, "fat_g": 1.7, "fibre_g": 1.9},
        "biryani":    {"calories": 200, "protein_g": 8.0, "carbs_g": 30.0, "fat_g": 6.0, "fibre_g": 1.5},
        "paneer":     {"calories": 265, "protein_g": 18.0,"carbs_g": 3.4,  "fat_g": 20.0,"fibre_g": 0.0},
        "idli":       {"calories": 58,  "protein_g": 2.0, "carbs_g": 12.0, "fat_g": 0.4, "fibre_g": 0.5},
        "dosa":       {"calories": 168, "protein_g": 3.7, "carbs_g": 30.0, "fat_g": 3.7, "fibre_g": 1.5},
        "samosa":     {"calories": 262, "protein_g": 3.5, "carbs_g": 28.0, "fat_g": 15.0,"fibre_g": 2.0},
        "noodle":     {"calories": 138, "protein_g": 4.5, "carbs_g": 25.0, "fat_g": 2.0, "fibre_g": 1.8},
        "omelette":   {"calories": 154, "protein_g": 11.0,"carbs_g": 0.4,  "fat_g": 12.0,"fibre_g": 0.0},
        "yogurt":     {"calories": 59,  "protein_g": 10.0,"carbs_g": 3.6,  "fat_g": 0.4, "fibre_g": 0.0},
        "coffee":     {"calories": 2,   "protein_g": 0.3, "carbs_g": 0.0,  "fat_g": 0.0, "fibre_g": 0.0},
    }

    text_lower = ingredients_text.lower()

    # Count how many times each food appears — pick the one mentioned FIRST
    # (first mention = most prominent ingredient in the model's output)
    matched = {}
    for food in NUTRITION_TABLE:
        idx = text_lower.find(food)
        if idx != -1:
            matched[food] = idx   # store position of first mention

    if matched:
        # Use the food mentioned earliest in the text (most prominent)
        primary_food = min(matched, key=matched.get)

        # If multiple foods found, sum up unique ones for a combined estimate
        unique_foods  = list(set(matched.keys()))[:5]   # max 5 ingredients
        if len(unique_foods) > 1:
            total = {"calories": 0, "protein_g": 0, "carbs_g": 0, "fat_g": 0, "fibre_g": 0}
            for food in unique_foods:
                for key in total:
                    total[key] += NUTRITION_TABLE[food][key]
            # Average across ingredients (rough estimate per serving)
            count = len(unique_foods)
            logger.info(f"Combined nutrition for: {unique_foods}")
            return {k: round(v / count, 1) for k, v in total.items()}
        else:
            logger.info(f"Single food nutrition for: {primary_food}")
            return NUTRITION_TABLE[primary_food]

    # Try Open Food Facts as last resort
    try:
        import re as _re
        words   = _re.findall(r'\b[a-zA-Z]{4,}\b', ingredients_text)
        query   = words[0] if words else "food"
        r       = req_lib.get(
            "https://world.openfoodfacts.org/cgi/search.pl",
            params={"search_terms": query, "search_simple": 1,
                    "action": "process", "json": 1, "page_size": 3,
                    "fields": "nutriments"},
            timeout=8,
        )
        for product in r.json().get("products", []):
            n   = product.get("nutriments", {})
            cal = float(n.get("energy-kcal_100g") or 0)
            if cal > 0:
                return {
                    "calories":  round(cal, 1),
                    "protein_g": round(float(n.get("proteins_100g",      0) or 0), 1),
                    "carbs_g":   round(float(n.get("carbohydrates_100g", 0) or 0), 1),
                    "fat_g":     round(float(n.get("fat_100g",           0) or 0), 1),
                    "fibre_g":   round(float(n.get("fiber_100g",         0) or 0), 1),
                }
    except Exception as e:
        logger.warning(f"OpenFoodFacts failed: {e}")

    return {}

def _parse_response(raw: str) -> dict:
    import re

    result = {
        "ingredients":   "",
        "portion_notes": "Portion estimated from image.",
        "raw_text":      raw,
        "calories":      None,
        "protein_g":     None,
        "carbs_g":       None,
        "fat_g":         None,
        "fibre_g":       None,
    }

    # ── Strategy 1: structured CaLoRAify format ───────────────────────────
    if "Ingredients detected:" in raw:
        ing_start = raw.index("Ingredients detected:") + len("Ingredients detected:")
        # End at next section or newline
        for end_marker in ["Portion Analysis:", "JSON Summary:", "\n\n"]:
            if end_marker in raw[ing_start:]:
                ing_end = raw.index(end_marker, ing_start)
                break
        else:
            ing_end = min(ing_start + 200, len(raw))
        result["ingredients"] = raw[ing_start:ing_end].strip().rstrip(".")

    if "Portion Analysis:" in raw:
        pa_start = raw.index("Portion Analysis:") + len("Portion Analysis:")
        pa_end   = raw.find("JSON Summary:", pa_start)
        pa_end   = pa_end if pa_end != -1 else pa_start + 150
        result["portion_notes"] = raw[pa_start:pa_end].strip()

    if "JSON Summary:" in raw:
        json_start  = raw.index("JSON Summary:") + len("JSON Summary:")
        brace_start = raw.find("{", json_start)
        brace_end   = raw.rfind("}") + 1
        if brace_start != -1 and brace_end > brace_start:
            try:
                nutrition = json.loads(raw[brace_start:brace_end])
                result.update({
                    "calories":  nutrition.get("calories_kcal") or nutrition.get("calories"),
                    "protein_g": nutrition.get("protein_g"),
                    "carbs_g":   nutrition.get("carbs_g") or nutrition.get("carbohydrate_g"),
                    "fat_g":     nutrition.get("fat_g"),
                    "fibre_g":   nutrition.get("fibre_g"),
                })
            except json.JSONDecodeError:
                pass

    # ── Strategy 2: scan raw text for known food words directly ───────────
    # This works regardless of what format the model uses
    if not result["ingredients"] or result["calories"] is None:
        FOOD_WORDS = [
            "banana", "apple", "orange", "mango", "grape", "strawberry",
            "watermelon", "pineapple", "coconut", "lettuce", "salad",
            "carrot", "onion", "rice", "chicken", "egg", "bread", "milk",
            "cheese", "pizza", "burger", "pasta", "fish", "potato",
            "broccoli", "tomato", "sandwich", "soup", "chocolate", "cake",
            "dal", "roti", "biryani", "paneer", "idli", "dosa", "samosa",
            "noodle", "omelette", "yogurt", "coffee", "curry", "taco",
            "sushi", "steak", "bacon", "butter", "cream", "icecream",
        ]
        text_lower = raw.lower()
        found_foods = []
        for food in FOOD_WORDS:
            if food in text_lower:
                found_foods.append(food)

        if found_foods:
            # Use found foods as ingredients if not already set
            if not result["ingredients"]:
                result["ingredients"] = ", ".join(found_foods)
            # Get nutrition from found foods
            if result["calories"] is None:
                nutrition = _get_nutrition_from_api(", ".join(found_foods))
                if nutrition:
                    result.update(nutrition)

    # ── Strategy 3: fallback — use entire raw text for API lookup ─────────
    if not result["ingredients"]:
        result["ingredients"] = raw.strip()[:150]

    if result["calories"] is None and result["ingredients"]:
        nutrition = _get_nutrition_from_api(result["ingredients"])
        if nutrition:
            result.update(nutrition)

    return result

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "cuda": torch.cuda.is_available(),
    }


@app.post("/analyze", response_model=NutritionResponse)
def analyze(req: AnalyzeRequest, _=Depends(verify_api_key)):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    image = _decode_image(req.image_b64)

    t0  = time.monotonic()
    raw = _run_inference(image, req.max_new_tokens)
    ms  = int((time.monotonic() - t0) * 1000)

    parsed = _parse_response(raw)
    return NutritionResponse(**parsed, latency_ms=ms)

@app.post("/debug")
def debug(req: AnalyzeRequest, _=Depends(verify_api_key)):
    image = _decode_image(req.image_b64)
    raw   = _run_inference(image, req.max_new_tokens)
    return {"raw_text": raw}
