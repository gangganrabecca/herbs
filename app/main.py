from typing import List, Dict
import io
import math
import base64
import json

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor
from huggingface_hub import snapshot_download
from pathlib import Path
import os
import uvicorn

torch.set_num_threads(1)

app = FastAPI(title="Herbal Plant Identifier", version="0.1.0")

# CORS (allow local dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Frontend directory (app/frontend next to this file)
FRONTEND_DIR = Path(__file__).parent / "frontend"
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
async def index():
    index_path = FRONTEND_DIR / "index.html"
    return FileResponse(str(index_path))


@app.get("/favicon.ico")
async def favicon():
    # Return a tiny transparent PNG to avoid 404s
    buf = io.BytesIO()
    Image.new("RGBA", (16, 16), (0, 0, 0, 0)).save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


# Herbal plant catalog (no DB). Keep concise but useful.
HERBS: List[Dict] = [
    {
        "name": "Aloe vera",
        "scientific_name": "Aloe vera",
        "aliases": ["Aloe"],
        "details": "Succulent used for soothing burns, skin hydration, and minor wound care.",
        "benefits": "Soothes burns, moisturizes skin, may aid wound healing.",
        "cautions": "Possible latex sensitivity; avoid ingesting latex; may interact with diabetes meds.",
    },
    {
        "name": "Basil (Holy basil)",
        "scientific_name": "Ocimum tenuiflorum",
        "aliases": ["Tulsi", "Holy basil"],
        "details": "Holy basil is used in teas for stress relief and respiratory support.",
        "benefits": "Adaptogenic; supports stress response and respiratory comfort.",
        "cautions": "May affect blood clotting and blood sugar; consult if pregnant or on anticoagulants.",
    },
    {
        "name": "Chamomile",
        "scientific_name": "Matricaria chamomilla",
        "aliases": ["German chamomile", "Roman chamomile"],
        "details": "Calming herb used for sleep support and mild digestive comfort.",
        "benefits": "Promotes relaxation, sleep, and eases mild stomach upset.",
        "cautions": "Allergy possible if sensitive to ragweed family; may enhance sedatives.",
    },
    {
        "name": "Ginger",
        "scientific_name": "Zingiber officinale",
        "aliases": ["Adrak"],
        "details": "Rhizome used for nausea relief, digestion, and warming teas.",
        "benefits": "Helps nausea, motion sickness, and digestion; anti-inflammatory.",
        "cautions": "Large doses may cause heartburn; caution with anticoagulants.",
    },
    {
        "name": "Turmeric",
        "scientific_name": "Curcuma longa",
        "aliases": ["Haldi"],
        "details": "Golden rhizome valued for curcumin; used for inflammation support.",
        "benefits": "Anti-inflammatory and antioxidant support.",
        "cautions": "May interact with blood thinners and gallbladder issues.",
    },
    {
        "name": "Peppermint",
        "scientific_name": "Mentha × piperita",
        "aliases": ["Mint"],
        "details": "Cooling mint used for digestion, headaches, and soothing teas.",
        "benefits": "Eases indigestion and tension headaches; refreshing aroma.",
        "cautions": "May worsen reflux; avoid applying near infants' noses.",
    },
    {
        "name": "Lemongrass",
        "scientific_name": "Cymbopogon citratus",
        "aliases": ["Tanglad"],
        "details": "Citrusy grass used in teas and cuisine; refreshing.",
        "benefits": "Digestive comfort; uplifting aroma.",
        "cautions": "Essential oil may irritate skin; dilute before topical use.",
    },
    {
        "name": "Lavender",
        "scientific_name": "Lavandula angustifolia",
        "aliases": ["True lavender"],
        "details": "Fragrant spikes used for relaxation, sleep, and calming aromatics.",
        "benefits": "Promotes relaxation and sleep; mild stress relief.",
        "cautions": "Essential oil may irritate sensitive skin; possible hormonal effects in children.",
    },
    {
        "name": "Rosemary",
        "scientific_name": "Salvia rosmarinus",
        "aliases": ["Rosmarinus officinalis"],
        "details": "Woody herb used for memory, focus, and savory dishes.",
        "benefits": "Cognitive support and antioxidant properties.",
        "cautions": "High doses may raise blood pressure; avoid in epilepsy.",
    },
    {
        "name": "Oregano",
        "scientific_name": "Origanum vulgare",
        "aliases": ["Wild marjoram"],
        "details": "Pungent herb with antimicrobial reputation; Mediterranean cuisine staple.",
        "benefits": "Antimicrobial and antioxidant support.",
        "cautions": "Essential oil is potent; dilute. May irritate mucosa.",
    },
    {
        "name": "Thyme",
        "scientific_name": "Thymus vulgaris",
        "aliases": ["Garden thyme"],
        "details": "Culinary herb also used for respiratory comfort.",
        "benefits": "Supports respiratory health; antimicrobial.",
        "cautions": "Essential oil is strong; avoid undiluted use.",
    },
    {
        "name": "Sage",
        "scientific_name": "Salvia officinalis",
        "aliases": ["Common sage"],
        "details": "Aromatic leaves used for sore throat gargles and cooking.",
        "benefits": "Soothes sore throat; cognitive support.",
        "cautions": "Thujone in essential oil can be neurotoxic at high doses.",
    },
    {
        "name": "Echinacea",
        "scientific_name": "Echinacea purpurea",
        "aliases": ["Purple coneflower"],
        "details": "Popular immune-support herb with daisy-like flowers.",
        "benefits": "Immune modulation; may reduce duration of colds.",
        "cautions": "Allergy possible in Asteraceae-sensitive individuals.",
    },
    {
        "name": "Ginseng",
        "scientific_name": "Panax ginseng",
        "aliases": ["Asian ginseng"],
        "details": "Adaptogen used for energy and resilience.",
        "benefits": "Supports vitality, cognition, and stress response.",
        "cautions": "May raise blood pressure; interacts with anticoagulants and stimulants.",
    },
    {
        "name": "Ginkgo",
        "scientific_name": "Ginkgo biloba",
        "aliases": ["Maidenhair tree"],
        "details": "Leaves used for circulation and cognitive support.",
        "benefits": "Supports memory and peripheral circulation.",
        "cautions": "May increase bleeding risk with anticoagulants.",
    },
    {
        "name": "Dandelion",
        "scientific_name": "Taraxacum officinale",
        "aliases": ["Lion's tooth"],
        "details": "Leaf and root used traditionally for digestion and liver support.",
        "benefits": "Mild diuretic (leaf); digestive and liver support (root).",
        "cautions": "Allergy possible; caution with diuretics.",
    },
    {
        "name": "Lemon balm",
        "scientific_name": "Melissa officinalis",
        "aliases": ["Melissa"],
        "details": "Citrusy mint family herb for calm and sleep support.",
        "benefits": "Calming; may aid sleep and digestion.",
        "cautions": "May lower thyroid activity; use caution with hypothyroidism.",
    },
    {
        "name": "Calendula",
        "scientific_name": "Calendula officinalis",
        "aliases": ["Marigold"],
        "details": "Bright flowers used topically for skin comfort.",
        "benefits": "Soothes skin; supports wound healing.",
        "cautions": "Allergy possible in Asteraceae-sensitive individuals.",
    },
    {
        "name": "Plantain",
        "scientific_name": "Plantago major",
        "aliases": ["Broadleaf plantain"],
        "details": "Common weed used for skin and minor wound support.",
        "benefits": "Soothes insect bites and minor skin irritations.",
        "cautions": "Generally safe; ensure proper identification.",
    },
    {
        "name": "Milk thistle",
        "scientific_name": "Silybum marianum",
        "aliases": ["Blessed thistle"],
        "details": "Purple-flowered plant used for liver support.",
        "benefits": "Supports liver function (silymarin).",
        "cautions": "May interact with medications metabolized by liver enzymes.",
    },
    {
        "name": "Ashwagandha",
        "scientific_name": "Withania somnifera",
        "aliases": ["Indian ginseng"],
        "details": "Adaptogen used for stress and sleep support.",
        "benefits": "Reduces stress, supports sleep and vitality.",
        "cautions": "Avoid in hyperthyroidism; caution with sedatives.",
    },
    {
        "name": "Moringa",
        "scientific_name": "Moringa oleifera",
        "aliases": ["Drumstick tree"],
        "details": "Nutritious leaves used as food and tonic.",
        "benefits": "Rich in vitamins; antioxidant support.",
        "cautions": "Seeds and root may be toxic; use leaves primarily.",
    },
    {
        "name": "Neem",
        "scientific_name": "Azadirachta indica",
        "aliases": ["Indian lilac"],
        "details": "Bitter leaves used in traditional skincare and oral care.",
        "benefits": "Antimicrobial; supports skin and oral health.",
        "cautions": "Avoid high oral doses; not for young children.",
    },
    {
        "name": "Garlic",
        "scientific_name": "Allium sativum",
        "aliases": ["Ajo"],
        "details": "Bulb used for cardiovascular and immune support.",
        "benefits": "Supports healthy cholesterol and immune function.",
        "cautions": "Increases bleeding risk with anticoagulants; odor.",
    },
]

# Optional external catalog override
DATA_DIR = Path(__file__).parent / "data"
CATALOG_PATH = DATA_DIR / "herbs.json"

def load_catalog() -> List[Dict]:
    try:
        if CATALOG_PATH.exists():
            with open(CATALOG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list) and data:
                return data
    except Exception:
        pass
    return HERBS

CATALOG: List[Dict] = load_catalog()

# Build zero-shot prompts from the catalog
CANDIDATE_TEXTS: List[str] = []
CANDIDATE_META: List[Dict] = []

# Extra descriptors for visually similar mints
HERB_DESCRIPTORS: Dict[str, List[str]] = {
    "Oregano": [
        "close-up of oregano leaves, small oval opposite leaves, slightly fuzzy texture, woody stems",
        "culinary oregano sprig with small rounded leaves",
    ],
    "Peppermint": [
        "peppermint sprig with spear-shaped serrated bright green leaves",
        "close-up of peppermint leaves, smooth stems, pronounced veins",
    ],
    "Lemon balm": [
        "lemon balm with heart-shaped crinkled leaves, softly serrated edges",
        "close-up of lemon balm leaves, textured surface, citrus-scented balm",
    ],
}

BASE_TEMPLATES: List[str] = [
    "a clear photo of the herbal plant {name}",
    "a close-up of {name} leaves",
    "a sprig of {name}, studio background",
]

for herb in CATALOG:
    all_names = [herb["name"], herb.get("scientific_name", ""), *herb.get("aliases", [])]
    descriptors = HERB_DESCRIPTORS.get(herb["name"], [])
    for n in all_names:
        if not n:
            continue
        for tpl in BASE_TEMPLATES:
            CANDIDATE_TEXTS.append(tpl.format(name=n))
            CANDIDATE_META.append(herb)
        for desc in descriptors:
            CANDIDATE_TEXTS.append(f"{desc}")
            CANDIDATE_META.append(herb)


# Load CLIP lazily to reduce startup memory
DEVICE = "cpu"
MODEL_NAME = os.getenv("MODEL_NAME", "openai/clip-vit-base-patch32").strip() or "openai/clip-vit-base-patch32"
clip_model = None
clip_processor = None

def _ensure_model_loaded():
    global clip_model, clip_processor
    if clip_model is None or clip_processor is None:
        # Prefer local model directory if provided
        local_dir = os.getenv("CLIP_LOCAL_DIR", "").strip()
        model_source = None
        if local_dir:
            try:
                m = CLIPModel.from_pretrained(local_dir, local_files_only=True)
                p = CLIPProcessor.from_pretrained(local_dir, local_files_only=True)
                model_source = local_dir
            except Exception:
                # Fallback to MODEL_NAME; may still try network if cache exists
                pass
        if model_source is None:
            # Try to resolve into a local snapshot directory to avoid repeated downloads
            try:
                snapshot_path = snapshot_download(
                    repo_id=MODEL_NAME,
                    local_dir=str(Path(__file__).parent / "models" / "clip"),
                    local_dir_use_symlinks=False,
                    resume_download=True,
                    allow_patterns=["*.json", "*.bin", "*.txt"],
                )
                m = CLIPModel.from_pretrained(snapshot_path, local_files_only=True)
                p = CLIPProcessor.from_pretrained(snapshot_path, local_files_only=True)
                model_source = snapshot_path
            except Exception as e:
                raise HTTPException(status_code=503, detail=f"Model files not available offline and download failed: {e}")
        m = m.to(DEVICE)
        m.eval()
        clip_model = m
        clip_processor = p

def _softmax(x: torch.Tensor) -> torch.Tensor:
    x_max = torch.max(x)
    exps = torch.exp(x - x_max)
    return exps / torch.sum(exps)


def _classify_image(image: Image.Image, top_k: int = 3):
    _ensure_model_loaded()
    with torch.no_grad():
        inputs = clip_processor(text=CANDIDATE_TEXTS, images=image, return_tensors="pt", padding=True).to(DEVICE)
        outputs = clip_model(**inputs)
        # Logits per image vs each text prompt
        logits = outputs.logits_per_image.squeeze(0).detach().cpu()
        probs = _softmax(logits)

        # Aggregate by herb name: take max probability across all prompts for that herb
        by_herb: Dict[str, Dict] = {}
        for idx in range(probs.shape[0]):
            meta = CANDIDATE_META[idx]
            herb_name = meta["name"]
            p = float(probs[idx].item())
            rec = by_herb.get(herb_name)
            if rec is None or p > rec["confidence"]:
                by_herb[herb_name] = {
                    "name": herb_name,
                    "scientific_name": meta.get("scientific_name", ""),
                    "confidence": p,
                    "details": meta.get("details", ""),
                    "benefits": meta.get("benefits", ""),
                    "cautions": meta.get("cautions", ""),
                    "matched_text": CANDIDATE_TEXTS[idx],
                }

        # Sort herbs by confidence and return top_k unique herbs
        ranked = sorted(by_herb.values(), key=lambda r: r["confidence"], reverse=True)
        return ranked[:top_k]


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.on_event("startup")
async def warmup():
    # Avoid heavy warmup on low-memory environments
    pass


def _read_image_from_request_sync(data: bytes):
    image = Image.open(io.BytesIO(data)).convert("RGB")
    return image


async def _read_image_from_request(request: Request) -> Image.Image:
    ct = request.headers.get("content-type", "")
    if "multipart/form-data" in ct:
        form = await request.form()
        # Prefer explicit 'file'
        file = form.get("file")
        if file is None:
            # Fallback: find first image-like file in form
            for _, v in form.multi_items():
                if hasattr(v, "read"):
                    try:
                        ctype = getattr(v, "content_type", "") or ""
                        if ctype.startswith("image/"):
                            file = v
                            break
                    except Exception:
                        pass
        if file is None:
            raise HTTPException(status_code=422, detail="No image file found in multipart form. Use field 'file'.")
        data = await file.read()
        if not data:
            raise HTTPException(status_code=422, detail="Empty uploaded file.")
        return _read_image_from_request_sync(data)
    if "application/json" in ct:
        try:
            payload = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON body.")
        # Accept common keys: image, file, data, image_base64
        b64 = None
        for key in ("image", "file", "data", "image_base64"):
            val = payload.get(key)
            if isinstance(val, str):
                b64 = val
                break
        if not b64:
            raise HTTPException(status_code=422, detail="JSON must include base64 image string under 'image' or 'file' or 'data'.")
        # Strip data URL prefix if present
        if b64.startswith("data:"):
            try:
                b64 = b64.split(",", 1)[1]
            except Exception:
                pass
        try:
            data = base64.b64decode(b64, validate=False)
        except Exception:
            raise HTTPException(status_code=400, detail="Failed to decode base64 image.")
        return _read_image_from_request_sync(data)
    # Otherwise, assume raw bytes (e.g., image/jpeg or application/octet-stream)
    data = await request.body()
    if not data:
        raise HTTPException(status_code=422, detail="No image provided in request body.")
    try:
        return _read_image_from_request_sync(data)
    except Exception:
        raise HTTPException(status_code=400, detail="Failed to read the image from raw body.")


@app.post("/api/predict")
async def predict(request: Request):
    image = await _read_image_from_request(request)

    # Allow overriding top_k via query param
    try:
        k = int(request.query_params.get("k", 5))
    except Exception:
        k = 5
    k = max(1, min(10, k))

    results = _classify_image(image, top_k=k)
    if not results:
        raise HTTPException(status_code=500, detail="No prediction generated.")

    return JSONResponse({
        "predictions": results,
        "items": results,
    })


@app.post("/detect")
async def detect(request: Request):
    image = await _read_image_from_request(request)

    try:
        k = int(request.query_params.get("k", 5))
    except Exception:
        k = 5
    k = max(1, min(10, k))

    results = _classify_image(image, top_k=k)
    if not results:
        raise HTTPException(status_code=500, detail="No prediction generated.")

    return JSONResponse({
        "predictions": results,
        "items": results,
    })


@app.post("/detect-json")
async def detect_json(request: Request):
    # Accept JSON with base64, but also fallback to other supported encodings
    image = await _read_image_from_request(request)

    try:
        k = int(request.query_params.get("k", 5))
    except Exception:
        k = 5
    k = max(1, min(10, k))

    results = _classify_image(image, top_k=k)

    if not results:
        raise HTTPException(status_code=500, detail="No prediction generated.")

    return JSONResponse({
        "predictions": results,
        "items": results,
    })

@app.get("/api/herbs")
async def list_herbs():
    return {"items": CATALOG}

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run("app.main:app", host=host, port=port, reload=os.getenv("RELOAD", "0") == "1")
