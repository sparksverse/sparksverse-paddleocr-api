"""
Sparksverse PaddleOCR API
Powered by PaddleOCR 3.x (PP-OCRv5)

Endpoints:
- POST /ocr        - Text recognition
- POST /ocr_table  - Table recognition (PP-StructureV3)

Migration note (2.x -> 3.x):
- `use_angle_cls`  -> `use_textline_orientation`
- `use_gpu=...`    -> `device="cpu"|"gpu"`
- `show_log`       -> removed (logging redesigned)
- `.ocr(img, cls=True)` -> `.predict(img)` (result shape changed)
- `PPStructure`    -> `PPStructureV3`
The public JSON response shape is kept identical to v1 so existing callers
(PDF pipeline, HF API consumers) keep working without changes.
"""

import io
import logging
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from paddleocr import PaddleOCR, PPStructureV3
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PADDLEOCR_VERSION = "3.7.0"
SERVICE_VERSION = "2.0.0"

# Initialize FastAPI app
app = FastAPI(
    title="Sparksverse PaddleOCR API",
    description="High-accuracy OCR service powered by PaddleOCR 3.x (PP-OCRv5)",
    version=SERVICE_VERSION,
    docs_url="/",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
# PaddleOCR 3.x removed `use_gpu`; use `device` instead ("cpu" or "gpu").
DEVICE = "cpu"


# Language options
class LangEnum(str, Enum):
    ch = "ch"  # Chinese (Simplified + Traditional)
    en = "en"  # English
    japan = "japan"  # Japanese
    korean = "korean"  # Korean
    chinese_cht = "chinese_cht"  # Traditional Chinese


# Engine caches
ocr_cache: Dict[str, PaddleOCR] = {}
structure_engine: Optional[PPStructureV3] = None


def get_ocr(lang: str) -> PaddleOCR:
    """Get or create a PaddleOCR (3.x) instance with caching."""
    if lang not in ocr_cache:
        logger.info(f"Initializing PaddleOCR for language: {lang}, device: {DEVICE}")
        ocr_cache[lang] = PaddleOCR(
            lang=lang,
            device=DEVICE,
            use_textline_orientation=True,   # replaces 2.x use_angle_cls
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
        )
    return ocr_cache[lang]


def get_structure_engine() -> PPStructureV3:
    """Get or create the PP-StructureV3 table-recognition pipeline (cached).

    PP-StructureV3 does not take a `lang` argument; it uses the multilingual
    default models, which cover Chinese + English tables well.
    """
    global structure_engine
    if structure_engine is None:
        logger.info(f"Initializing PP-StructureV3, device: {DEVICE}")
        structure_engine = PPStructureV3(
            device=DEVICE,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_formula_recognition=False,
            use_chart_recognition=False,
        )
    return structure_engine


def _as_dict(res: Any) -> Dict[str, Any]:
    """Extract the inner result dict from a PaddleOCR 3.x Result object.

    3.x Result objects expose their data via the `.json` attribute as
    {"res": {...}}. Fall back gracefully if the object is already a dict.
    """
    data = getattr(res, "json", None)
    if isinstance(data, dict):
        return data.get("res", data)
    if isinstance(res, dict):
        return res.get("res", res)
    return {}


def _to_list(value: Any) -> Any:
    """Convert numpy arrays (and nested ones) to plain JSON-serializable lists."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_to_list(v) for v in value]
    return value


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Sparksverse PaddleOCR API",
        "version": SERVICE_VERSION,
        "paddleocr_version": PADDLEOCR_VERSION,
    }


@app.post("/ocr")
async def ocr_recognition(
    file: UploadFile = File(...),
    lang: LangEnum = LangEnum.ch,
) -> List[Dict[str, Any]]:
    """
    Text recognition endpoint.

    Args:
        file: Image file (PNG, JPG, etc.)
        lang: Language for OCR (ch, en, japan, korean, chinese_cht)

    Returns:
        List of recognized text items with boxes, text, and confidence scores.
        Response shape is identical to v1: [{"boxes", "txt", "score"}].
    """
    try:
        # Read and validate image
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        # Convert to RGB numpy array
        try:
            image = Image.open(io.BytesIO(contents)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

        img_array = np.array(image)

        # Run OCR (3.x predict API)
        logger.info(f"Processing OCR for image: {file.filename}, language: {lang}")
        ocr = get_ocr(lang=lang.value)
        result = ocr.predict(img_array)

        if not result:
            return []

        final_result: List[Dict[str, Any]] = []
        for res in result:
            data = _as_dict(res)
            texts = data.get("rec_texts", []) or []
            scores = data.get("rec_scores", []) or []
            # Prefer 4-point polygons (matches v1 box shape); fall back to det polys
            polys = data.get("rec_polys")
            if polys is None:
                polys = data.get("dt_polys", [])
            polys = list(polys) if polys is not None else []

            for i, txt in enumerate(texts):
                box = _to_list(polys[i]) if i < len(polys) else []
                score = float(scores[i]) if i < len(scores) else 0.0
                final_result.append({
                    "boxes": box,
                    "txt": txt,
                    "score": score,
                })

        logger.info(f"OCR completed: {len(final_result)} text items recognized")
        return final_result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"OCR error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")


@app.post("/ocr_table")
async def table_recognition(
    file: UploadFile = File(...),
    lang: LangEnum = LangEnum.ch,
) -> Dict[str, Any]:
    """
    Table recognition endpoint (PP-StructureV3).

    Args:
        file: Image file containing tables
        lang: Kept for API compatibility; PP-StructureV3 uses the multilingual
              default models and does not take a per-request language.

    Returns:
        Structured table data with HTML representation.
        Response shape is identical to v1: {"htmls", "bboxes", "types"}.
    """
    try:
        # Read and validate image
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        # Convert to RGB numpy array
        try:
            image = Image.open(io.BytesIO(contents)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

        img_array = np.array(image)

        # Run table recognition (PP-StructureV3)
        logger.info(f"Processing table for image: {file.filename}, language: {lang}")
        engine = get_structure_engine()
        result = engine.predict(img_array)

        htmls: List[str] = []
        bboxes: List[Any] = []
        types: List[str] = []

        for res in result:
            data = _as_dict(res)
            for table in data.get("table_res_list", []) or []:
                htmls.append(table.get("pred_html", ""))
                bboxes.append(_to_list(table.get("cell_box_list", [])))
                types.append("table")

        logger.info(f"Table recognition completed: {len(htmls)} tables found")

        return {
            "htmls": htmls,
            "bboxes": bboxes,
            "types": types,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Table recognition error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Table processing failed: {str(e)}")


if __name__ == '__main__':
    uvicorn.run(
        app=app,
        host="0.0.0.0",
        port=7860,  # HuggingFace Spaces default port
    )
