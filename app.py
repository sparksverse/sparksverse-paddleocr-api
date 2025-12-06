"""
Sparksverse PaddleOCR API
Based on PaddleOCR 3.0.3 (PP-OCRv5) - June 2025
High accuracy Chinese OCR with no cost!

Endpoints:
- POST /ocr - Text recognition
- POST /ocr_table - Table recognition
"""

import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from paddleocr import PaddleOCR, PPStructure
from PIL import Image
import io
import numpy as np
from enum import Enum
from typing import List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Sparksverse PaddleOCR API",
    description="High-accuracy OCR service powered by PaddleOCR 3.0.3 (PP-OCRv5)",
    version="1.0.0",
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
USE_GPU = False  # Set to True if GPU is available
OUTPUT_DIR = 'output'

# Language options
class LangEnum(str, Enum):
    ch = "ch"  # Chinese (Simplified + Traditional)
    en = "en"  # English
    japan = "japan"  # Japanese
    korean = "korean"  # Korean
    chinese_cht = "chinese_cht"  # Traditional Chinese

# OCR instance cache
ocr_cache: Dict[str, PaddleOCR] = {}
table_engine_cache: Dict[str, PPStructure] = {}


def get_ocr(lang: str, use_gpu: bool = False) -> PaddleOCR:
    """Get or create PaddleOCR instance with caching"""
    cache_key = f"{lang}_{use_gpu}"
    if cache_key not in ocr_cache:
        logger.info(f"Initializing PaddleOCR for language: {lang}, GPU: {use_gpu}")
        ocr_cache[cache_key] = PaddleOCR(
            use_angle_cls=True,
            lang=lang,
            use_gpu=use_gpu,
            show_log=False,
        )
    return ocr_cache[cache_key]


def get_table_engine(lang: str, use_gpu: bool = False) -> PPStructure:
    """Get or create PPStructure instance with caching"""
    cache_key = f"{lang}_{use_gpu}"
    if cache_key not in table_engine_cache:
        logger.info(f"Initializing PPStructure for language: {lang}, GPU: {use_gpu}")
        table_engine_cache[cache_key] = PPStructure(
            show_log=False,
            table=True,
            lang=lang,
            use_gpu=use_gpu,
        )
    return table_engine_cache[cache_key]


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Sparksverse PaddleOCR API",
        "version": "1.0.0",
        "paddleocr_version": "3.0.3",
    }


@app.post("/ocr")
async def ocr_recognition(
    file: UploadFile = File(...),
    lang: LangEnum = LangEnum.ch,
) -> List[Dict[str, Any]]:
    """
    Text recognition endpoint

    Args:
        file: Image file (PNG, JPG, etc.)
        lang: Language for OCR (ch, en, japan, korean, chinese_cht)

    Returns:
        List of recognized text items with boxes, text, and confidence scores
    """
    try:
        # Read and validate image
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        # Convert to PIL Image
        try:
            image = Image.open(io.BytesIO(contents))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

        # Get OCR instance
        ocr = get_ocr(lang=lang.value, use_gpu=USE_GPU)

        # Convert to numpy array
        img_array = np.array(image)

        # Perform OCR
        logger.info(f"Processing OCR for image: {file.filename}, language: {lang}")
        result = ocr.ocr(img_array, cls=True)

        if not result or not result[0]:
            return []

        # Format results
        final_result = []
        for line in result[0]:
            box = line[0]  # Bounding box coordinates
            text_info = line[1]  # (text, confidence)

            final_result.append({
                "boxes": box,
                "txt": text_info[0],
                "score": float(text_info[1]),
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
    Table recognition endpoint

    Args:
        file: Image file containing tables
        lang: Language for OCR

    Returns:
        Structured table data with HTML representation
    """
    try:
        # Read and validate image
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        # Convert to PIL Image
        try:
            image = Image.open(io.BytesIO(contents))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

        # Get table engine
        table_engine = get_table_engine(lang=lang.value, use_gpu=USE_GPU)

        # Convert to numpy array
        img_array = np.array(image)

        # Perform table recognition
        logger.info(f"Processing table for image: {file.filename}, language: {lang}")
        result = table_engine(img_array)

        # Format results
        htmls = []
        types = []
        bboxes = []

        for item in result:
            item_res = item.get('res', {})
            htmls.append(item_res.get('html', ''))
            types.append(item.get('type', ''))
            bboxes.append(item.get('bbox', ''))

        logger.info(f"Table recognition completed: {len(result)} items found")

        return {
            'htmls': htmls,
            'bboxes': bboxes,
            'types': types,
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
