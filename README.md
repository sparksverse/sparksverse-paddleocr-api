---
title: Sparksverse PaddleOCR API
emoji: 📝
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: apache-2.0
---

# Sparksverse PaddleOCR API

High-accuracy OCR service powered by **PaddleOCR 3.7.0 (PP-OCRv5)**

## 🌟 Features

- **PP-OCRv5 Engine**: 13% accuracy improvement over v4
- **Multi-language Support**: Chinese (Simplified/Traditional), English, Japanese, Korean
- **High Performance**: Optimized for exam papers, handwriting, and complex documents
- **Table Recognition**: Extract structured data from tables
- **Free & Open Source**: No API costs, self-hosted

## 🚀 Quick Start

### API Endpoints

#### 1. Text Recognition - `/ocr`

**Request:**
```bash
curl -X POST "https://sparksverse-paddleocr-api.hf.space/ocr" \
  -F "file=@your_image.jpg" \
  -F "lang=ch"
```

**Response:**
```json
[
  {
    "boxes": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
    "txt": "recognized text",
    "score": 0.95
  }
]
```

**Parameters:**
- `file`: Image file (required)
- `lang`: Language - `ch`, `en`, `japan`, `korean`, `chinese_cht` (default: `ch`)

#### 2. Table Recognition - `/ocr_table`

**Request:**
```bash
curl -X POST "https://sparksverse-paddleocr-api.hf.space/ocr_table" \
  -F "file=@table_image.jpg" \
  -F "lang=ch"
```

**Response:**
```json
{
  "htmls": ["<table>...</table>"],
  "bboxes": [[x1,y1,x2,y2]],
  "types": ["table"]
}
```

## 📊 Language Support

| Code | Language | Description |
|------|----------|-------------|
| `ch` | Chinese | Simplified + Traditional (auto-detect) |
| `en` | English | English text |
| `japan` | Japanese | Japanese text |
| `korean` | Korean | Korean text |
| `chinese_cht` | Traditional Chinese | Traditional Chinese only |

## 🎯 Use Cases

Perfect for:
- ✅ Exam paper digitization
- ✅ Handwriting recognition
- ✅ Document scanning
- ✅ Table extraction
- ✅ Multi-language text recognition

## 🔧 Technology Stack

- **PaddleOCR**: 3.7.0 (PP-OCRv5) on PaddlePaddle 3.3.1
- **Table recognition**: PP-StructureV3
- **FastAPI**: Modern Python web framework
- **Docker**: Containerized deployment

## 📝 License

Apache 2.0

## 🦊 About Sparksverse

Built with ❤️ by [Sparksverse](https://sparksverse.com) - Empowering education through AI
