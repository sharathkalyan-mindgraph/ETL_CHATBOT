# scripts/etl.py
from pathlib import Path
from PIL import Image
import pdfplumber
import docx
import pytesseract
import cv2
import numpy as np
import os
import traceback
import tempfile
import re

# ✅ Set path for Tesseract (Windows only)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"



# PDF Extraction

def extract_text_from_pdf(path):
    """Extract text from a PDF safely; fallback to OCR if needed"""
    texts = []
    try:
        if not os.path.exists(path):
            print(f"[ERROR] PDF file not found: {path}")
            return ""

        with pdfplumber.open(path) as pdf:
            for page_no, page in enumerate(pdf.pages, start=1):
                try:
                    txt = page.extract_text()
                    if txt and txt.strip():
                        texts.append(txt)
                    else:
                        # OCR fallback for image-based pages
                        img = page.to_image(resolution=300).original
                        text_ocr = pytesseract.image_to_string(img, lang="eng")
                        if text_ocr.strip():
                            texts.append(text_ocr)
                except Exception as e:
                    print(f"[WARN] Failed to extract text from page {page_no} in {path}: {e}")

        combined_text = "\n".join(texts)
        print(f"[INFO] Extracted {len(combined_text)} characters from PDF {os.path.basename(path)}")
        return combined_text

    except Exception as e:
        print(f"[ERROR] PDF extraction failed for {path}: {e}")
        traceback.print_exc()
        return ""



# DOCX Extraction

def extract_text_from_docx(path):
    """Extract text from DOCX safely"""
    try:
        if not os.path.exists(path):
            print(f"[ERROR] DOCX file not found: {path}")
            return ""

        doc = docx.Document(path)
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        combined = "\n".join(paragraphs)
        print(f"[INFO] Extracted {len(combined)} characters from DOCX {os.path.basename(path)}")
        return combined

    except Exception as e:
        print(f"[ERROR] Failed to extract text from DOCX {path}: {e}")
        traceback.print_exc()
        return ""



# Image Preprocessing for OCR

def preprocess_image_for_ocr(path):
    """Preprocess image to enhance OCR accuracy"""
    try:
        image = cv2.imread(path)
        if image is None:
            print(f"[ERROR] Unable to read image file: {path}")
            return None

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        gray = cv2.equalizeHist(gray)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        temp_path = os.path.join(tempfile.gettempdir(), f"temp_ocr_{os.getpid()}.png")
        cv2.imwrite(temp_path, gray)
        return temp_path

    except Exception as e:
        print(f"[ERROR] Failed to preprocess image {path}: {e}")
        traceback.print_exc()
        return None



# OCR Extraction

def extract_text_from_image(path):
    """Extract text from image using Tesseract OCR with fallback strategy"""
    try:
        processed_path = preprocess_image_for_ocr(path)
        text = ""

        # Try OCR on processed image first
        if processed_path and os.path.exists(processed_path):
            text = pytesseract.image_to_string(Image.open(processed_path), lang="eng")
            os.remove(processed_path)

        # Fallback: if no text detected, try original image
        if not text.strip():
            print(f"[INFO] Retrying OCR on original image: {os.path.basename(path)}")
            text = pytesseract.image_to_string(Image.open(path), lang="eng")

        # Final fallback: different PSM modes
        if not text.strip():
            print(f"[INFO] Trying alternate OCR mode (PSM 6) for {os.path.basename(path)}")
            text = pytesseract.image_to_string(Image.open(path), lang="eng", config="--psm 6")

        if text.strip():
            print(f"[INFO] OCR extracted {len(text)} characters from {os.path.basename(path)}")
        else:
            print(f"[WARN] No text detected by OCR in {os.path.basename(path)}")

        return text.strip()

    except Exception as e:
        print(f"[ERROR] OCR failed for {path}: {e}")
        traceback.print_exc()
        return ""




# Auto File Type Extraction

def extract_from_file(path):
    """Extract text from supported document formats"""
    try:
        ext = Path(path).suffix.lower()

        if ext == ".pdf":
            return extract_text_from_pdf(path)
        elif ext in [".docx", ".doc"]:
            return extract_text_from_docx(path)
        elif ext in [".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"]:
            return extract_text_from_image(path)
        elif os.path.isfile(path):
            try:
                return open(path, "r", encoding="utf-8").read()
            except UnicodeDecodeError:
                print(f"[WARN] {path} is not a UTF-8 text file. Skipping.")
                return ""
        else:
            print(f"[WARN] Unsupported or missing file type: {path}")
            return ""
    except Exception as e:
        print(f"[ERROR] Unexpected error while extracting from {path}: {e}")
        traceback.print_exc()
        return ""



# Chunking Function

def doc_to_chunks(doc_local_path, doc_id, chunk_size=800, overlap=100):
    """Split extracted text into overlapping chunks cleanly (word-aware)"""
    try:
        text = extract_from_file(doc_local_path)
        if not text.strip():
            print(f"[WARN] No text extracted from {doc_local_path}")
            return []

        # Clean up whitespace
        text = re.sub(r"\s+", " ", text).strip()

        chunks = []
        words = text.split()
        current_chunk = []
        current_length = 0
        cid = 0

        for word in words:
            if current_length + len(word) + 1 <= chunk_size:
                current_chunk.append(word)
                current_length += len(word) + 1
            else:
                chunk_text = " ".join(current_chunk)
                chunks.append({
                    "id": f"{doc_id}_chunk_{cid}",
                    "text": chunk_text,
                    "metadata": {
                        "source": doc_local_path,
                        "doc_id": doc_id,
                        "chunk_index": cid
                    }
                })
                cid += 1
                # start overlap portion
                current_chunk = current_chunk[-overlap//10:] + [word]
                current_length = sum(len(w) + 1 for w in current_chunk)

        if current_chunk:
            chunks.append({
                "id": f"{doc_id}_chunk_{cid}",
                "text": " ".join(current_chunk),
                "metadata": {
                    "source": doc_local_path,
                    "doc_id": doc_id,
                    "chunk_index": cid
                }
            })

        print(f"[INFO] Split {os.path.basename(doc_local_path)} into {len(chunks)} chunks.")
        return chunks

    except Exception as e:
        print(f"[ERROR] Failed to split document {doc_local_path} into chunks: {e}")
        traceback.print_exc()
        return []
