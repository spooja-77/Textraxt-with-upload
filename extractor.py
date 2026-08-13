import base64
import fitz  # PyMuPDF
from groq import Groq
from backend.config import MIN_TEXT_CHARS, OCR_PROMPT, VISION_MODEL


def ocr_image(client: Groq, image_bytes: bytes, mime: str) -> str:
    """Transcribe a document image (printed or handwritten) using Groq Vision model."""
    b64 = base64.b64encode(image_bytes).decode()
    resp = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": OCR_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"},
                    },
                ],
            }
        ],
        temperature=0,
    )
    return resp.choices[0].message.content or ""


def extract_pdf(client: Groq | None, data: bytes) -> tuple[str, str]:
    """Extract text layer from PDF, falling back to Vision OCR for scanned pages."""
    parts, ocr_pages = [], 0
    with fitz.open(stream=data, filetype="pdf") as doc:
        total = len(doc)
        for i, page in enumerate(doc, start=1):
            text = page.get_text().strip()
            if len(text) >= MIN_TEXT_CHARS:
                parts.append(f"[Page {i}]\n{text}")
            elif client is not None:
                png = page.get_pixmap(dpi=200).tobytes("png")
                parts.append(f"[Page {i} — OCR]\n{ocr_image(client, png, 'image/png')}")
                ocr_pages += 1
            else:
                parts.append(f"[Page {i}]\n(Scanned page — API key required for OCR)")
                
        kind = f"PDF · {total} page{'s' if total > 1 else ''}"
        if ocr_pages:
            kind += f" · {ocr_pages} OCR"
    return "\n\n".join(parts), kind


def extract_file_content(
    client: Groq | None, data: bytes, filename: str, mime_type: str = "image/png"
) -> tuple[str, str]:
    """Extract raw text and return (extracted_text, file_kind_label)."""
    name_lower = filename.lower()
    
    if name_lower.endswith(".pdf"):
        return extract_pdf(client, data)
        
    if name_lower.endswith((".png", ".jpg", ".jpeg", ".webp")):
        if client is None:
            return "Image uploaded (API key required for OCR)", "Image · OCR Pending"
        return ocr_image(client, data, mime_type or "image/png"), "Image · OCR"
        
    # Default text files
    return data.decode("utf-8", errors="replace"), "Text File"
