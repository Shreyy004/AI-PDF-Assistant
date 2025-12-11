import PyPDF2
from pdf2image import convert_from_bytes
import pytesseract
from PIL import Image
import io

def extract_text_from_pdfs(pdf_files):
    full_text = ""

    for file in pdf_files:
        try:
            # Try reading with PyPDF2
            reader = PyPDF2.PdfReader(file)
            text_found = False
            temp_text = ""

            for page in reader.pages:
                text = page.extract_text()
                if text and text.strip():
                    temp_text += text + "\n"
                    text_found = True

            if text_found:
                full_text += temp_text
            else:
                # OCR fallback
                file.seek(0)  # Ensure read pointer is reset
                images = convert_from_bytes(file.read())
                for img in images:
                    ocr_text = pytesseract.image_to_string(img)
                    full_text += ocr_text + "\n"

        except Exception as e:
            print(f"⚠️ Error reading {file.name}: {e}")
            continue

    return full_text.strip()
