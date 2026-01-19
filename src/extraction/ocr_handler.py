import pytesseract
from pdf2image import convert_from_path
from pathlib import Path

class OCRHandler:
    """Gère l'OCR pour les PDF scannés"""
    
    def __init__(self, pdf_path):
        self.pdf_path = Path(pdf_path)
    
    def extract_with_ocr(self, lang='eng'):
        """
        Extrait le texte d'un PDF scanné
        lang='eng' pour anglais, 'fra' pour français
        """
        # Convertit chaque page PDF en image
        images = convert_from_path(self.pdf_path)
        
        text = ""
        for i, image in enumerate(images):
            text += f"\n--- Page {i + 1} ---\n"
            # OCR sur l'image
            page_text = pytesseract.image_to_string(image, lang=lang)
            text += page_text
        
        return text