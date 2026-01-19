import fitz  # PyMuPDF
import pdfplumber
from pathlib import Path

class PDFExtractor:
    """Classe pour extraire le texte des PDF scientifiques"""
    
    def __init__(self, pdf_path):
        self.pdf_path = Path(pdf_path)
        
    def extract_with_pymupdf(self):
        """Méthode 1 : Extraction rapide avec PyMuPDF"""
        text = ""
        doc = fitz.open(self.pdf_path)
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text += f"\n--- Page {page_num + 1} ---\n"
            text += page.get_text()
        
        doc.close()
        return text
    
    def extract_with_pdfplumber(self):
        """Méthode 2 : Meilleure pour les tableaux"""
        text = ""
        
        with pdfplumber.open(self.pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text += f"\n--- Page {page_num + 1} ---\n"
                text += page.extract_text()
        
        return text
    
    def extract_tables(self):
        """Extrait uniquement les tableaux du PDF"""
        tables = []
        
        with pdfplumber.open(self.pdf_path) as pdf:
            for page in pdf.pages:
                page_tables = page.extract_tables()
                if page_tables:
                    tables.extend(page_tables)
        
        return tables
    
    def save_text(self, text, output_path):
        """Sauvegarde le texte extrait"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        
        print(f"✅ Texte sauvegardé dans : {output_path}")