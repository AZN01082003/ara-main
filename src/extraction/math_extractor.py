from pix2tex.cli import LatexOCR
from pathlib import Path
import fitz  # PyMuPDF

class MathExtractor:
    """Extrait les formules mathématiques en LaTeX"""
    
    def __init__(self):
        self.model = LatexOCR()
    
    def extract_math_from_image(self, image_path):
        """Convertit une image de formule en LaTeX"""
        latex_code = self.model(image_path)
        return latex_code
    
    def extract_images_from_pdf(self, pdf_path, output_folder):
        """Extrait toutes les images d'un PDF"""
        doc = fitz.open(pdf_path)
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        
        image_list = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            images = page.get_images()
            
            for img_index, img in enumerate(images):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                
                # Sauvegarde l'image
                image_path = output_folder / f"page{page_num}_img{img_index}.png"
                with open(image_path, "wb") as img_file:
                    img_file.write(image_bytes)
                
                image_list.append(image_path)
        
        return image_list