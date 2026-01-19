"""
Nettoyage du texte extrait des PDF
"""

import re
from unidecode import unidecode


class TextCleaner:
    """Nettoie et normalise le texte extrait"""
    
    def __init__(self):
        pass
    
    def clean(self, text):
        """Pipeline de nettoyage complet"""
        
        # Étape 1 : Supprimer les caractères spéciaux bizarres
        text = self._remove_special_chars(text)
        
        # Étape 2 : Normaliser les espaces
        text = self._normalize_spaces(text)
        
        # Étape 3 : Fusionner les mots coupés en fin de ligne
        text = self._fix_hyphenation(text)
        
        # Étape 4 : Nettoyer les en-têtes/pieds de page
        text = self._remove_headers_footers(text)
        
        # Étape 5 : Normaliser la ponctuation
        text = self._normalize_punctuation(text)
        
        return text
    
    def _remove_special_chars(self, text):
        """Enlève les caractères de contrôle bizarres"""
        # Garde lettres, chiffres, ponctuation de base
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\'\"\n]', ' ', text)
        return text
    
    def _normalize_spaces(self, text):
        """Normalise les espaces multiples"""
        # Remplace espaces multiples par un seul
        text = re.sub(r' +', ' ', text)
        
        # Remplace sauts de ligne multiples par double saut
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        return text.strip()
    
    def _fix_hyphenation(self, text):
        """
        Fusionne les mots coupés en fin de ligne
        Exemple: "connec-\ntion" → "connection"
        """
        # Motif : mot- suivi d'un saut de ligne et d'une lettre
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
        return text
    
    def _remove_headers_footers(self, text):
        """
        Supprime les en-têtes/pieds de page répétitifs
        (numéros de page, titres répétés)
        """
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Ignorer les lignes qui sont juste des numéros
            if re.match(r'^\s*\d+\s*$', line):
                continue
            
            # Ignorer les lignes très courtes (probablement des en-têtes)
            if len(line.strip()) < 3:
                continue
            
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _normalize_punctuation(self, text):
        """Normalise la ponctuation"""
        # Ajouter espace après point/virgule si manquant
        text = re.sub(r'([\.,:;!?])([A-Za-z])', r'\1 \2', text)
        
        # Supprimer espaces avant ponctuation
        text = re.sub(r'\s+([\.,:;!?])', r'\1', text)
        
        return text
    
    def clean_for_search(self, text):
        """Nettoyage spécial pour la recherche (enlève accents, minuscules)"""
        # Minuscules
        text = text.lower()
        
        # Enlever les accents
        text = unidecode(text)
        
        return text