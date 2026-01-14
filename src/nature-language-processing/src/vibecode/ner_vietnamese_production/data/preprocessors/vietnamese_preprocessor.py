import re
from typing import List, Tuple
from utils.constants import VIETNAMESE_STOP_WORDS
from utils.helpers import normalize_vietnamese_text

class VietnamesePreprocessor:

    def __init__(self, lowercase: bool = False):
        self.lowercase = lowercase


    def preprocess(self, text: str) -> str:
        text = normalize_vietnamese_text(text)

        if self.lowercase:
            text = text.lower()

        return text
    

    def tokenize_words(self, text: str) -> List[str]:
        """
        Simple word tokenization for Vietnamese
        """
        text = self.preprocess(text)
        # Split by whitespace
        tokens = text.split()
        return tokens
    

    def remove_stop_words(self, tokens: List[str]) -> List[str]:
        return [
            token for token in tokens
            if token.lower() not in VIETNAMESE_STOP_WORDS
        ]
    
    def handle_special_characters(self, text: str) -> str:
        # Keep Vietnamese characters, numbers, and basic punctuation
        text = re.sub(r'[^\w\sàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđĐ.,!?;:-]', '', text)
        return text