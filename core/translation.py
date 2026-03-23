"""Korean to English translation using MarianMT."""

import logging
import torch
from core.model_paths import MODEL_DIR

logger = logging.getLogger(__name__)


class KoreanTranslator:
    """Translates Korean text to English using Helsinki-NLP/opus-mt-ko-en."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._cache = {}

    def _ensure_loaded(self):
        if self.model is None:
            from transformers import MarianMTModel, MarianTokenizer
            model_name = "Helsinki-NLP/opus-mt-ko-en"
            try:
                self.tokenizer = MarianTokenizer.from_pretrained(
                    model_name, cache_dir=MODEL_DIR, local_files_only=True)
                self.model = MarianMTModel.from_pretrained(
                    model_name, cache_dir=MODEL_DIR, local_files_only=True).to(self.device)
            except OSError:
                self.tokenizer = MarianTokenizer.from_pretrained(
                    model_name, cache_dir=MODEL_DIR)
                self.model = MarianMTModel.from_pretrained(
                    model_name, cache_dir=MODEL_DIR).to(self.device)

    @staticmethod
    def contains_korean(text):
        return any(ord('\uAC00') <= ord(c) <= ord('\uD7A3') for c in text)

    def translate(self, text):
        """Translate text. Returns original if not Korean or on failure."""
        if not self.contains_korean(text):
            return text
        if text in self._cache:
            return self._cache[text]
        try:
            self._ensure_loaded()
            inputs = self.tokenizer(text, return_tensors="pt", padding=True).to(self.device)
            translated = self.model.generate(**inputs, max_length=100)
            result = self.tokenizer.decode(translated[0], skip_special_tokens=True)
            self._cache[text] = result
            return result
        except Exception as e:
            logger.warning(f"[Translation failed] {e}")
            return text
