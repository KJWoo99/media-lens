"""Korean to English translation using MarianMT (subprocess-isolated)."""

import os
import sys
import logging
import subprocess

if __name__ != "__main__":
    from core.model_paths import MODEL_DIR
else:
    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, _ROOT)
    from core.model_paths import MODEL_DIR

logger = logging.getLogger(__name__)


# ── subprocess entry point ────────────────────────────────────────────────────
def _subprocess_main():
    """Persistent worker: read lines from stdin, write translations to stdout."""
    import torch
    from transformers import MarianMTModel, MarianTokenizer
    model_name = "Helsinki-NLP/opus-mt-ko-en"
    tok   = MarianTokenizer.from_pretrained(model_name, cache_dir=MODEL_DIR)
    model = MarianMTModel.from_pretrained(model_name, cache_dir=MODEL_DIR)
    model.eval()

    sys.stdout.buffer.write(b"READY\n")
    sys.stdout.buffer.flush()

    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            break
        text = line.decode("utf-8").rstrip("\n")
        try:
            inputs = tok(text, return_tensors="pt", padding=True)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=64, num_beams=4,
                                     do_sample=False, forced_bos_token_id=None)
            result = tok.decode(out[0], skip_special_tokens=True)
        except Exception:
            result = text
        sys.stdout.buffer.write((result + "\n").encode("utf-8"))
        sys.stdout.buffer.flush()


# ── main process side ─────────────────────────────────────────────────────────
class KoreanTranslator:
    """Translates Korean text via a single persistent background subprocess."""

    def __init__(self):
        self._cache = {}
        self._proc  = None

    def _ensure_proc(self):
        if self._proc is not None and self._proc.poll() is None:
            return
        self._proc = subprocess.Popen(
            [sys.executable, os.path.abspath(__file__)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        ready = self._proc.stdout.readline()  # wait for b"READY\n"
        if not ready.strip():
            raise RuntimeError("Translation subprocess failed to start")

    @staticmethod
    def contains_korean(text):
        return any('\uAC00' <= c <= '\uD7A3' for c in text)

    def translate(self, text):
        if not self.contains_korean(text):
            return text
        if text in self._cache:
            return self._cache[text]
        try:
            self._ensure_proc()
            self._proc.stdin.write((text + "\n").encode("utf-8"))
            self._proc.stdin.flush()
            result = self._proc.stdout.readline().decode("utf-8").rstrip("\n")
            self._cache[text] = result
            return result
        except Exception as e:
            logger.warning(f"[Translation failed] {e}")
            self._proc = None  # force restart next time
            return text

    def shutdown(self):
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.stdin.close()
                self._proc.wait(timeout=5)
            except Exception:
                self._proc.kill()


if __name__ == "__main__":
    _subprocess_main()
