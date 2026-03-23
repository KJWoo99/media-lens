# 트러블슈팅 기록

---

## 1. CLIP 한국어 번역 실패 — `sentencepiece` 누락

**증상**
```
[Translation failed] MarianTokenizer requires the SentencePiece library but it was not found
[CLIP 번역] '강아지' → '강아지'
```

**원인**
`requirements.txt`에 `sentencepiece`가 빠져 있었고, MarianMT 토크나이저가 내부적으로 요구하는 패키지임.

**해결**
```bash
pip install sentencepiece sacremoses
```
`requirements.txt`에 두 패키지 추가.

---

## 2. MarianMT 모델 미다운로드

**증상**
```
[Translation failed] argument should be a str or an os.PathLike object
where __fspath__ returns a str, not 'NoneType'
```

**원인**
`Helsinki-NLP/opus-mt-ko-en` 모델이 `models/` 폴더에 없는 상태.
`local_files_only=True` 실패 후 온라인 다운로드 시도했으나 sentencepiece 미설치로 토크나이저 초기화 중 `vocab_file=None` 발생.

**해결**
sentencepiece 설치 후 수동으로 모델 다운로드:
```python
from transformers import MarianMTModel, MarianTokenizer
MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ko-en", cache_dir="models/")
MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-ko-en", cache_dir="models/")
```

---

## 3. meta tensor 오류 — `.to(device)` 실패

**증상**
```
[Translation failed] Cannot copy out of meta tensor; no data!
Please use torch.nn.Module.to_empty() instead of torch.nn.Module.to()
```

**원인**
transformers 5.x에서 `from_pretrained()` 시 weights를 meta tensor(빈 껍데기)로 lazy load함.
이후 `.to(device)` 호출 시 "데이터 없는 tensor를 복사할 수 없다"는 에러 발생.

**시도한 방법들 (실패)**
- `low_cpu_mem_usage=False` → 5.x에서 효과 없음
- `device_map=self.device` → `accelerate` 패키지 필요 에러

**해결**
번역 모델을 CPU에 고정(`self.device = "cpu"`), `.to()` 호출 제거 → `.from_pretrained()` 기본값(CPU)으로 로드.

---

## 4. 번역 결과 쓰레기값 — 앱 환경 오염

**증상**
```
[CLIP 번역] '강아지' → 'the  happy 보십시다下疳下疳 one one extinct'
[CLIP 번역] '고양이' → '엄습'
```
단독 실행 시에는 정상 작동.

**원인**
transformers 5.x lazy loading + SigLIP2(GemmaTokenizer)/CLIP이 이미 로드된 상태에서 MarianMT weights가 materialized될 때 글로벌 상태 오염 발생.
`model.generate()`가 올바른 가중치 대신 이상한 값으로 연산.

디버그로 확인한 사항:
- `device=cpu`, `is_meta=False`, `dtype=float32` → 정상
- input_ids 정상 (`[[26476, 0]]` for '강아지')
- output_ids 비정상 (`[[65000, 13, 9, ...]]`)
- 단독 실행 시 `Puppy`, `Cat.`, `Man` 정상 출력

**해결**
`translation.py`를 **서브프로세스로 격리**:
- 앱과 별개의 Python 프로세스에서 MarianMT 로드 및 번역 실행
- stdin/stdout(bytes)으로 텍스트 송수신
- 결과 인메모리 캐싱으로 반복 쿼리 즉시 응답

---

## 5. 서브프로세스 파이프 오류 — `[Errno 22] Invalid argument`

**증상**
```
[Translation failed] [Errno 22] Invalid argument
```

**원인**
`translation.py`를 서브프로세스로 실행 시 top-level에서 `from core.model_paths import MODEL_DIR` 실행.
서브프로세스는 `sys.path`에 프로젝트 루트가 없어 `ModuleNotFoundError` 발생 → 프로세스 즉시 종료 → stdin 파이프 write 실패.

**해결**
`__main__` 가드에서 `sys.path` 먼저 설정:
```python
if __name__ != "__main__":
    from core.model_paths import MODEL_DIR
else:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.model_paths import MODEL_DIR
```

---

## 6. 첫 번역 10초 지연

**증상**
첫 한국어 검색 시 약 10초 대기 (이후 동일 단어는 0.25초).

**원인**
서브프로세스 방식으로 격리 후 처음 번역 시 Python 프로세스 + MarianMT 모델 로드 시간 발생.

**해결**
`CLIPEngine.__init__()`에서 백그라운드 스레드로 번역 서브프로세스를 미리 warm-up:
```python
import threading
threading.Thread(target=self.translator._ensure_proc, daemon=True).start()
```
앱 시작과 동시에 프로세스가 올라가므로 첫 검색 시 대기 없음.
