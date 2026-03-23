# 설치 가이드

## 필수 환경

- Windows 10/11
- Python 3.12.x
- Miniconda
- NVIDIA GPU (RTX 3060 Ti / RTX 4070 권장)
- CUDA Toolkit 12.8
- FFmpeg (비디오 분석용)

---

## 1. Miniconda 가상환경 생성

```bash
conda create -n media_manager python=3.12 -y
conda activate media_manager
```

---

## 2. 패키지 설치

```bash
pip install -r requirements.txt
```

설치되는 패키지 목록:

| 패키지 | 용도 |
|--------|------|
| torch / torchvision | 딥러닝 프레임워크 (CUDA 12.8 빌드) |
| PyQt6 | GUI 프레임워크 |
| transformers | CLIP, SigLIP2, MarianMT 모델 로드 |
| Pillow | 이미지 처리 |
| opencv-python | 컴퓨터 비전 |
| numpy | 수치 연산 |
| scikit-learn | 코사인 유사도 계산 |
| send2trash | 휴지통 삭제 (안전한 파일 삭제) |
| tqdm | 프로그레스 바 |
| tensorrt | TensorRT FP16 가속 |
| onnx | ONNX 모델 포맷 |
| onnxruntime-gpu | ONNX Runtime GPU 추론 |
| onnxscript | ONNX 스크립트 지원 |

---

## 3. HEIC / AVIF 이미지 지원 (선택)

iPhone 사진(HEIC) 또는 AVIF 포맷을 처리하려면 추가 설치:

```bash
pip install pillow-heif
```

설치하지 않아도 JPG/PNG/WebP 등 일반 포맷은 정상 동작합니다.

---

## 4. CUDA 확인

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

정상 출력 예시:
```
True
NVIDIA GeForce RTX 4070
```

`False`가 나오면 torch 재설치:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

---

## 5. FFmpeg 설치 (비디오 분석 기능 사용 시)

1. https://www.gyan.dev/ffmpeg/builds/ 에서 다운로드
2. 압축 해제 후 `bin` 폴더를 시스템 환경변수 PATH에 추가
3. 확인:

```bash
ffmpeg -version
ffprobe -version
```

또는 conda로 설치:
```bash
conda install -c conda-forge ffmpeg -y
```

---

## 6. 실행

```bash
conda activate media_manager
cd media_manager
python main.py
```

---

## 모델 다운로드

첫 실행 시 필요한 모델을 자동으로 프로젝트 내 `models/` 폴더에 다운로드합니다.

| 모델 | 크기 | 용도 |
|------|------|------|
| `apple/DFN5B-CLIP-ViT-H-14-378` | ~3.5 GB | 이미지 검색 |
| `google/siglip2-so400m-patch14-384` | ~900 MB | 이미지 검색 Beta |
| `Helsinki-NLP/opus-mt-ko-en` | ~300 MB | 한국어 번역 (CLIP용) |
| `facebookresearch/dinov2` (dinov2_vitb14) | ~330 MB | 이미지 중복 검출 |

> 인터넷 연결 필요. 이후 실행은 `models/` 폴더에서 오프라인 로드됩니다.

---

## 문제 해결

### TensorRT 엔진 빌드 실패

첫 실행 시 TensorRT 엔진을 자동 빌드합니다 (GPU별, 모델별 1~5분 소요). 빌드는 별도 서브프로세스에서 실행되어 앱이 충돌해도 안전하게 복구됩니다. 빌드된 엔진은 `_engine_cache/` 폴더에 GPU 모델별로 캐시됩니다. 빌드 실패 시 자동으로 PyTorch로 폴백합니다.

### CLIP 모델 다운로드가 느림

첫 실행 시 HuggingFace에서 CLIP 모델(~3.5GB)을 프로젝트 `models/` 폴더에 다운로드합니다. 이후 실행은 로컬에서 즉시 로드됩니다.

### onnxruntime-gpu CUDA 버전 불일치

```bash
pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
```

### transformers 버전 호환 문제 (SigLIP2)

SigLIP2는 `transformers 5.x`에서 GemmaTokenizer 관련 버그가 있습니다. 앱 내부에서 자동으로 패치되므로 별도 조치는 불필요합니다.

### 크래시 / 오류 진단

앱 실행 폴더에 자동 생성되는 로그 파일을 확인하세요:

- `crash_diag.log` — C 레벨 네이티브 크래시 스택 트레이스
- `unhandled_exception.log` — 미처리 Python 예외 및 Qt 오류

### 가상환경 삭제

```bash
conda deactivate
conda remove -n media_manager --all -y
```
