# Media Manager

이미지 검색, 이미지 중복 검출, 비디오 중복 검출을 하나의 Apple 스타일 데스크톱 앱으로 통합한 미디어 분석 프로그램.

---

## 기능

### 1. 이미지 검색 (CLIP)

텍스트 또는 이미지로 이미지를 찾는 시맨틱 검색 기능.

- **모델**: `apple/DFN5B-CLIP-ViT-H-14-378` (ViT-H-14, 1024차원 임베딩)
- **오프라인 우선**: 모델 로컬 `models/` 디렉토리 우선 로드, 최초 1회만 다운로드
- **한국어 지원**: 한국어 입력 시 MarianMT(`Helsinki-NLP/opus-mt-ko-en`)로 자동 번역 후 검색, 번역 결과 인메모리 캐싱
- **파노라마 처리**: 가로 비율 3.0 초과 이미지는 자동 분할 후 세그먼트별 max 유사도 매칭
- **배치 처리**: GPU VRAM 기반 동적 배치 크기 (8GB→16, 12GB→24) + 스레드 풀 이미지 로딩
- **OOM 복구**: GPU 메모리 부족 시 배치 크기 자동 축소 후 재시도
- **캐싱**: SQLite 기반 영구 캐시 (모델 해시별 관리), 배치 조회/저장 지원
- **검색 모드**:
  - **텍스트 검색**: 텍스트 쿼리로 이미지 검색 (한국어/영어)
  - **이미지 검색**: 쿼리 이미지와 유사한 이미지 검색 (Browse 버튼)
- **서브폴더 검색**: Subfolders 체크박스로 하위 폴더까지 재귀 검색
- **UI**: 결과 개수 조절 (5~50), Min score 슬라이더 (실시간 결과 필터링), 썸네일 그리드, 클릭 시 원본 미리보기

### 2. 이미지 검색 Beta (SigLIP2)

Google SigLIP2 기반 시맨틱 검색. 번역 모델 없이 한국어/영어 네이티브 지원.

- **모델**: `google/siglip2-so400m-patch14-384` (1152차원 임베딩)
- **오프라인 우선**: 모델 로컬 `models/` 디렉토리 우선 로드
- **네이티브 다국어**: 한국어/영어를 번역 없이 직접 처리
- **추론 엔진**: 이미지 인코더 TensorRT FP16 > PyTorch 자동 폴백 (텍스트 인코더는 PyTorch)
- **배치 처리**: GPU VRAM 기반 동적 배치 크기 (TRT: 8GB→32, 12GB→48 / PyTorch: 8GB→16, 12GB→24)
- **OOM 복구**: GPU 메모리 부족 시 배치 크기 자동 축소 후 재시도
- **캐싱**: CLIP과 동일한 SQLite 캐시 테이블 공유 (model_hash로 구분)
- **검색 모드**: 텍스트 검색 + 이미지-by-이미지 검색 모두 지원
- **서브폴더 검색**: 재귀 검색 지원
- **점수 임계값**: green > 20%, accent > 10% (CLIP보다 낮음, sigmoid loss 특성)
- **transformers 5.x 호환**: GemmaTokenizer 자동 패치

### 3. 이미지 중복 검출 (DINOv2)

자기지도 학습 기반 특징 벡터 비교로 중복/유사 이미지 탐지.

- **모델**: DINOv2-Base (`facebookresearch/dinov2` → `dinov2_vitb14`, 768차원 특징 벡터)
- **추론 엔진**: TensorRT FP16 > PyTorch 자동 폴백
- **모드**:
  - **1폴더 모드**: 폴더 내 모든 이미지 N x N 비교
  - **2폴더 모드**: 두 폴더 간 M x N 교차 비교
- **비교 패널**: 이미지 나란히 표시 + 차이 히트맵 (Beyond Compare 스타일)
- **분석 카드**: 포맷, 해상도, 파일 크기, 픽셀 차이를 한눈에 확인
- **기능**: 중복 이미지 휴지통 삭제 (send2trash), 결과 텍스트 파일 내보내기
- **유사도 조절**: 0.80 ~ 1.00 슬라이더
- **파이프라인 처리**: CPU 전처리(스레드 풀 4스레드)와 GPU 추론을 동시 실행, VRAM 기반 동적 배치 (8GB→96, 12GB→128, TensorRT 시 1.5배)
- **캐싱**: 특징 벡터 SQLite 캐시, 배치 조회/저장 지원 (반복 스캔 시 즉시 로드)
- **진행 상태**: 단계별 상태 메시지 + ETA(예상 남은 시간) 실시간 표시
- **안전한 중단/재시작**: Stop 후 폴더 변경 → 재스캔 시 이전 스레드 안전하게 정리

### 4. 비디오 중복 검출 (Perceptual Hash)

중복, 동일 콘텐츠, 부분 일치 영상 탐지.

- **방식**: DCT 기반 프레임 해시 + GPU 코사인 유사도
- **연속 프레임 스킵**: 해밍 거리 ≤ 2인 연속 유사 프레임 자동 필터링 (노이즈 제거)
- **매칭 유형**:
  - **완전 일치**: 파일 해시 + 크기 동일 (100% 일치)
  - **동일 콘텐츠**: 같은 영상 다른 인코딩 (유사도 >= 90%)
  - **부분 일치**: 짧은 영상이 긴 영상에 포함 (슬라이딩 윈도우, >= 85%)
- **파일 해시**: 시작/중간/끝 4MB씩 3구간 샘플링 (총 12MB)
- **모드**: 1폴더 / 2폴더 비교 지원
- **프레임 추출**: FFmpeg 우선, OpenCV 폴백
- **멀티프로세싱**: ProcessPoolExecutor로 병렬 분석
- **캐싱**: 비디오 메타데이터 + 프레임 해시 SQLite 캐시
- **내보내기**: JSON 형식

---

## 프로젝트 구조

```
media_manager/
├── main.py                          # 앱 진입점 (크래시 진단 로그, Qt 메시지 핸들러)
├── requirements.txt                 # 패키지 목록
├── INSTALL.md                       # 설치 가이드
├── README.md                        # 프로젝트 설명 (이 파일)
│
├── ui/                              # PyQt6 UI 레이어
│   ├── styles.py                    # Apple 스타일 QSS 테마
│   ├── components.py                # 공용 위젯 (FolderPicker, InfoCard, StatusBar 등)
│   ├── main_window.py               # 메인 윈도우 + 사이드바 네비게이션 + GPU 관리
│   ├── image_search_page.py         # 이미지 검색 페이지 (CLIP)
│   ├── image_search_siglip_page.py  # 이미지 검색 페이지 (SigLIP2 beta)
│   ├── image_duplicate_page.py      # 이미지 중복 검출 페이지
│   ├── video_duplicate_page.py      # 비디오 중복 검출 페이지
│   ├── cache_page.py                # 캐시 통계 + 삭제 (5번째 탭)
│   └── update_dialog.py             # 모델 업데이트 알림 + 다운로드 다이얼로그
│
├── core/                            # 추론 엔진
│   ├── model_paths.py               # 중앙화된 모델 저장 경로 (models/ 폴더)
│   ├── model_updater.py             # 백그라운드 모델 업데이트 체커 + 다운로더
│   ├── inference_engine.py          # TensorRT 엔진 빌드/캐시 + DINOv2 TRT + 공용 유틸
│   ├── clip_engine.py               # CLIP 모델 래퍼 (TRT 서브프로세스 빌드)
│   ├── siglip2_engine.py            # SigLIP2 모델 래퍼 (TRT 서브프로세스 빌드)
│   ├── resnet_engine.py             # DINOv2 특징 추출 (TRT/PyTorch, ETA 표시)
│   ├── video_analyzer.py            # 비디오 분석 엔진 (프레임 해시)
│   ├── cache_manager.py             # 통합 SQLite 캐시 (3개 테이블, 배치 조회 지원)
│   ├── config.py                    # JSON 폴더 경로 설정 저장
│   └── translation.py               # 한→영 MarianMT 번역기 (오프라인 우선)
│
├── utils/                           # 유틸리티
│   ├── image_utils.py               # 이미지 로드 (HEIC/AVIF 지원), 전처리, 해상도
│   └── video_utils.py               # FFmpeg/FFprobe 래퍼
│
├── models/                          # (자동 생성) HuggingFace + torch.hub 모델 캐시
├── _engine_cache/                   # (자동 생성) TensorRT 엔진 캐시 (.engine, .onnx)
├── _cache/                          # (자동 생성) SQLite 데이터베이스
├── _config/                         # (자동 생성) 사용자 설정 (폴더 경로 등)
├── crash_diag.log                   # (자동 생성) C-레벨 크래시 덤프 (SIGSEGV/SIGABRT)
└── unhandled_exception.log          # (자동 생성) 미처리 Python/Qt 예외 로그
```

---

## 추론 파이프라인

```
우선순위: TensorRT FP16  >  PyTorch CUDA  >  PyTorch CPU
```

### DINOv2 (이미지 중복)

```
이미지 파일들
  -> [폴더 스캔] 이미지 파일 수집 + 상태 표시 ("Scanning folder...", "Found N images")
  -> [배치 캐시 조회] SQLite WHERE IN 쿼리로 캐시 히트 일괄 확인 ("Checking cache...")
  -> [캐시 미스] 미캐시 이미지만 추출 대상 선정 ("Cache: X/Y, extracting Z remaining...")
  -> [CPU 스레드 풀] OpenCV 로드 + 224x224 리사이즈 + ImageNet 정규화 (4스레드 병렬)
  -> [파이프라인] 다음 배치 전처리 ↔ 현재 배치 GPU 추론 동시 실행
  -> DINOv2-Base CLS 토큰 추출 (동적 배치: 8GB→96, 12GB→128)
     ├── TensorRT FP16 엔진 (가능한 경우, 배치 1.5배)
     └── PyTorch 모델 (폴백)
  -> 768차원 특징 벡터 + ETA 실시간 표시 ("Extracting: 150/500 (ETA 1m 23s)")
  -> [배치 캐시 저장] executemany로 일괄 저장 ("Saving N features to cache...")
  -> [해상도 수집] "Collecting resolutions..."
  -> [유사도 계산] "Computing similarity matrix..."
  -> 코사인 유사도 행렬 계산
  -> 임계값 필터링 -> 중복 쌍 출력 + ETA ("Comparing: 50,000/150,000 (ETA 12s)")
```

### CLIP (이미지 검색)

```
텍스트 쿼리
  -> 한국어 감지 -> MarianMT 번역 (한국어인 경우, 결과 인메모리 캐싱)
  -> 번역 모델 오프라인 우선 로드 (local_files_only → OSError 시 온라인 폴백)
  -> CLIP 텍스트 인코더 -> 텍스트 임베딩 (1024차원)

또는 이미지 쿼리 (이미지 검색 모드)
  -> 쿼리 이미지 CLIP 임베딩 계산

이미지 파일들
  -> [배치 캐시 조회] SQLite WHERE IN 쿼리로 캐시 히트 일괄 확인
  -> [CPU 스레드 풀] PIL 로드 + 리사이즈 (최대 1024px) (4스레드 병렬)
  -> 파노라마 분할 (가로 비율 > 3.0인 경우, 세그먼트별 임베딩 개별 저장)
  -> CLIP 비전 인코더 -> 이미지 임베딩 (1024차원, 동적 배치: 8GB→16, 12GB→24)
  -> OOM 시 배치 크기 자동 축소 후 재시도

검색: 코사인 유사도(텍스트/이미지 임베딩, 이미지 임베딩) -> 파노라마는 세그먼트별 max -> Top-K 결과
```

### SigLIP2 (이미지 검색 Beta)

```
텍스트/이미지 쿼리 (한국어/영어 번역 없이 직접 처리)
  -> SigLIP2 텍스트/이미지 인코더 -> 임베딩 (768차원)

이미지 파일들
  -> [배치 캐시 조회] SQLite WHERE IN 쿼리
  -> [CPU 스레드 풀] PIL 로드 + 리사이즈 (최대 1024px) (4스레드 병렬)
  -> SigLIP2 이미지 인코더 -> 임베딩 (768차원)
     ├── TensorRT FP16 엔진 (8GB→96, 12GB→128)
     └── PyTorch 모델 (폴백: 8GB→48, 12GB→64)
  -> OOM 시 배치 크기 자동 축소 후 재시도

검색: 코사인 유사도 -> Top-K 결과
```

### 비디오 분석

```
비디오 파일
  -> FFprobe 메타데이터 (해상도, FPS, 길이, 프레임 수)
  -> FFmpeg 프레임 추출 (초당 1프레임) [OpenCV 폴백]
  -> 프레임별: 8x8 리사이즈 -> 그레이스케일 -> DCT -> 해시 (16바이트)
  -> 연속 유사 프레임 스킵 (해밍 거리 ≤ 2)
  -> 파일 해시: 시작/중간/끝 4MB씩 3구간 샘플링
  -> 비교:
     ├── 완전 일치: 파일 해시 + 크기 비교
     ├── 동일 콘텐츠: 프레임 해시 대각선 코사인 유사도 (GPU)
     └── 부분 일치: 긴 영상에서 슬라이딩 윈도우 검색
```

---

## 캐시 시스템

단일 SQLite 데이터베이스 (`_cache/media_cache.db`) 에 3개 테이블:

| 테이블 | 내용 | 키 |
|--------|------|-----|
| `video_cache` | 프레임 해시, 메타데이터, 길이, 해상도 | MD5(크기 + 수정시간 + 전체경로) |
| `image_feature_cache` | DINOv2 768차원 특징 벡터 | MD5(크기 + 수정시간 + 전체경로) |
| `clip_cache` | CLIP/SigLIP2 임베딩, 모델별 관리 | MD5(크기 + 수정시간 + 전체경로) + model_hash |

**캐시 무효화**: 파일 크기 또는 수정 시간이 변경되면 캐시가 자동 무효화. 파일 단위로 관리되므로 폴더에 이미지 추가/삭제 시 변경되지 않은 기존 파일은 캐시 재사용.

배치 조회(`WHERE IN`) 및 배치 저장(`executemany`)으로 반복 스캔 시 I/O 최소화.

**캐시 관리 탭**: 모델별 캐시 통계 (CLIP, SigLIP2, DINOv2, Video, DB 크기), 선택 삭제, 무효 항목 삭제, 전체 삭제. 시작 시 자동으로 삭제/변경된 파일의 캐시 엔트리 제거.

---

## 모델 업데이트 시스템

앱 시작 5초 후 백그라운드에서 자동으로 모델 업데이트를 체크:

- **HuggingFace 모델**: SHA 비교로 업데이트 감지 (CLIP, MarianMT, SigLIP2)
- **torch.hub 모델**: GitHub API SHA 비교로 업데이트 감지 (DINOv2)
- **업데이트 알림**: 다이얼로그로 알림, "지금 업데이트" 또는 "나중에" 선택
- **다운로드 + 재시작**: 업데이트 다운로드 후 앱 자동 재시작
- **TRT 엔진 캐시 무효화**: 모델 업데이트 시 관련 .engine 파일 자동 삭제 (다음 시작 시 재빌드)

---

## TRT 엔진 빌드 (서브프로세스 격리)

TensorRT 엔진 빌드는 별도 서브프로세스에서 실행:

- **주 프로세스 보호**: TRT 네이티브 크래시 시 Qt 앱이 종료되지 않음
- **TracerWarning 격리**: ONNX export 경고가 서브프로세스에만 나타남
- **Windows 충돌 다이얼로그 억제**: WER 다이얼로그 팝업 없이 즉시 종료
- **10분 타임아웃**: 빌드가 무한 대기되지 않도록 자동 강제 종료
- **글로벌 Lock**: 동시 TRT 빌드 방지 (GPU 충돌 예방)
- **동적 배치 범위**: 엔진 빌드 시 최소/최적/최대 배치 크기 최적화 프로파일 포함

---

## 스캔 중단/재시작 안전성

- **시그널 disconnect**: Stop 시 이전 스레드의 progress/finished/error 시그널 분리 → 새 스캔 오염 방지
- **scan ID**: 각 스캔에 고유 ID 부여, `finally` 블록에서 자기 세션만 `is_processing` 해제 → 이전 스레드가 새 스캔을 중단하는 문제 방지
- **상태 초기화**: 재스캔 시 progress bar, 상태 텍스트, 결과 목록 즉시 초기화
- **탭 전환 시 확인**: 작업 중 탭 이동 시 중단 여부 사용자 확인 다이얼로그

---

## 오류 진단

앱 실행 폴더에 자동 생성:

| 파일 | 내용 |
|------|------|
| `crash_diag.log` | C 레벨 네이티브 크래시 스택 트레이스 (SIGSEGV, SIGABRT) |
| `unhandled_exception.log` | 미처리 Python 예외, Qt 스레드 예외, Qt qFatal 메시지 |

---

## UI 디자인

Apple 스타일 디자인 시스템:

- **레이아웃**: 좌측 사이드바 (200px) + 우측 컨텐츠 영역
- **색상**: 라이트 테마 (배경 `#f5f5f7`, 액센트 `#0071e3`)
- **폰트**: Segoe UI (Windows)
- **카드**: 둥근 모서리 (10px), 미세 보더
- **컨트롤**: 커스텀 프로그레스 바, 세그먼트 탭, 분석 카드
- **폴더 선택**: 드래그 앤 드롭 지원, 페이지별 마지막 경로 자동 저장
- **상태바**: GPU 이름 + VRAM 표시
- **시작 시 캐시 자동 정리**: 삭제/변경된 파일의 캐시 엔트리 백그라운드 제거
- **로그 정리**: httpx, huggingface_hub 등 불필요 HTTP 로그 억제

---

## 대상 하드웨어

| GPU | VRAM | TensorRT | 비고 |
|-----|------|----------|------|
| RTX 3060 Ti | 8GB | 지원 | FP16 엔진 GPU별 캐시 |
| RTX 4070 | 12GB | 지원 | FP16 엔진 GPU별 캐시 |

CUDA 12.8 호환. TensorRT 엔진은 GPU 모델당 최초 1회 빌드 후 `_engine_cache/`에 캐시됩니다.

---

## 지원 포맷

- **이미지**: JPG, JPEG, PNG, BMP, GIF, WebP, TIF, TIFF, HEIC, HEIF, AVIF
  - HEIC/HEIF/AVIF는 `pillow-heif` 패키지 설치 시 지원
- **비디오**: MP4, AVI, MKV, MOV, WMV, FLV, WebM, M4V
