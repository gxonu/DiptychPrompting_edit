# 🧠 Subject-Driven Image Editing with Diptych Prompting

## 📘 Overview

이 프로젝트는 **FLUX + ControlNet Inpainting** 기반으로,

하나의 **참조 이미지(reference)** 에서 특정 **주제(subject)** 를 추출하여

다른 **타깃 이미지(target)** 의 지정된 **마스크 영역**에 삽입·편집하는 시스템입니다.

원래의 `diptych_prompting_inference.py`는 전체 패널을 통째로 생성하는 구조였으나,

이 수정 버전(`subject_driven_editing.py`)은 **부분 편집(partial inpainting)** 을 지원하며

**주제 인식(SAM + GroundingDINO)** 을 자동으로 수행합니다.

---

## ⚙️ Environment Setup

### 1. Conda 환경 생성 및 활성화

```bash
conda create -n diptychprompting python=3.10
conda activate diptychprompting

```

### 2. Requirements 설치

```bash
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
pip install -r requirements.txt

```

> 💡 주의: FLUX 모델은 40GB 이상의 VRAM을 요구합니다.
> 
> 
> A100(40GB 이상) 혹은 동일급 GPU 환경에서 실행하는 것을 권장합니다.
> 

---

## 🧩 Workflow (전체 작업 순서)

### Step 1️⃣ — 마스크 좌표 확인 (`check_mask.py`)

먼저 target 이미지 위에 마스크가 올바르게 덮이는지 시각적으로 확인해야 합니다.

```bash
python check_mask.py \
  --reference_image ./assets/reference.jpg \
  --target_image ./assets/target.jpg \
  --mask_x 200 --mask_y 250 --mask_w 350 --mask_h 300 \
  --width 512 --height 512 --pixel_offset 0

```

생성 결과:

- `mask_check/debug_mask_overlay.png` → 왼쪽은 참조 이미지, 오른쪽은 target + 회색 마스크
- `mask_check/debug_masked_target.png` → target 이미지만 회색 마스크로 덮인 버전

이 단계에서 좌표가 잘 맞는지 확인 후 `subject_driven_editing.py`에 동일한 값을 사용합니다.

---

### Step 2️⃣ — 본격 편집 실행 (`subject_driven_editing.py`)

```bash
python subject_driven_editing.py \
  --reference_image_path ./assets/reference.jpg \
  --target_image_path ./assets/target.jpg \
  --subject_name "bear plushie" \
  --mask_type rectangle \
  --mask_x 200 --mask_y 250 --mask_w 350 --mask_h 300 \
  --prompt "a bear plushie sitting on a tree branch" \
  --output_path ./outputs/edited.png

```

---

## 🧠 내부 동작 요약

| 단계 | 내용 |
| --- | --- |
| ① GroundingDINO | `--subject_name`으로 입력된 객체를 reference 이미지에서 탐지 |
| ② SAM (Segment Anything) | 탐지된 영역을 정밀 분할하여 주제만 추출 |
| ③ Diptych 구성 | `[왼쪽] = 분할된 주제`, `[오른쪽] = target + 회색 마스크` |
| ④ FLUX ControlNet Inpainting | 마스크 내부를 prompt에 따라 재생성 |
| ⑤ Hard Inpainting Fusion | `(1-M)*target + M*edited`로 결과를 합성 |

---

## 🧾 주요 인자 설명

| 인자 | 설명 |
| --- | --- |
| `--reference_image_path` | 주제(Subject)가 포함된 참조 이미지 |
| `--target_image_path` | 편집 대상 이미지 |
| `--subject_name` | GroundingDINO가 인식할 객체명 |
| `--prompt` | 편집 후 오른쪽 패널을 묘사하는 문장 |
| `--mask_type` | 마스크 형태 선택 (`rectangle`, `circle`, `custom`) |
| `--mask_x, --mask_y, --mask_w, --mask_h` | 사각형 마스크 좌표 |
| `--mask_path` | custom grayscale mask 경로 |
| `--attn_enforce` | 좌·우 패널 cross-attention 강도 |
| `--ctrl_scale` | ControlNet 조건 강도 |
| `--guidance_scale` | 텍스트 프롬프트의 가중치 |
| `--num_inference_steps` | denoising step 수 |
| `--pixel_offset` | 테두리 artifact 제거용 crop offset |
| `--output_path` | 결과 이미지 저장 경로 (폴더 자동 생성) |

---

## 📤 출력 결과

| 파일 | 설명 |
| --- | --- |
| `edited.png` | 최종 하드 인페인팅된 결과 이미지 |
| `mask.png` | 사용된 실제 마스크 (cropped 버전) |
| `full_diptych.png` | 좌: reference / 우: 최종 결과 패널 전체 |
| `debug_full_mask_L.png` | full binary mask (L 모드) |
| `debug_diptych_control.png` | 마스크가 덮인 control diptych |

---

## 💡 권장 팁

1. **좌표 검증 필수:** `check_mask.py`로 항상 먼저 마스크 위치를 확인하세요.
2. **prompt 구체화:** `"a small bear plushie sitting on a tree branch"` 처럼 구체적으로 작성할수록 결과 품질이 높습니다.
3. **VRAM 관리:** `-height` / `-width`를 512로 줄이면 24GB GPU에서도 제한적으로 실행 가능.
4. **밝기 일치:** Reference와 Target의 조명·배경이 유사할수록 결과가 자연스럽습니다.

---

## 🧪 예시 실행 시나리오

**목표:** 곰인형(bear plushie)을 새(bird) 사진의 나뭇가지 위에 배치

```bash
# 1️⃣ 마스크 위치 확인
python check_mask.py \
  --reference_image ./assets/bear_plushie.jpg \
  --target_image ./assets/bird1.jpg \
  --mask_x 200 --mask_y 250 --mask_w 350 --mask_h 300 \
  --width 512 --height 512 --pixel_offset 0

# 2️⃣ 실제 편집
python subject_driven_editing.py \
  --reference_image_path ./assets/bear_plushie.jpg \
  --target_image_path ./assets/bird1.jpg \
  --subject_name "bear plushie" \
  --mask_type rectangle \
  --mask_x 200 --mask_y 250 --mask_w 350 --mask_h 300 \
  --prompt "a bear plushie sitting on a tree branch" \
  --output_path ./outputs/bird_with_bear.png

```

---

## 💾 Output Directory Structure

```
output/
└── edit/
    └── bird1_bear_plushie_1030_2124/
        ├── edited.png
        ├── mask.png
        ├── full_diptych.png
        ├── debug_full_mask_L.png
        └── debug_diptych_control.png

```

---

## 📦 System Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.5.1 (CUDA 12.4)
- GPU Memory ≥ 40GB (A100 권장)
- Dependencies:
    - `transformers`, `diffusers`, `opencv-python`, `torchvision`, `Pillow`, `numpy`