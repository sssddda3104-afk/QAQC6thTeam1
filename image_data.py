"""
이미지 데이터 페이지 지원 모듈.

deep_final_clean2.ipynb 기준 최종 배포 모델:
- 아키텍처: torchvision resnet18, fc를 Linear(in_features, 2)로 교체
- 전처리: Resize(256,256) → Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
  (주의: ImageNet 정규화는 6번 섹션의 별도 freeze 비교 실험에서만 쓰였고,
   실제 배포 모델은 baseline(0.5,0.5,0.5) 정규화로 학습됨)
- 클래스 인덱스: 0=불량, 1=정상 (classification_report의 labels=[0,1],
  target_names=["불량","정상"] 순서와 동일)
- 가중치 파일: models/final_resnet18_deploy.pt (state_dict)
"""

from pathlib import Path
from dataclasses import dataclass
from functools import lru_cache
import re
import io
import pandas as pd

IMG_DIR = Path(__file__).parent / "images"
MODEL_PATH = Path(__file__).parent / "models" / "final_resnet18_deploy.pt"
CONFIDENCE_CSV = Path(__file__).parent / "data" / "confidence_scores.csv"


@lru_cache(maxsize=1)
def load_confidence_scores() -> pd.DataFrame | None:
    """compute_confidence_scores.py로 미리 계산해둔 1,452장 전체의 판정
    판정 확률(불량일 확률)을 읽어온다. 파일이 없으면 None을 반환 — 호출부에서
    "아직 계산 안 됨" 안내를 보여주면 된다."""
    if not CONFIDENCE_CSV.exists():
        return None
    return pd.read_csv(CONFIDENCE_CSV)

IMG_SIZE = 256
NORM_MEAN = (0.5, 0.5, 0.5)
NORM_STD = (0.5, 0.5, 0.5)
CLASS_NAMES = {0: "불량", 1: "정상"}

# ── deep_final_clean2.ipynb 8번 섹션(최종 결과 비교)에서 그대로 옮긴 값 ──
MODEL_META = {
    "final_model": "ResNet18 (Transfer Learning)",
    "f1_mean": 1.000,
    "f1_std": 0.000,
    "normal_count": 1378,
    "defect_count": 74,
    "unique_defect_groups": 26,  # 근접중복 제거 후 실질 그룹 수
}

MODEL_COMPARISON = pd.DataFrame([
    {"모델": "ResNet18", "방식": "지도(Transfer)", "F1": "1.000 ± 0.000", "비고": "최종 선정"},
    {"모델": "PatchCore", "방식": "비지도", "F1": "0.962 ± 0.052", "비고": "AUROC 1.000±0.000"},
    {"모델": "Scratch CNN", "방식": "지도(From-scratch)", "F1": "0.957 ± 0.072", "비고": "상대적으로 부정확"},
    {"모델": "PaDiM", "방식": "비지도", "F1": "0.847 ± 0.186", "비고": "AUROC 0.991±0.020"},
])


@dataclass
class SampleImage:
    label: str          # 화면에 보일 이름
    category: str       # "정상" | "불량"
    original_path: Path
    gradcam_path: Path | None  # 정상 이미지는 Grad-CAM이 따로 없을 수 있음
    sort_key: int = 0    # 파일명의 숫자 — 갤러리 정렬용


# 지금 실제로 갖고 있는 샘플. 나중에 team이 이미지를 더 주면
# 이 리스트에 항목만 추가하면 화면에 바로 반영된다.
#
# generate_gallery_images.py로 74장(불량)+1,378장(정상) 전체를
# images/ 폴더에 생성해둔 상태이므로, 하드코딩 대신 폴더를 스캔해서
# 자동으로 목록을 구성한다. 파일명 규칙: {prefix}_{n}_original.png /
# {prefix}_{n}_gradcam.png (prefix: error=불량, normal=정상)
_FILENAME_PATTERN = re.compile(r"^(error|normal)_(\d+)_original\.png$")


def _build_sample_images() -> list["SampleImage"]:
    if not IMG_DIR.exists():
        return []

    found = []
    for fp in IMG_DIR.glob("*_original.png"):
        m = _FILENAME_PATTERN.match(fp.name)
        if not m:
            continue
        prefix, n = m.group(1), int(m.group(2))
        category = "불량" if prefix == "error" else "정상"
        gradcam_path = IMG_DIR / f"{prefix}_{n}_gradcam.png"
        found.append(
            SampleImage(
                label=f"{prefix.capitalize()}_{n}",
                category=category,
                original_path=fp,
                gradcam_path=gradcam_path if gradcam_path.exists() else None,
                sort_key=n,
            )
        )
    # 번호(숫자) 기준 정렬 — 카테고리별로 1,2,3... 순서가 되도록
    found.sort(key=lambda s: (s.category, s.sort_key))
    return found


SAMPLE_IMAGES: list["SampleImage"] = _build_sample_images()


def refresh_samples() -> int:
    """images/ 폴더를 다시 스캔해서 SAMPLE_IMAGES를 갱신한다.
    새로고침 버튼에서 호출 — Streamlit 앱을 재시작하지 않아도
    폴더에 새로 추가된 이미지가 반영되도록 함.
    반환값: 갱신 후 총 이미지 개수.
    """
    global SAMPLE_IMAGES
    SAMPLE_IMAGES = _build_sample_images()
    return len(SAMPLE_IMAGES)


def get_samples_df() -> pd.DataFrame:
    rows = [{"번호": s.label, "대분류": s.category} for s in SAMPLE_IMAGES]
    return pd.DataFrame(rows)


def get_sample(label: str) -> SampleImage | None:
    for s in SAMPLE_IMAGES:
        if s.label == label:
            return s
    return None


@lru_cache(maxsize=1)
def _load_model():
    """모델을 한 번만 로드해서 캐시 — 매번 새로 읽으면 느리다."""
    import torch
    import torch.nn as nn
    from torchvision.models import resnet18

    m = resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 2)
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    m.load_state_dict(state_dict)
    m.eval()
    return m


def predict_image(image_bytes: bytes) -> dict:
    """실제 ResNet18 추론 + Grad-CAM 생성. 가중치 파일이 없으면 available=False로 안내."""
    if not MODEL_PATH.exists():
        return {
            "available": False,
            "message": f"모델 가중치 파일이 없습니다 ({MODEL_PATH.name}을 models/ 폴더에 넣어주세요).",
        }

    try:
        import torch
        import torch.nn.functional as F
        from PIL import Image
        import numpy as np
        import torchvision.transforms as T
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        from pytorch_grad_cam.utils.image import show_cam_on_image

        model = _load_model()

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        transform = T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=NORM_MEAN, std=NORM_STD),
        ])
        x = transform(img).unsqueeze(0)

        # ── 1) 예측 (no_grad로 빠르게) ──
        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits, dim=1).numpy()[0]
        pred_idx = int(np.argmax(probs))

        # ── 2) Grad-CAM 생성 — 예측된 클래스 기준(왜 그렇게 판정했는지 설명)
        #    generate_gallery_images.py에서는 갤러리용으로 항상 클래스 0(불량)
        #    기준으로 생성했지만, 여기서는 실제 판정 근거를 보여주는 게 목적이라
        #    예측된 클래스를 타깃으로 함. GradCAM은 내부적으로 forward+backward를
        #    수행하므로 no_grad 밖에서 새로 호출해야 함.
        #    with 문으로 사용해야 호출 후 등록된 훅(hook)이 해제된다 —
        #    안 그러면 업로드할 때마다 모델에 훅이 계속 쌓여서 반복 사용 시
        #    메모리 누수·오작동 위험이 있음.
        with GradCAM(model=model, target_layers=[model.layer4[-1]]) as cam:
            grayscale_cam = cam(input_tensor=x, targets=[ClassifierOutputTarget(pred_idx)])[0]

        img_resized = np.array(img.resize((IMG_SIZE, IMG_SIZE)))
        img_norm = img_resized.astype(np.float32) / 255.0
        vis = show_cam_on_image(img_norm, grayscale_cam, use_rgb=True)

        buf = io.BytesIO()
        Image.fromarray(vis).save(buf, format="PNG")
        gradcam_bytes = buf.getvalue()

        return {
            "available": True,
            "prediction": CLASS_NAMES[pred_idx],
            "confidence": float(probs[pred_idx]),
            "probs": {CLASS_NAMES[i]: float(probs[i]) for i in range(2)},
            "gradcam_image": gradcam_bytes,
        }
    except Exception as e:
        return {"available": False, "message": f"추론 중 오류가 발생했습니다: {e}"}

