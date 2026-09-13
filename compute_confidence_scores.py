# -*- coding: utf-8 -*-
r"""
dashboard\images 폴더에 이미 있는 갤러리 이미지
(error_{n}_original.png / normal_{n}_original.png)를 그대로 활용해서
"불량일 확률"(판정 확신도의 원재료)을 계산하고 CSV로 저장한다.

generate_gallery_images.py가 만든 원본 리사이즈본을 재사용하는 방식이라
deep\resized\통합 원본 경로를 몰라도 되고, 이 파일 하나만 dashboard 폴더
안에서 실행하면 된다.

실행 위치: C:\python_study\final\dashboard (이 스크립트가 있는 바로 그 폴더)
    py -3.11 compute_confidence_scores.py

결과: dashboard\data\confidence_scores.csv
      (컬럼: label, category, sort_key, defect_prob)
이 CSV를 image_data.py의 load_confidence_scores()가 읽어서
대시보드의 "판정 확신도 분포" 그래프를 그린다.
"""

import os
import re
import glob

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.models import resnet18
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd

# ────────────────────────────────────────────────────────────
# 경로 설정 — 전부 dashboard 폴더 기준 상대경로
# ────────────────────────────────────────────────────────────
IMAGES_DIR = "images"                       # error_1_original.png 등이 있는 폴더
MODEL_PATH = os.path.join("models", "final_resnet18_deploy.pt")
OUTPUT_CSV = os.path.join("data", "confidence_scores.csv")

# error_1_original.png / normal_938_original.png 형태에서 (구분, 번호) 추출
FILENAME_PATTERN = re.compile(r"^(error|normal)_(\d+)_original\.png$", re.IGNORECASE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 갤러리 이미지가 이미 256x256으로 리사이즈되어 있지만, Resize를 한 번 더
# 걸어도 사이즈가 같으면 그대로 통과되므로(no-op) 안전하다 — 다른 곳(image_data.py,
# generate_gallery_images.py)의 eval_transform과 완전히 동일하게 맞춤.
eval_transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2(),
])


def load_model(model_path):
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        raise SystemExit(f"모델 파일이 없습니다: {MODEL_PATH}")

    all_files = glob.glob(os.path.join(IMAGES_DIR, "*_original.png"))
    print(f"이미지 {len(all_files)}장 발견 ({IMAGES_DIR} 폴더)")

    files_with_meta = []
    unmatched = []
    for fp in all_files:
        m = FILENAME_PATTERN.match(os.path.basename(fp))
        if not m:
            unmatched.append(fp)
            continue
        prefix, n = m.group(1).lower(), int(m.group(2))
        category = "불량" if prefix == "error" else "정상"
        label = f"{prefix.capitalize()}_{n}"  # image_data.py의 SampleImage.label과 동일 형식
        files_with_meta.append((fp, label, category, n))

    if unmatched:
        print(f"경고: 패턴에 안 맞아 건너뛴 파일 {len(unmatched)}개 (예: {unmatched[0]})")

    if not files_with_meta:
        raise SystemExit(f"처리할 이미지가 없습니다. IMAGES_DIR 경로를 확인하세요: {IMAGES_DIR}")

    model = load_model(MODEL_PATH)

    rows = []
    with torch.no_grad():
        for i, (fp, label, category, n) in enumerate(files_with_meta):
            img = Image.open(fp).convert("RGB")
            img_np = np.array(img)
            x = eval_transform(image=img_np)["image"].unsqueeze(0).to(device)
            logits = model(x)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            defect_prob = float(probs[0]) * 100  # 클래스 0 = 불량

            rows.append({
                "label": label,
                "category": category,
                "sort_key": n,
                "defect_prob": round(defect_prob, 4),
            })

            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(files_with_meta)} 처리 중...")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(f"\n저장 완료: {OUTPUT_CSV} (총 {len(df)}행)")
    print(df.groupby("category")["defect_prob"].describe())
