import os
import random
import json
from albumentations import (
    Compose, OneOf, RandomBrightnessContrast, HueSaturationValue, MotionBlur, GaussNoise, GaussianBlur, Affine
)
from tqdm import tqdm
import cv2

# 사용자 설정
ORIGINAL_PATH = "C:/dataset/train/original"  # 원본 데이터 경로
OUTPUT_PATH = "C:/dataset/train/augmented"  # 증강 데이터 저장 경로
AUGMENTATION_PROPORTION = 0.3  # 각 증강 기법에 적용할 비율 (30%)
LOG_FILE = "augmentation_log.json"  # 증강 로그 파일

# 증강 함수 정의
def get_augmentations():
    """
    각 증강 기법을 정의하고 주요 파라미터를 설명합니다.
    """
    return {
        "brightness": RandomBrightnessContrast(
            p=1.0,  # 이 증강이 적용될 확률
            brightness_limit=0.2,  # 밝기 변화 범위 (예: 0.2는 ±20% 범위에서 밝기 조정)
            contrast_limit=0.2  # 대비 변화 범위 (예: 0.2는 ±20% 범위에서 대비 조정)
        ),
        "contrast": RandomBrightnessContrast(
            p=1.0,
            brightness_limit=0,  # 밝기는 변화시키지 않음
            contrast_limit=0.5  # 대비를 최대 ±50%까지 조정
        ),
        "hue": HueSaturationValue(
            p=1.0,  # 이 증강이 적용될 확률
            hue_shift_limit=10,  # 색조 변경 범위 (예: ±10)
            sat_shift_limit=20,  # 채도 변경 범위 (예: ±20)
            val_shift_limit=10  # 밝기 변경 범위 (예: ±10)
        ),
        "motion_blur": MotionBlur(
            p=1.0,  # 이 증강이 적용될 확률
            blur_limit=5  # 블러 강도 (커널 크기 최대값: 5)
        ),
        "gaussian_noise": GaussNoise(
            p=1.0,  # 이 증강이 적용될 확률
            var_limit=(10.0, 50.0)  # 가우시안 노이즈의 분산 범위 (예: 10~50)
        ),
        "affine": Affine(
            p=1.0,  # 이 증강이 적용될 확률
            scale=(0.8, 1.2),  # 스케일 변환 범위 (예: 80%~120%)
            rotate=(-15, 15)  # 회전 각도 범위 (예: -15도에서 15도)
        )
    }

# 파이프라인 정의
def get_augmentation_pipeline():
    """
    증강 파이프라인을 정의하고 순차적/랜덤 적용 방식을 설명합니다.
    """
    return Compose([
        RandomBrightnessContrast(
            p=0.8  # 80% 확률로 밝기/대비 조정
        ),
        HueSaturationValue(
            p=0.6  # 60% 확률로 색조/채도/밝기 조정
        ),
        OneOf([  # 아래 세 가지 중 하나를 랜덤하게 선택
            MotionBlur(p=0.5),  # 50% 확률로 모션 블러
            GaussianBlur(p=0.5),  # 50% 확률로 가우시안 블러
            GaussNoise(p=0.5)  # 50% 확률로 가우시안 노이즈 추가
        ], p=0.7),  # 위의 세 가지 중 하나를 70% 확률로 선택
        Affine(
            p=0.5  # 50% 확률로 아핀 변환 적용
        )
    ])

# 증강 적용 및 저장 함수
def apply_augmentation(input_dir, output_dir, augmentations, proportion):
    os.makedirs(output_dir, exist_ok=True)
    logs = []  # 증강 로그 저장
    image_dir = os.path.join(input_dir, "images")
    label_dir = os.path.join(input_dir, "labels")
    generated_counts = {aug_name: 0 for aug_name in augmentations.keys()}  # 생성된 파일 수 카운트

    if not os.path.exists(image_dir) or not os.path.exists(label_dir):
        raise FileNotFoundError("이미지 또는 라벨 디렉토리가 존재하지 않습니다.")

    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    for aug_name, transform in augmentations.items():
        aug_dir = os.path.join(output_dir, aug_name)
        os.makedirs(os.path.join(aug_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(aug_dir, "labels"), exist_ok=True)

        # 30% 비율로 랜덤 샘플 선택
        selected_files = random.sample(image_files, int(len(image_files) * proportion))

        for file in tqdm(selected_files, desc=f"Applying {aug_name}"):
            img_path = os.path.join(image_dir, file)
            label_path = os.path.join(label_dir, os.path.splitext(file)[0] + ".txt")

            # 이미지 읽기
            image = cv2.imread(img_path)
            if image is None:
                continue

            # 증강 적용
            augmented = transform(image=image)["image"]

            # 새로운 파일 이름 생성
            new_filename = f"{os.path.splitext(file)[0]}_{aug_name}{os.path.splitext(file)[1]}"
            output_img_path = os.path.join(aug_dir, "images", new_filename)
            output_label_path = os.path.join(aug_dir, "labels", f"{os.path.splitext(file)[0]}_{aug_name}.txt")

            # 증강된 이미지와 라벨 저장
            cv2.imwrite(output_img_path, augmented)
            if os.path.exists(label_path):
                with open(label_path, "r") as f:
                    label_data = f.read()
                with open(output_label_path, "w") as f:
                    f.write(label_data)

            # 증강 로그 저장
            logs.append({
                "original_image": img_path,
                "augmented_image": output_img_path,
                "augmentation": aug_name
            })
            generated_counts[aug_name] += 1

    # 각 증강 기법별 생성된 파일 수 출력
    for aug_name, count in generated_counts.items():
        print(f"{aug_name} 증강으로 생성된 이미지 수: {count}장")
    return logs

# 메인 함수
def main():
    augmentations = get_augmentations()  # 개별 증강
    pipeline = get_augmentation_pipeline()  # 파이프라인 증강

    # 개별 증강 적용
    logs = apply_augmentation(
        input_dir=ORIGINAL_PATH,
        output_dir=OUTPUT_PATH,
        augmentations=augmentations,
        proportion=AUGMENTATION_PROPORTION
    )

    # 증강 로그를 JSON 파일로 저장
    with open(LOG_FILE, "w") as log_file:
        json.dump(logs, log_file, indent=4)

    print(f"증강이 완료되었습니다! 증강 로그가 '{LOG_FILE}'에 저장되었습니다.")

if __name__ == "__main__":
    main()
