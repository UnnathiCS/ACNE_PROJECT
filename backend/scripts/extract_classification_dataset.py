import os
import cv2
from tqdm import tqdm

# =====================================================

# PATHS

# =====================================================

# YOLO dataset root

YOLO_ROOT = "backend/dataset/detection"

# Train split

TRAIN_IMG_DIR = os.path.join(YOLO_ROOT, "train", "images")
TRAIN_LABEL_DIR = os.path.join(YOLO_ROOT, "train", "labels")

# Validation split

VALID_IMG_DIR = os.path.join(YOLO_ROOT, "valid", "images")
VALID_LABEL_DIR = os.path.join(YOLO_ROOT, "valid", "labels")

# Test split

TEST_IMG_DIR = os.path.join(YOLO_ROOT, "test", "images")
TEST_LABEL_DIR = os.path.join(YOLO_ROOT, "test", "labels")

# Output classification dataset
# NOTE: use 'datasets' (plural) to match training script expectations
OUTPUT_ROOT = "backend/datasets/classification"

# =====================================================

# CLASS MAP

# IMPORTANT:

# MUST MATCH data.yaml

# =====================================================

# Revised 5-class mapping:
# 0: blackhead
# 1: nodulocystic  (merge original 'cyst' and 'nodule')
# 2: papule
# 3: pustule
# 4: whitehead

CLASS_NAMES = {
    0: "blackhead",
    1: "nodulocystic",
    2: "papule",
    3: "pustule",
    4: "whitehead"
}

# =====================================================

# SETTINGS

# =====================================================

PADDING = 10

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png")

# =====================================================

# CREATE OUTPUT STRUCTURE

# =====================================================

for split in ["train", "valid", "test"]:

    for class_name in CLASS_NAMES.values():

        os.makedirs(
            os.path.join(
                OUTPUT_ROOT,
                split,
                class_name
            ),
            exist_ok=True
        )


# =====================================================

# CORE FUNCTION

# =====================================================

def process_split(
img_dir,
label_dir,
split_name
):

    print(f"\nProcessing {split_name} split...")

    # Defensive checks
    if not os.path.exists(img_dir):
        print(f"Image directory '{img_dir}' not found. Skipping {split_name}.")
        return

    if not os.path.exists(label_dir):
        print(f"Label directory '{label_dir}' not found. Skipping {split_name}.")
        return

    image_files = [
        f for f in os.listdir(img_dir)
        if f.lower().endswith(VALID_EXTENSIONS)
    ]

    # diagnostics
    stats = {
        'images_total': len(image_files),
        'missing_label': 0,
        'lines_total': 0,
        'invalid_lines': 0,
        'skipped_crop': 0,
        'saved': 0,
        'failed_writes': 0,
    }

    for img_name in tqdm(image_files):

        img_path = os.path.join(img_dir, img_name)

        label_name = img_name.rsplit(".", 1)[0] + ".txt"

        label_path = os.path.join(label_dir, label_name)

        if not os.path.exists(label_path):
            stats['missing_label'] += 1
            continue

        image = cv2.imread(img_path)

        if image is None:
            continue

        h, w, _ = image.shape

        with open(label_path, "r") as f:
            lines = f.readlines()

        for idx, line in enumerate(lines):

            stats['lines_total'] += 1

            parts = line.strip().split()

            if len(parts) != 5:
                stats['invalid_lines'] += 1
                continue

            try:
                class_id_f, xc, yc, bw, bh = map(float, parts)
            except ValueError:
                stats['invalid_lines'] += 1
                continue

            class_id = int(class_id_f)

            # Map original YOLO classes to the new 5-class taxonomy:
            # original dataset assumed:
            # 0: blackhead, 1: cyst, 2: nodule, 3: papule, 4: pustule, 5: whitehead
            # Merge 1 (cyst) and 2 (nodule) -> 1 (nodulocystic)
            if class_id == 0:
                mapped_id = 0
            elif class_id in (1, 2):
                mapped_id = 1
            elif class_id == 3:
                mapped_id = 2
            elif class_id == 4:
                mapped_id = 3
            elif class_id == 5:
                mapped_id = 4
            else:
                stats['invalid_lines'] += 1
                continue

            # YOLO normalized coords -> pixel coords
            x_center = xc * w
            y_center = yc * h

            box_w = bw * w
            box_h = bh * h

            x1 = int(x_center - box_w / 2)
            y1 = int(y_center - box_h / 2)

            x2 = int(x_center + box_w / 2)
            y2 = int(y_center + box_h / 2)

            # Add padding
            x1 -= PADDING
            y1 -= PADDING

            x2 += PADDING
            y2 += PADDING

            # Clamp to image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)

            x2 = min(w, x2)
            y2 = min(h, y2)

            # Validate crop
            if x2 <= x1 or y2 <= y1:
                stats['skipped_crop'] += 1
                continue

            crop = image[y1:y2, x1:x2]

            if crop.size == 0:
                stats['skipped_crop'] += 1
                continue

            class_name = CLASS_NAMES[mapped_id]

            save_name = (
                f"{img_name.rsplit('.',1)[0]}"
                f"_{idx}.jpg"
            )

            save_path = os.path.join(
                OUTPUT_ROOT,
                split_name,
                class_name,
                save_name
            )

            written = cv2.imwrite(save_path, crop)
            if written:
                stats['saved'] += 1
            else:
                stats['failed_writes'] += 1
                # Print a warning occasionally to avoid flooding
                if stats['failed_writes'] <= 10 or stats['failed_writes'] % 100 == 0:
                    print(f"Warning: failed to write {save_path}")

    print(f"Finished {split_name}")
    # Print diagnostic summary for this split
    print(f"Summary for {split_name}: images={stats['images_total']}, missing_label={stats['missing_label']}, lines={stats['lines_total']}, invalid_lines={stats['invalid_lines']}, skipped_crop={stats['skipped_crop']}, saved={stats['saved']}, failed_writes={stats['failed_writes']}")


# =====================================================

# RUN

# =====================================================

process_split(
TRAIN_IMG_DIR,
TRAIN_LABEL_DIR,
"train"
)

# only run if folders exist

if os.path.exists(VALID_IMG_DIR):


    process_split(
        VALID_IMG_DIR,
        VALID_LABEL_DIR,
        "valid"
    )


if os.path.exists(TEST_IMG_DIR):


    process_split(
        TEST_IMG_DIR,
        TEST_LABEL_DIR,
        "test"
    )


print("\n✅ Classification dataset generated successfully.")
