import os
import shutil
import random

random.seed(42)

root = "backend/dataset/classification"

train_dir = os.path.join(root, "train")
valid_dir = os.path.join(root, "valid")
test_dir = os.path.join(root, "test")

os.makedirs(valid_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

for cls in os.listdir(train_dir):

    cls_train = os.path.join(train_dir, cls)

    if not os.path.isdir(cls_train):
        continue

    images = [
        f for f in os.listdir(cls_train)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    random.shuffle(images)

    n = len(images)

    valid_count = int(0.1 * n)
    test_count = int(0.1 * n)

    cls_valid = os.path.join(valid_dir, cls)
    cls_test = os.path.join(test_dir, cls)

    os.makedirs(cls_valid, exist_ok=True)
    os.makedirs(cls_test, exist_ok=True)

    for img in images[:valid_count]:
        shutil.copy2(
            os.path.join(cls_train, img),
            os.path.join(cls_valid, img)
        )

    for img in images[valid_count:valid_count + test_count]:
        shutil.move(
            os.path.join(cls_train, img),
            os.path.join(cls_test, img)
        )

print("Done")