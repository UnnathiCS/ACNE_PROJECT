import os

root = "/Users/unnathics/Documents/SEM-6/Acne_project/Acne_code/acne_severity_analytics/backend/dataset/classification/train"

for cls in sorted(os.listdir(root)):

    cls_path = os.path.join(root, cls)

    if os.path.isdir(cls_path):

        count = len([
            f for f in os.listdir(cls_path)
            if f.endswith((".jpg", ".png", ".jpeg"))
        ])

        print(f"{cls}: {count}")