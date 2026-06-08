import os

for split in ["train","valid","test"]:
    print(f"\n{split}")

    path = f"backend/dataset/classification/{split}"

    for cls in sorted(os.listdir(path)):
        p = os.path.join(path, cls)

        if os.path.isdir(p):
            print(cls, len(os.listdir(p)))