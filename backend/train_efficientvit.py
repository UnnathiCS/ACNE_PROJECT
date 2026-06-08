import os
import math
import time
from argparse import ArgumentParser
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

try:
    import timm
except Exception:
    timm = None

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight


# EfficientViT classifier uses merged 'nodulocystic' class (cyst + nodule)
CLASS_NAMES = [
    "blackhead",
    "nodulocystic",
    "papule",
    "pustule",
    "whitehead",
]


class RemapDataset(torch.utils.data.Dataset):
    """Wrap an ImageFolder-like dataset and remap its integer labels using an index map.

    Kept as a top-level class so it can be pickled when DataLoader uses multiple workers.
    """
    def __init__(self, base_ds, idx_map):
        self.base = base_ds
        self.idx_map = idx_map

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x, y = self.base[i]
        return x, self.idx_map.get(y, y)


def make_dataloaders(root_dir, img_size=224, batch_size=32, num_workers=4, sample_per_class=0):
    # support both singular `backend/dataset/classification` and
    # plural `backend/datasets/classification` paths
    if os.path.isdir(root_dir):
        base_dir = root_dir
    else:
        # try alternate common path
        alt = root_dir.replace('/datasets/', '/dataset/') if '/datasets/' in root_dir else root_dir.replace('/dataset/', '/datasets/')
        base_dir = alt if os.path.isdir(alt) else root_dir

    train_dir = os.path.join(base_dir, "train")
    valid_dir = os.path.join(base_dir, "valid")
    test_dir = os.path.join(base_dir, "test")

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    # helper: count image files under a dir
    def _count_images(d):
        cnt = 0
        if not os.path.isdir(d):
            return 0
        for root, _, files in os.walk(d):
            for f in files:
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")):
                    cnt += 1
        return cnt

    train_count = _count_images(train_dir)
    if train_count == 0:
        raise FileNotFoundError(f"No training images found in {train_dir}")

    # If valid/test are missing or empty, or any class subfolder is empty, fall back to using train (warn user)
    def _has_per_class_images(d):
        # return True if directory exists and every CLASS_NAMES subfolder has >=1 image
        if not os.path.isdir(d):
            return False
        for cname in CLASS_NAMES:
            sub = os.path.join(d, cname)
            if not os.path.isdir(sub):
                return False
            found = 0
            for root, _, files in os.walk(sub):
                for f in files:
                    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")):
                        found += 1
                        break
                if found:
                    break
            if found == 0:
                return False
        return True

    valid_ok = _has_per_class_images(valid_dir)
    test_ok = _has_per_class_images(test_dir)

    if not valid_ok:
        print(f"Warning: validation folder {valid_dir} is missing or some class folders are empty. Falling back to train as validation (not recommended).")
        valid_dir = train_dir
    if not test_ok:
        print(f"Warning: test folder {test_dir} is missing or some class folders are empty. Falling back to train as test (not recommended).")
        test_dir = train_dir

    # Build raw ImageFolder datasets first
    raw_train = ImageFolder(train_dir, transform=train_tf)
    raw_valid = ImageFolder(valid_dir, transform=eval_tf)
    raw_test = ImageFolder(test_dir, transform=eval_tf)

    # Remap original detector classes into classifier classes.
    # detector may have 'cyst' and 'nodule' separately; merge them into 'nodulocystic'.
    # Build original index -> new index map
    orig_name_to_idx = raw_train.class_to_idx
    name_to_new_name = {}
    for name in orig_name_to_idx.keys():
        if name.lower() in ("cyst", "nodule"):
            name_to_new_name[name] = "nodulocystic"
        else:
            name_to_new_name[name] = name

    new_name_to_idx = {n: i for i, n in enumerate(CLASS_NAMES)}
    orig_idx_to_new_idx = {orig_idx: new_name_to_idx.get(name_to_new_name[orig_name], 0)
                           for orig_name, orig_idx in orig_name_to_idx.items()}

    # Optional: sample up to `sample_per_class` images per NEW class from the training set
    if sample_per_class and sample_per_class > 0:
        import random
        from collections import defaultdict
        grouped = defaultdict(list)
        for path, orig_idx in raw_train.samples:
            new_idx = orig_idx_to_new_idx.get(orig_idx, 0)
            grouped[new_idx].append((path, orig_idx))
        selected = []
        for cls in range(len(CLASS_NAMES)):
            items = grouped.get(cls, [])
            if len(items) > sample_per_class:
                selected.extend(random.sample(items, sample_per_class))
            else:
                selected.extend(items)
        # replace raw_train samples / imgs / targets (ImageFolder expects these fields)
        raw_train.samples = selected
        raw_train.imgs = selected
        raw_train.targets = [orig_idx for _, orig_idx in selected]

    train_ds = RemapDataset(raw_train, orig_idx_to_new_idx)
    valid_ds = RemapDataset(raw_valid, orig_idx_to_new_idx)
    test_ds = RemapDataset(raw_test, orig_idx_to_new_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, valid_loader, test_loader, train_ds


def build_model(num_classes, model_name="efficientvit_b0", pretrained=True):
    if timm is None:
        raise RuntimeError("timm is required for this script. Install with `pip install timm`.")

    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    return model


def compute_class_weights(dataset, device=None):
    # Build label list from dataset (supports RemapDataset and ImageFolder)
    labels = []
    # try attribute .base for remap wrapper
    if hasattr(dataset, 'base') and hasattr(dataset.base, 'samples'):
        base_samples = dataset.base.samples
        for _, y in base_samples:
            # map original -> new if remap dataset present
            if hasattr(dataset, 'idx_map'):
                labels.append(int(dataset.idx_map.get(y, y)))
            else:
                labels.append(int(y))
    elif hasattr(dataset, 'samples'):
        for _, y in dataset.samples:
            labels.append(int(y))
    else:
        # fallback: empty
        labels = []

    if len(labels) == 0:
        return torch.ones(len(CLASS_NAMES), dtype=torch.float32, device=device)

    import numpy as _np
    classes = _np.array(list(range(len(CLASS_NAMES))))
    weights = compute_class_weight('balanced', classes=classes, y=_np.array(labels))
    w_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    return w_tensor


def evaluate(model, loader, device):
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            outputs = model(images)
            _, p = torch.max(outputs, 1)
            preds.extend(p.cpu().tolist())
            trues.extend(targets.tolist())

    acc = accuracy_score(trues, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(trues, preds, labels=list(range(len(CLASS_NAMES))), zero_division=0)
    cm = confusion_matrix(trues, preds, labels=list(range(len(CLASS_NAMES))))
    return {
        "accuracy": acc,
        "precision_per_class": precision,
        "recall_per_class": recall,
        "f1_per_class": f1,
        "confusion_matrix": cm,
    }


def train(
    data_root,
    weights_path,
    model_name="efficientvit_b0",
    img_size=224,
    batch_size=32,
    head_epochs=4,
    finetune_epochs=10,
    epochs=None,
    lr=1e-4,
    weight_decay=1e-2,
    device=None,
    patience=5,
    num_workers=4,
    pretrained=True,
    sample_per_class=0,
):

    # Normalize device to torch.device
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    train_loader, valid_loader, test_loader, train_ds = make_dataloaders(
        data_root,
        img_size=img_size,
        batch_size=batch_size,
        num_workers=num_workers,
        sample_per_class=sample_per_class,
    )

    model = build_model(len(CLASS_NAMES), model_name=model_name, pretrained=pretrained)
    model.to(device)

    # Ensure weights dir exists
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)

    # Checkpoint path
    checkpoint_path = weights_path.replace(".pth", "_checkpoint.pth")
    start_epoch = 1

    # Prepare criterion with computed class weights
    class_weights = compute_class_weights(train_ds, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Determine total epochs (phased schedule if not overridden)
    if epochs is None:
        total_epochs = int(head_epochs) + int(finetune_epochs)
    else:
        total_epochs = int(epochs)

    # Freeze/unfreeze helpers
    def freeze_backbone_and_unfreeze_head(m):
        for p in m.parameters():
            p.requires_grad = False
        # try to find the classifier module
        cls = None
        try:
            cls = m.get_classifier()
        except Exception:
            cls = None
        if cls is not None:
            for p in cls.parameters():
                p.requires_grad = True
        else:
            # fallback: unfreeze params whose name contains common classifier keywords
            keywords = ("head", "classifier", "fc", "head.fc", "last", "ln", "norm")
            for name, p in m.named_parameters():
                if any(k in name.lower() for k in keywords):
                    p.requires_grad = True

    def unfreeze_all(m):
        for p in m.parameters():
            p.requires_grad = True

    # Apply initial freeze for head warmup
    freeze_backbone_and_unfreeze_head(model)

    # create optimizer only for params that require grad (head params initially)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)

    # mixed precision scaler (use CUDA availability)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # Training state
    best_metric = 0.0
    epochs_no_improve = 0
    unfreeze_performed = False

    # If checkpoint exists, load model + optimizer + scheduler + scaler + epoch + best_metric
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif "model_state" in checkpoint:
            model.load_state_dict(checkpoint["model_state"])
        try:
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            if "scaler_state_dict" in checkpoint and device.type == "cuda":
                try:
                    scaler.load_state_dict(checkpoint["scaler_state_dict"])
                except Exception:
                    pass
        except Exception:
            print("Warning: failed to fully restore optimizer/scheduler/scaler state from checkpoint.")
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_metric = float(checkpoint.get("best_metric", 0.0))
        print(f"Resuming from epoch {start_epoch}")

        # If training already finished, exit early to avoid silent no-op
        if start_epoch > total_epochs:
            print(
                "Training already completed (checkpoint epoch >= total_epochs). "
                "Delete the checkpoint to retrain from scratch."
            )
            return

    # Main training loop
    for epoch in range(start_epoch, total_epochs + 1):
        model.train()
        running_loss = 0.0
        all_preds = []
        all_trues = []
        t0 = time.time()

        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                outputs = model(images)
                # support models that may return (logits, feat_maps)
                if isinstance(outputs, (tuple, list)):
                    logits = outputs[0]
                else:
                    logits = outputs
                loss = criterion(logits, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(logits, 1)
            all_preds.extend(preds.cpu().tolist())
            all_trues.extend(targets.cpu().tolist())

        # Step scheduler
        try:
            scheduler.step()
        except Exception:
            pass

        epoch_loss = running_loss / len(train_loader.dataset)
        train_acc = accuracy_score(all_trues, all_preds)

        val_stats = evaluate(model, valid_loader, device)

        # use validation f1 mean as metric
        import numpy as _np
        val_f1_mean = float(_np.array(val_stats["f1_per_class"]).mean())

        print(f"Epoch {epoch}/{total_epochs} — train_loss={epoch_loss:.4f}, train_acc={train_acc:.4f}, val_acc={val_stats['accuracy']:.4f}, val_f1_mean={val_f1_mean:.4f}, time={(time.time()-t0):.1f}s")

        # early stopping logic based on val_f1_mean (update best_metric first)
        if val_f1_mean > best_metric:
            best_metric = val_f1_mean
            epochs_no_improve = 0
            # save best model weights (state_dict only)
            try:
                torch.save(model.state_dict(), weights_path)
                print(f"Saved best model (val_f1_mean={best_metric:.4f}) to {weights_path}")
            except Exception as e:
                print(f"Warning: failed to save best model: {e}")
        else:
            epochs_no_improve += 1

        # Save checkpoint every epoch (with updated best_metric)
        try:
            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_metric": best_metric,
                "scaler_state_dict": scaler.state_dict(),
            }
            torch.save(ckpt, checkpoint_path)
        except Exception as e:
            print(f"Warning: failed to save checkpoint: {e}")

        # If we have finished the head warmup, unfreeze for finetune
        if (not unfreeze_performed) and (epoch >= int(head_epochs)):
            print("Head warmup complete — unfreezing backbone for fine-tuning.")
            unfreeze_all(model)
            # recreate optimizer to include newly unfrozen params
            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)
            unfreeze_performed = True

        if epochs_no_improve >= patience:
            print(f"Early stopping after {epoch} epochs (no improvement for {patience} epochs)")
            break

    # final evaluation on test set
    print("Evaluating on test set...")
    test_stats = evaluate(model, test_loader, device)
    print(f"Test accuracy: {test_stats['accuracy']:.4f}")
    for i, cname in enumerate(CLASS_NAMES):
        print(f"Class {cname}: precision={test_stats['precision_per_class'][i]:.4f}, recall={test_stats['recall_per_class'][i]:.4f}, f1={test_stats['f1_per_class'][i]:.4f}")

    print("Confusion matrix:")
    print(test_stats["confusion_matrix"])


def parse_args():
    p = ArgumentParser()
    p.add_argument("--data_root", default="backend/dataset/classification", help="Root path to classification dataset (script will try backend/dataset then backend/datasets)")
    p.add_argument("--weights", default="backend/weights/efficientvit_acne_classifier.pth")
    p.add_argument("--model", default="efficientvit_b0")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--head_epochs", type=int, default=4, help="Epochs to train classifier head with backbone frozen")
    p.add_argument("--finetune_epochs", type=int, default=10, help="Epochs to fine-tune whole model after head warmup")
    p.add_argument("--epochs", type=int, default=None, help="(optional) total epochs; if provided overrides phased schedule")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=1e-2)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--sample_per_class", type=int, default=0, help="If >0 sample up to this many images per class from the training set to create a small balanced subset")
    p.add_argument("--no_pretrained", dest="pretrained", action="store_false")
    p.set_defaults(pretrained=True)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(os.path.dirname(args.weights), exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    # Accept either 'backend/dataset/classification' or 'backend/datasets/classification'
    data_root = args.data_root
    if not os.path.isdir(data_root):
        alt = data_root.replace("/dataset/", "/datasets/")
        if os.path.isdir(alt):
            print(f"Note: using alternate dataset path {alt}")
            data_root = alt
    train(
        data_root=data_root,
        weights_path=args.weights,
        model_name=args.model,
        img_size=args.img_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.wd,
        device=device,
        patience=args.patience,
        num_workers=args.num_workers,
        pretrained=args.pretrained,
        sample_per_class=args.sample_per_class,
    )
