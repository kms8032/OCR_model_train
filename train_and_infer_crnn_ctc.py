import os, csv, random, time
from typing import List, Tuple
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import mlflow
import mlflow.pytorch

# =========================
# 기본 설정
# =========================
DATA_DIR = "dataset"
IMG_DIR  = os.path.join(DATA_DIR, "images")
CSV_PATH = os.path.join(DATA_DIR, "images/labels.csv")

IMG_W, IMG_H = 128, 32
BATCH_SIZE   = 64
EPOCHS       = 40
LR           = 1e-3
WEIGHT_DECAY = 1e-4
SEED         = 42
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

CHARS = "0123456789"

random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
# GPU 연산 속도 저하로 인해 수정 ( 최고 성능, 정확도 속도 중요 시 주석 처리 )
# torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True # False -> True

# =========================
# 라벨 인코더/디코더 (CTC)
# =========================
class LabelConverter:
    def __init__(self, chars: str):
        self.chars = chars
        self.blank = 0
        self.char2idx = {c: i+1 for i, c in enumerate(chars)}
        self.idx2char = {i+1: c for i, c in enumerate(chars)}

    def encode(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        targets, lengths = [], []
        for s in texts:
            ids = [self.char2idx[c] for c in s]
            targets.extend(ids)
            lengths.append(len(ids))
        return torch.tensor(targets, dtype=torch.long), torch.tensor(lengths, dtype=torch.long)

    def decode_greedy(self, log_probs: torch.Tensor) -> List[str]:
        max_idx = log_probs.argmax(dim=2).cpu().numpy().T
        results = []
        for seq in max_idx:
            prev = self.blank
            out = []
            for idx in seq:
                idx = int(idx)
                if idx != self.blank and idx != prev:
                    out.append(self.idx2char.get(idx, ""))
                prev = idx
            results.append("".join(out))
        return results

converter = LabelConverter(CHARS)

# =========================
# Dataset & Transform
# =========================
class PlateOCRDataset(Dataset):
    def __init__(self, csv_path: str, img_dir: str, transform=None):
        self.items = []
        missing = 0
        with open(csv_path, newline="", encoding="utf-8") as f:
            first_line = f.readline()
            f.seek(0)
            if ("filename" in first_line.lower()) and ("label" in first_line.lower()):
                rdr = csv.DictReader(f)
                for row in rdr:
                    img_path = os.path.join(img_dir, row["filename"])
                    if os.path.exists(img_path):
                        self.items.append((img_path, row["label"]))
                    else:
                        missing += 1
            else:
                rdr = csv.reader(f)
                for r in rdr:
                    if len(r) >= 2:
                        img_path = os.path.join(img_dir, r[0])
                        if os.path.exists(img_path):
                            self.items.append((img_path, r[1]))
                        else:
                            missing += 1
        if missing > 0:
            print(f"[Warning] {missing} file(s) listed but not found in images/. Skipped.")
        self.transform = transform

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        img_path, label = self.items[i]
        img = Image.open(img_path).convert("L")
        if self.transform:
            img = self.transform(img)
        return img, label

# 데이터 증강 및 전처리
train_tf = T.Compose([
    T.Resize((IMG_H, IMG_W)),
    # 1. 원근 왜곡 (p=0.4, 40% 확률로 적용)
    T.RandomPerspective(distortion_scale=0.3, p=0.4, fill=0), 
    T.ColorJitter(brightness=0.4, contrast=0.4),
    T.RandomAffine(degrees=2, shear=2, translate=(0.02, 0.02), scale=(0.95, 1.05)),
    # 2. 가우시안 블러 (p=0.3, 30% 확률로 적용)
    T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.3),
    T.ToTensor(),
    T.Normalize([0.5], [0.5]),
    T.RandomErasing(p=0.25, scale=(0.02, 0.06), ratio=(0.3, 3.3), value='random', inplace=False)
])
val_tf = T.Compose([
    T.Resize((IMG_H, IMG_W)),
    T.ToTensor(),
    T.Normalize([0.5], [0.5])
])

# =========================
# CSV Split (train / val / test)
# =========================
def split_csv(csv_path, out_train, out_val, out_test, ratio_train=0.8, ratio_val=0.1):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        first_line = f.readline()
        f.seek(0)
        if ("filename" in first_line.lower()) and ("label" in first_line.lower()):
            rdr = csv.DictReader(f)
            for r in rdr:
                rows.append({"filename": r["filename"], "label": r["label"]})
        else:
            print("[Info] No header detected in labels.csv. Assuming 'filename,label' format.")
            rdr = csv.reader(f)
            for r in rdr:
                if len(r) >= 2:
                    rows.append({"filename": r[0], "label": r[1]})

    random.shuffle(rows)
    n_train = int(len(rows) * ratio_train)
    n_val = int(len(rows) * ratio_val)
    train_rows = rows[:n_train]
    val_rows = rows[n_train:n_train+n_val]
    test_rows = rows[n_train+n_val:]

    for path, rows_ in [(out_train, train_rows), (out_val, val_rows), (out_test, test_rows)]:
        with open(path, "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=["filename", "label"])
            wr.writeheader()
            wr.writerows(rows_)
    print(f"[Split] Train: {len(train_rows)} | Val: {len(val_rows)} | Test: {len(test_rows)}")

def collate_fn(batch):
    imgs, labels = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    tgts, tgt_lens = converter.encode(list(labels))
    return imgs, list(labels), tgts, tgt_lens

# =========================
# CRNN 모델 (4자리용 최적화 구조)
# =========================
class CRNN(nn.Module):
    def __init__(self, num_classes: int, hidden=256, rnn_layers=2, rnn_dropout=0.3):
        super().__init__()

        def conv_bn_relu(i_c, o_c, k=3, s=1, p=1, use_bn=True, drop_p=0.0):
            layers = [nn.Conv2d(i_c, o_c, k, s, p, bias=not use_bn)]
            if use_bn:
                layers.append(nn.BatchNorm2d(o_c))
            layers.append(nn.ReLU(inplace=True))
            if drop_p > 0:
                layers.append(nn.Dropout2d(drop_p))
            return nn.Sequential(*layers)

        # T_steps ≈ 7, 4자리 인식에 충분
        self.cnn = nn.Sequential(
            conv_bn_relu(1,   64, 3,1,1), nn.MaxPool2d(2,2),
            conv_bn_relu(64, 128, 3,1,1), nn.MaxPool2d(2,2),
            conv_bn_relu(128,256,3,1,1),
            conv_bn_relu(256,256,3,1,1), nn.MaxPool2d(2,2),
            conv_bn_relu(256,512,3,1,1), conv_bn_relu(512,512,3,1,1), nn.MaxPool2d(2,2),
            nn.Conv2d(512,512,(2,1),1,0), nn.ReLU(inplace=True)
        )

        self.rnn = nn.LSTM(
            input_size=512, hidden_size=hidden,
            num_layers=rnn_layers, bidirectional=True,
            batch_first=False, dropout=rnn_dropout if rnn_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden*2, num_classes)

    def forward(self, x):
        feat = self.cnn(x)
        feat = feat.squeeze(2)
        feat = feat.permute(2,0,1)
        seq, _ = self.rnn(feat)
        logits = self.fc(seq)
        return logits, logits.size(0)

# =========================
# Metrics
# =========================
def cer(preds: List[str], gts: List[str]) -> float:
    def edit(a,b):
        dp = [[i+j if i*j==0 else 0 for j in range(len(b)+1)] for i in range(len(a)+1)]
        for i in range(1,len(a)+1):
            for j in range(1,len(b)+1):
                dp[i][j] = min(
                    dp[i-1][j]+1, dp[i][j-1]+1,
                    dp[i-1][j-1] + (a[i-1]!=b[j-1])
                )
        return dp[-1][-1]
    tot, dist = 0, 0
    for p,g in zip(preds,gts):
        dist += edit(p,g); tot += len(g)
    return dist / max(tot,1)

def seq_accuracy(preds: List[str], gts: List[str]) -> float:
    correct = sum(p == g for p, g in zip(preds, gts))
    return correct / max(len(gts), 1)

# =========================
# Train 함수
# =========================
def train():
    os.makedirs("checkpoints", exist_ok=True)
    tr_csv = os.path.join(DATA_DIR, "train.csv")
    va_csv = os.path.join(DATA_DIR, "val.csv")
    te_csv = os.path.join(DATA_DIR, "test.csv")
    split_csv(CSV_PATH, tr_csv, va_csv, te_csv)

    train_ds = PlateOCRDataset(tr_csv, IMG_DIR, transform=train_tf)
    val_ds   = PlateOCRDataset(va_csv, IMG_DIR, transform=val_tf)
    test_ds  = PlateOCRDataset(te_csv, IMG_DIR, transform=val_tf)

    pin = (DEVICE == "cuda")
    pin = (DEVICE == "cuda")
    train_loader = DataLoader(
        train_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=8,      
        pin_memory=pin, 
        collate_fn=collate_fn, 
        drop_last=False,
        persistent_workers=True  
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=4,       
        pin_memory=pin, 
        collate_fn=collate_fn, 
        drop_last=False,
        persistent_workers=True 
    )
    test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=pin,
    collate_fn=collate_fn,
    drop_last=False,
    persistent_workers=True
    )

    num_classes = 1 + len(CHARS)
    model = CRNN(num_classes).to(DEVICE)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE=="cuda"))

    mlflow.start_run()
    mlflow.log_params({
        "epochs": EPOCHS, "lr": LR, "batch_size": BATCH_SIZE,
        "img_size": f"{IMG_W}x{IMG_H}", "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR", "chars": CHARS
    })

    best_val_cer = float("inf")
    for epoch in range(1, EPOCHS+1):
        model.train()
        tr_loss = 0
        for imgs, labels, targets, target_lengths in train_loader:
            imgs, targets, target_lengths = imgs.to(DEVICE), targets.to(DEVICE), target_lengths.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(DEVICE=="cuda")):
                logits, T_steps = model(imgs)
                log_probs = logits.log_softmax(2)
                input_lengths = torch.full((logits.size(1),), T_steps, dtype=torch.long, device=DEVICE)
                loss = criterion(log_probs, targets, input_lengths, target_lengths)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer); scaler.update()
            tr_loss += loss.item()
        scheduler.step()

        model.eval()
        vl_loss, all_preds, all_gts = 0, [], []
        with torch.no_grad():
            for imgs, labels, targets, target_lengths in val_loader:
                imgs, targets, target_lengths = imgs.to(DEVICE), targets.to(DEVICE), target_lengths.to(DEVICE)
                logits, T_steps = model(imgs)
                log_probs = logits.log_softmax(2)
                input_lengths = torch.full((logits.size(1),), T_steps, dtype=torch.long, device=DEVICE)
                loss = criterion(log_probs, targets, input_lengths, target_lengths)
                vl_loss += loss.item()
                decoded = [p[:4] for p in converter.decode_greedy(log_probs.cpu())]  # 4자리 고정
                all_preds.extend(decoded); all_gts.extend(list(labels))

        val_cer = cer(all_preds, all_gts)
        val_acc = seq_accuracy(all_preds, all_gts)
        print(f"[Epoch {epoch:02d}] Train {tr_loss/len(train_loader):.4f} | Val {vl_loss/len(val_loader):.4f} | CER {val_cer:.4f} | SeqAcc {val_acc:.4f}")

        mlflow.log_metric("train_loss", tr_loss/len(train_loader), step=epoch)
        mlflow.log_metric("val_loss", vl_loss/len(val_loader), step=epoch)
        mlflow.log_metric("val_cer", val_cer, step=epoch)
        mlflow.log_metric("val_seq_acc", val_acc, step=epoch)

        if val_cer < best_val_cer:
            best_val_cer = val_cer
            torch.save({"model": model.state_dict(), "chars": CHARS}, "checkpoints/crnn_best.pth")
            mlflow.log_artifact("checkpoints/crnn_best.pth")
            print("  -> saved best checkpoint")

    # ======================
    # Test 평가
    # ======================
    print("\n[Testing best model...]")
    best_ckpt = torch.load("checkpoints/crnn_best.pth", map_location=DEVICE)
    model.load_state_dict(best_ckpt["model"])
    model.eval()

    te_loss, preds, gts = 0, [], []
    with torch.no_grad():
        for imgs, labels, targets, target_lengths in test_loader:
            imgs, targets, target_lengths = imgs.to(DEVICE), targets.to(DEVICE), target_lengths.to(DEVICE)
            logits, T_steps = model(imgs)
            log_probs = logits.log_softmax(2)
            input_lengths = torch.full((logits.size(1),), T_steps, dtype=torch.long, device=DEVICE)
            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            te_loss += loss.item()
            decoded = [p[:4] for p in converter.decode_greedy(log_probs.cpu())]
            preds.extend(decoded); gts.extend(list(labels))

    test_cer = cer(preds, gts)
    test_sar = seq_accuracy(preds, gts)
    test_loss = te_loss / len(test_loader)

    print(f"[Test] Loss {test_loss:.4f} | CER {test_cer:.4f} | SAR {test_sar:.4f}")
    mlflow.log_metric("test_loss", test_loss)
    mlflow.log_metric("test_cer", test_cer)
    mlflow.log_metric("test_sar", test_sar)
    mlflow.end_run()

    print("Done training + testing.")

if __name__ == "__main__":
    train()
