import os, csv, random
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
CSV_PATH = os.path.join(DATA_DIR, "labels.csv")

IMG_W, IMG_H = 128, 32
BATCH_SIZE   = 64
EPOCHS       = 70
LR           = 1e-4
SEED         = 42
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

CHARS = "0123456789"

random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

# =========================
# 라벨 인코더/디코더
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
                if idx != self.blank and idx != prev:
                    out.append(self.idx2char.get(int(idx), ""))
                prev = int(idx)
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
            rdr = csv.DictReader(f)
            for row in rdr:
                img_path = os.path.join(img_dir, row["filename"])
                if os.path.exists(img_path):
                    self.items.append((img_path, row["label"]))
                else:
                    missing += 1
        if missing > 0:
            print(f"[Warning] {missing} file(s) listed in {os.path.basename(csv_path)} not found in images/. Skipped.")
        self.transform = transform

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        img_path, label = self.items[i]
        img = Image.open(img_path).convert("L")
        if self.transform:
            img = self.transform(img)
        return img, label

train_tf = T.Compose([
    T.Resize((IMG_H, IMG_W)),
    T.ColorJitter(brightness=0.3, contrast=0.3),  # 증강(원본 파일은 건드리지 않음)
    T.ToTensor(),
    T.Normalize([0.5], [0.5])
])
val_tf = T.Compose([
    T.Resize((IMG_H, IMG_W)),
    T.ToTensor(),
    T.Normalize([0.5], [0.5])
])

# =========================
# 유틸: 파일명 0패딩
# =========================
def pad_stem_if_numeric(filename: str, width: int = 4) -> str:
    name, ext = os.path.splitext(filename)
    # 이름이 전부 숫자일 때만 패딩 (예: '1.png' -> '0001.png')
    if name.isdigit():
        return f"{int(name):0{width}d}{ext}"
    return filename

# =========================
# CSV Split (헤더 자동감지 + 0패딩 + 라벨단위 분할)
# =========================
def split_csv(csv_path, out_train, out_val, ratio=0.8, group_by_label: bool = True, pad_width: int = 4):
    rows = []

    # 1) 헤더 유무 자동 판단
    with open(csv_path, newline="", encoding="utf-8") as f:
        first_line = f.readline()
        f.seek(0)
        if ("filename" in first_line.lower()) and ("label" in first_line.lower()):
            # 정상 헤더
            rdr = csv.DictReader(f)
            for r in rdr:
                rows.append({"filename": r["filename"], "label": r["label"]})
        else:
            # 헤더 없음 → reader로 읽어서 강제 매핑
            print("[Info] No header detected in labels.csv. Assuming 'filename,label' format.")
            rdr = csv.reader(f)
            for r in rdr:
                if len(r) < 2: 
                    continue
                rows.append({"filename": r[0], "label": r[1]})

    # 2) 파일명 0패딩 적용
    for r in rows:
        r["filename"] = pad_stem_if_numeric(r["filename"], width=pad_width)

    # 3) 이미지 존재하는 것만 유지(여기서 1차 필터)
    filtered = []
    missing = 0
    for r in rows:
        img_path = os.path.join(IMG_DIR, r["filename"])
        if os.path.exists(img_path):
            filtered.append(r)
        else:
            missing += 1
    if missing > 0:
        print(f"[Warning] {missing} file(s) listed in labels.csv not found in images/. They will be dropped.")

    if not filtered:
        raise RuntimeError("No valid rows after filtering. Check your labels.csv and images/ folder.")

    # 4) 분할: (옵션) 같은 라벨 단위로 분할 → 데이터 누수 방지
    if group_by_label:
        grouped = {}
        for r in filtered:
            grouped.setdefault(r["label"], []).append(r)
        labels = list(grouped.keys())
        random.shuffle(labels)
        n_train_labels = int(len(labels) * ratio)
        train_labels = set(labels[:n_train_labels])

        train_rows, val_rows = [], []
        for lab, samples in grouped.items():
            (train_rows if lab in train_labels else val_rows).extend(samples)
    else:
        random.shuffle(filtered)
        n = int(len(filtered) * ratio)
        train_rows, val_rows = filtered[:n], filtered[n:]

    # 5) 저장
    with open(out_train, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=["filename","label"])
        wr.writeheader(); wr.writerows(train_rows)
    with open(out_val, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=["filename","label"])
        wr.writeheader(); wr.writerows(val_rows)

    print(f"[Split] Train: {len(train_rows)} | Val: {len(val_rows)} (group_by_label={group_by_label})")

def collate_fn(batch):
    imgs, labels = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    tgts, tgt_lens = converter.encode(list(labels))
    return imgs, list(labels), tgts, tgt_lens

# =========================
# CRNN 모델
# =========================
class CRNN(nn.Module):
    def __init__(self, num_classes: int, hidden=256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(64,128,3,1,1), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(128,256,3,1,1), nn.ReLU(),
            nn.Conv2d(256,256,3,1,1), nn.ReLU(), nn.MaxPool2d((2,2)),
            nn.Conv2d(256,512,3,1,1), nn.ReLU(), nn.BatchNorm2d(512),
            nn.Conv2d(512,512,3,1,1), nn.ReLU(), nn.MaxPool2d((2,2)),
            nn.Conv2d(512,512,2,1,0), nn.ReLU(),
        )
        self.rnn = nn.LSTM(input_size=512, hidden_size=hidden, num_layers=2,
                           bidirectional=True, batch_first=False)
        self.fc = nn.Linear(hidden*2, num_classes)

    def forward(self, x):
        feat = self.cnn(x)
        feat = feat.squeeze(2)
        feat = feat.permute(2,0,1)
        seq, _ = self.rnn(feat)
        logits = self.fc(seq)
        return logits, logits.size(0)

# =========================
# CER 계산
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

# =========================
# Train 함수 with MLflow
# =========================
def train():
    os.makedirs("checkpoints", exist_ok=True)
    tr_csv = os.path.join(DATA_DIR, "train.csv")
    va_csv = os.path.join(DATA_DIR, "val.csv")

    # 헤더 자동감지 + 0패딩 + 라벨단위 분할 + 존재 확인
    split_csv(CSV_PATH, tr_csv, va_csv, ratio=0.8, group_by_label=True, pad_width=4)

    train_ds = PlateOCRDataset(tr_csv, IMG_DIR, transform=train_tf)
    val_ds   = PlateOCRDataset(va_csv, IMG_DIR, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True, collate_fn=collate_fn)

    num_classes = 1 + len(CHARS)
    model = CRNN(num_classes=num_classes).to(DEVICE)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE=="cuda"))

    # ===== MLflow Logging Start =====
    mlflow.start_run()
    mlflow.log_param("img_size", (IMG_W, IMG_H))
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("learning_rate", LR)
    mlflow.log_param("optimizer", "Adam")
    mlflow.log_param("device", DEVICE)
    mlflow.log_param("num_classes", num_classes)

    best_val = float("inf")
    for epoch in range(1, EPOCHS+1):
        # ---- Train ----
        model.train()
        tr_loss = 0.0
        for imgs, labels, targets, target_lengths in train_loader:
            imgs = imgs.to(DEVICE)
            targets = targets.to(DEVICE)
            target_lengths = target_lengths.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(DEVICE=="cuda")):
                logits, T_steps = model(imgs)
                log_probs = logits.log_softmax(2)
                input_lengths = torch.full((logits.size(1),), T_steps, dtype=torch.long, device=DEVICE)
                loss = criterion(log_probs, targets, input_lengths, target_lengths)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            tr_loss += loss.item()

        # ---- Val ----
        model.eval()
        vl_loss = 0.0
        all_preds, all_gts = [], []
        with torch.no_grad():
            for imgs, labels, targets, target_lengths in val_loader:
                imgs = imgs.to(DEVICE)
                targets = targets.to(DEVICE)
                target_lengths = target_lengths.to(DEVICE)

                logits, T_steps = model(imgs)
                log_probs = logits.log_softmax(2)
                input_lengths = torch.full((logits.size(1),), T_steps, dtype=torch.long, device=DEVICE)
                loss = criterion(log_probs, targets, input_lengths, target_lengths)
                vl_loss += loss.item()

                preds = converter.decode_greedy(log_probs.cpu())
                all_preds.extend(preds)
                all_gts.extend(list(labels))

        val_cer = cer(all_preds, all_gts)

        # 콘솔 출력
        print(f"[Epoch {epoch:02d}] train {tr_loss/len(train_loader):.4f} | "
              f"val {vl_loss/len(val_loader):.4f} | CER {val_cer:.4f}")

        # ===== MLflow Log =====
        mlflow.log_metric("train_loss", tr_loss/len(train_loader), step=epoch)
        mlflow.log_metric("val_loss", vl_loss/len(val_loader), step=epoch)
        mlflow.log_metric("val_cer", val_cer, step=epoch)

        # 샘플 예측 저장 (첫 번째 배치만 기록)
        if epoch % 10 == 0:
            sample_preds = "\n".join([f"{p} | {g}" for p,g in zip(all_preds[:100], all_gts[:100])])
            mlflow.log_text(sample_preds, f"sample_predictions_{epoch}.txt")

        # 체크포인트 저장
        if vl_loss < best_val:
            best_val = vl_loss
            ckpt = {
                "model": model.state_dict(),
                "chars": CHARS,
                "img_size": (IMG_W, IMG_H)
            }
            path = "checkpoints/crnn_best.pth"
            torch.save(ckpt, path)
            print(f"  -> saved {path}")
            mlflow.pytorch.log_model(model, name="crnn_model")

    mlflow.end_run()
    print("Done training.")

if __name__ == "__main__":
    train()
