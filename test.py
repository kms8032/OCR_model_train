import cv2
import torch
import editdistance
from ultralytics import YOLO
from train_and_infer_crnn_ctc import CRNN, CHARS, converter
from PIL import Image
import torchvision.transforms as T

# ============================
# 1. CRNN 모델 불러오기
# ============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_W, IMG_H = 128, 32

def load_crnn(ckpt_path="checkpoints/crnn_best.pth"):
    """체크포인트에서 CRNN 모델 불러오기"""
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    chars = ckpt.get("chars", CHARS)
    model = CRNN(num_classes=1+len(chars)).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, chars

def run_crnn(model, chars, img_crop):
    """번호판 잘라낸 이미지 → CRNN OCR 실행"""
    transform = T.Compose([
        T.Resize((IMG_H, IMG_W)),
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,))
    ])
    img = Image.fromarray(cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)).convert("L")
    img = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        preds = model(img)  # [T, B, C] 형태
        _, preds_idx = preds.max(2)
        preds_idx = preds_idx.squeeze(1).cpu().numpy()
        pred_text = converter.decode(preds_idx, chars)
    return pred_text

# ============================
# 2. YOLO 모델 불러오기
# ============================
yolo_model = YOLO("best.pt")
crnn_model, chars = load_crnn("checkpoints/crnn_best.pth")

# ============================
# 3. 평가용 함수
# ============================
gt_texts, pred_texts = [], []

def calc_accuracy(gt_texts, pred_texts):
    correct = sum([gt == pred for gt, pred in zip(gt_texts, pred_texts)])
    return correct / len(gt_texts) if gt_texts else 0

def calc_cer(gt_texts, pred_texts):
    total_dist, total_len = 0, 0
    for gt, pred in zip(gt_texts, pred_texts):
        dist = editdistance.eval(gt, pred)
        total_dist += dist
        total_len += len(gt)
    return total_dist / total_len if total_len > 0 else 0

# ============================
# 4. 카메라 실행
# ============================
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO 탐지
    results = yolo_model(frame)
    boxes = results[0].boxes.xywh.cpu().numpy()

    if len(boxes) > 0:
        # 첫 번째 박스 사용
        box = boxes[0]
        x, y, w, h = int(box[0]-box[2]/2), int(box[1]-box[3]/2), int(box[2]), int(box[3])
        crop = frame[y:y+h, x:x+w]

        # CRNN OCR 실행
        pred_text = run_crnn(crnn_model, chars, crop)
        print(f"[예측 결과] {pred_text}")

        # 정답 입력
        gt_text = input("정답 번호판 입력 (Enter=건너뛰기): ").strip()
        if gt_text:
            gt_texts.append(gt_text)
            pred_texts.append(pred_text)

    # 종료 키
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# ============================
# 5. 최종 성능 출력
# ============================
print("\n=== 최종 성능 평가 ===")
print(f"총 입력 수: {len(gt_texts)}")
print(f"OCR Accuracy: {calc_accuracy(gt_texts, pred_texts):.2f}")
print(f"CER (Character Error Rate): {calc_cer(gt_texts, pred_texts):.2f}")
