import cv2
import torch
from PIL import Image
import torchvision.transforms as T
from ultralytics import YOLO
from collections import deque, Counter

# =====================
# CRNN 관련: 모델 불러오기
# =====================
from train_and_infer_crnn_ctc import CRNN, CHARS, converter

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_W, IMG_H = 128, 32

def load_model(ckpt_path="checkpoints/crnn_best.pth"):
    """체크포인트에서 CRNN 모델 로드"""
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    chars = ckpt.get("chars", CHARS)
    model = CRNN(num_classes=1+len(chars)).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, chars

# =====================
# YOLO & OCR 준비
# =====================
yolo_model = YOLO("best.pt")   # YOLO 학습한 번호판 모델 경로
ocr_model, _ = load_model("checkpoints/crnn_best.pth")

infer_tf = T.Compose([
    T.Resize((IMG_H, IMG_W)),
    T.ToTensor(),
    T.Normalize([0.5], [0.5])
])

@torch.no_grad()
def predict_ocr(img_pil):
    """단일 번호판 이미지 OCR"""
    x = infer_tf(img_pil).unsqueeze(0).to(DEVICE)   # [1,1,H,W]
    logits, T_steps = ocr_model(x)
    log_probs = logits.log_softmax(2)
    pred = converter.decode_greedy(log_probs.cpu())[0]
    return pred

# =====================
# 보정(스무딩 & 투표) → 4자리 제한 제거
# =====================
def smooth_box(prev_box, new_box, alpha=0.7):
    """YOLO 박스 좌표 이동평균"""
    if prev_box is None:
        return new_box
    (px1, py1, px2, py2) = prev_box
    (nx1, ny1, nx2, ny2) = new_box
    x1 = int(alpha * px1 + (1 - alpha) * nx1)
    y1 = int(alpha * py1 + (1 - alpha) * ny1)
    x2 = int(alpha * px2 + (1 - alpha) * nx2)
    y2 = int(alpha * py2 + (1 - alpha) * ny2)
    return (x1, y1, x2, y2)

history = deque(maxlen=10)

def stabilize_text(new_text):
    """OCR 결과 안정화 (길이 제한 없음)"""
    import re
    new_text = re.sub(r'[^0-9]', '', new_text)  # 숫자만 남기기

    if len(new_text) == 0:  # 빈 문자열이면 무시
        return None

    history.append(new_text)
    most_common = Counter(history).most_common(1)[0][0]
    return most_common

# =====================
# 실시간 카메라
# =====================
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
prev_box = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = yolo_model(frame)
    best_box = None
    best_conf = 0

    # 가장 높은 confidence 박스만 선택
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = box.conf[0].item()
            if conf > best_conf:
                best_conf = conf
                best_box = (x1, y1, x2, y2)

    if best_box and best_conf > 0.5:
        # 박스 스무딩
        smooth = smooth_box(prev_box, best_box, alpha=0.7)
        prev_box = smooth
        (x1, y1, x2, y2) = smooth

        # =====================
        # 영상 표시용 박스 (적당히 shrink)
        # =====================
        shrink_x, shrink_y = 5, 3
        dx1 = max(0, x1 + shrink_x)
        dy1 = max(0, y1 + shrink_y)
        dx2 = min(frame.shape[1]-1, x2 - shrink_x)
        dy2 = min(frame.shape[0]-1, y2 - shrink_y)

        # =====================
        # OCR용 crop (더 타이트하게 shrink)
        # =====================
        ocr_shrink_x, ocr_shrink_y = 8, 5
        ox1 = max(0, x1 + ocr_shrink_x)
        oy1 = max(0, y1 + ocr_shrink_y)
        ox2 = min(frame.shape[1]-1, x2 - ocr_shrink_x)
        oy2 = min(frame.shape[0]-1, y2 - ocr_shrink_y)

        crop = frame[oy1:oy2, ox1:ox2]

        if crop.size > 0:
            pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))
            pred_text = predict_ocr(pil_img)
            final_text = stabilize_text(pred_text)

            if final_text:  # 숫자 길이에 상관없이 출력
                # 영상에는 보기 좋은 박스 + 텍스트 출력
                cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), (0, 255, 0), 2)
                cv2.putText(frame, final_text, (dx1, dy1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    cv2.imshow("YOLO + OCR", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
