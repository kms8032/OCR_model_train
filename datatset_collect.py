import tkinter as tk
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
import torch
import os
import csv

# YOLOv8 모델 로드
model = YOLO("./best.pt")


model.model.float()

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# 저장 경로
save_dir = "./test"
os.makedirs(save_dir, exist_ok=True)

# CSV 라벨 파일
csv_path = os.path.join(save_dir, "labels.csv")
if not os.path.exists(csv_path):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label"])

root = tk.Tk()
label = tk.Label(root)
label.pack()

TARGET_SIZE = (128, 32)

current_frame = None
current_boxes = []
input_plate_number = None
image_num = 0
max_images = 5
captured_count = 0
exit_flag = False
need_new_number = True   # 새로운 번호를 입력해야 하는 상태


def ask_new_number():
    """새로운 번호 입력 (q 입력 시 종료)"""
    global input_plate_number, captured_count, need_new_number, exit_flag
    user_input = input("번호입력 (종료하려면 q 입력): ")

    if user_input.strip().lower() == 'q':  # q 입력하면 종료
        print("Q가 입력되어 프로그램을 종료합니다.")
        exit_flag = True
        cap.release()
        root.quit()
        return

    input_plate_number = user_input.strip()
    captured_count = 0
    need_new_number = False
    print(f"새 번호 [{input_plate_number}] 시작")


def save_frame(event=None):
    global image_num, captured_count, current_frame, current_boxes
    global input_plate_number, exit_flag, need_new_number

    if exit_flag or need_new_number:
        return

    if current_frame is None or not current_boxes:
        print("저장할 번호판이 없습니다.")
        return

    for (x1, y1, x2, y2) in current_boxes:
        cropped_frame = current_frame[y1:y2, x1:x2]
        resized_frame = cv2.resize(cropped_frame, TARGET_SIZE)

        image_number = str(image_num).zfill(5)
        filename = f"{input_plate_number}_{image_number}.png"
        image_path = os.path.join(save_dir, filename)
        cv2.imwrite(image_path, resized_frame)

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([filename, input_plate_number])

        print(f"{image_path} 저장 완료")
        image_num += 1
        captured_count += 1

        if captured_count >= max_images:
            print(f"{input_plate_number} 번호로 {max_images}장 저장 완료. 다시 번호를 입력하세요.")
            need_new_number = True   # 다음 루프에서 새 번호 입력 유도


def show_frame():
    global current_frame, current_boxes, exit_flag, need_new_number

    if exit_flag:
        return

    if need_new_number:
        ask_new_number()
        if exit_flag:  # q 입력 시 종료
            return

    ret, frame = cap.read()
    if not ret:
        return

    frame = cv2.resize(frame, (640, 480))

    results = model(frame)
    current_boxes = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            conf = box.conf[0].cpu().numpy()
            if conf > 0.5:
                x1, y1, x2, y2 = xyxy
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                current_boxes.append((x1, y1, x2, y2))

    current_frame = frame.copy()

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    imgtk = ImageTk.PhotoImage(image=img)
    label.imgtk = imgtk
    label.configure(image=imgtk)

    root.after(1, show_frame)


# 마우스 좌클릭 이벤트 → 번호판 저장
root.bind("<Button-1>", save_frame)

show_frame()
root.mainloop()

cap.release()
root.destroy()
