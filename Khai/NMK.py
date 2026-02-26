import cv2
import numpy as np
import sqlite3
import os
import sys
from PIL import Image, ImageFont, ImageDraw
import customtkinter as ctk
from tkinter import messagebox, simpledialog
from threading import Thread

# =========================================
# HÀM LẤY ĐƯỜNG DẪN RESOURCE (khi chạy exe)
# =========================================
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS  # khi build exe bằng PyInstaller
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# Thư mục dữ liệu
BASE_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))
DATASET_DIR = os.path.join(BASE_DIR, "dataSet")
RECOGNIZER_DIR = os.path.join(BASE_DIR, "recognizer")
DB_PATH = os.path.join(BASE_DIR, "data.db")
CASCADE_PATH = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")
MODEL_PATH = os.path.join(RECOGNIZER_DIR, "trainningData.yml")
FONT_PATH = os.path.join(BASE_DIR, "arial.ttf")


# =========================================
# 1. Thu thập dữ liệu 
# =========================================
def getdata():
    dialog = ctk.CTkInputDialog(text="Nhập ID:", title="Thu thập dữ liệu")
    user_id = dialog.get_input()
    if not user_id:
        return

    dialog = ctk.CTkInputDialog(text="Nhập tên:", title="Thu thập dữ liệu")
    name = dialog.get_input()
    if not name:
        return

    # ✅ Kiểm tra ID có tồn tại chưa
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute("SELECT * FROM people WHERE id=?", (user_id,))
    record = cursor.fetchone()
    if record is not None:
        conn.close()
        messagebox.showerror("Lỗi", f"❌ ID {user_id} đã tồn tại!\nVui lòng chọn ID khác.")
        return

    # Nếu chưa tồn tại → thêm mới
    conn.execute("INSERT INTO people (id, name) VALUES(?, ?)", (user_id, name))
    conn.commit()
    conn.close()

    # Thu thập dữ liệu bằng camera
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    cap = cv2.VideoCapture(0)
    sampleNum = 0

    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)

    window_name = "THU THAP DU LIEU"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    is_fullscreen = True

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            sampleNum += 1
            file_path = os.path.join(DATASET_DIR, f"User.{user_id}.{sampleNum}.jpg")
            cv2.imwrite(file_path, gray[y:y + h, x:x + w])

        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):   # Thoát
            break
        elif key == 27:  # ESC để thoát luôn
            break
        elif key == 0:   # F11 trên Windows gửi keycode = 0
            is_fullscreen = not is_fullscreen
            if is_fullscreen:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

        if sampleNum > 240:
            break

    cap.release()
    cv2.destroyAllWindows()
    messagebox.showinfo("Hoàn tất", f"✅ Đã thu thập dữ liệu cho ID {user_id} - {name}")



# =========================================
# 2. Huấn luyện dữ liệu 
# =========================================
def traindata(progressbar=None):
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    def getImageWithId(path):
        imagePaths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".jpg")]
        faces, ids = [], []
        for idx, imagePath in enumerate(imagePaths):
            faceImg = Image.open(imagePath).convert('L')
            faceNp = np.array(faceImg, 'uint8')
            filename = os.path.split(imagePath)[-1]
            id = int(filename.split('.')[1])
            faces.append(faceNp)
            ids.append(id)
            if progressbar:
                progressbar.set((idx+1)/len(imagePaths))
        return faces, ids

    if not os.path.exists(DATASET_DIR):
        messagebox.showwarning("Lỗi", "Chưa có dữ liệu để huấn luyện!")
        return

    faces, ids = getImageWithId(DATASET_DIR)
    if len(faces) == 0:
        messagebox.showwarning("Lỗi", "Không tìm thấy dữ liệu!")
        return

    recognizer.train(faces, np.array(ids))
    if not os.path.exists(RECOGNIZER_DIR):
        os.makedirs(RECOGNIZER_DIR)
    recognizer.save(MODEL_PATH)
    if progressbar:
        progressbar.set(1.0)
    messagebox.showinfo("Hoàn tất", "Đã huấn luyện và lưu model thành công!")


# =========================================
# 3. Nhận diện khuôn mặt
# =========================================
def recognizerdata():
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_PATH)

    def getProfile(id):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.execute("SELECT * FROM people WHERE id=?", (id,))
        profile = cursor.fetchone()
        conn.close()
        return profile

    pil_font = ImageFont.truetype(FONT_PATH, 28)

    def draw_text(frame, text, x, y, color=(0,255,0)):
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        draw.text((x,y), text, font=pil_font, fill=color)
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    cap = cv2.VideoCapture(0)
    threshold = 60

    window_name = "NHAN DIEN KHUON MAT"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    is_fullscreen = True

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)

        for (x, y, w, h) in faces:
            roi_gray = cv2.resize(gray[y:y + h, x:x + w], (200, 200))
            id, confidence = recognizer.predict(roi_gray)

            if confidence < threshold:
                profile = getProfile(id)
                if profile:
                    similarity = max(0, min(100, int(100 - confidence)))
                    text = f"ID:{profile[0]} - {profile[1]} ({similarity}%)"
                    frame = draw_text(frame, text, x, y + h + 30, (0, 255, 0))
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            else:
                frame = draw_text(frame, "Không xác định", x, y + h + 30, (255, 0, 0))
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # Q hoặc ESC thoát
            break
        elif key == 0:  # F11 toggle fullscreen
            is_fullscreen = not is_fullscreen
            if is_fullscreen:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

    cap.release()
    cv2.destroyAllWindows()


# =========================================
# 4. Xóa dữ liệu 
# =========================================
def deletedata():
    dialog = ctk.CTkInputDialog(text="Nhập ID cần xóa:", title="Xóa dữ liệu")
    user_id = dialog.get_input()
    if not user_id:
        return

    confirm = messagebox.askyesno("Xác nhận", f"⚠️ Bạn có chắc muốn xóa dữ liệu của ID {user_id}?")
    if not confirm:
        return

    # Xóa trong DB
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("DELETE FROM people WHERE id=?", (user_id,))
        conn.commit()
        conn.close()
    except Exception as e:
        messagebox.showerror("Lỗi", f"❌ Lỗi khi xóa trong DB: {e}")
        return

    # Xóa ảnh trong dataset
    deleted = 0
    if os.path.exists(DATASET_DIR):
        for f in os.listdir(DATASET_DIR):
            if f.startswith(f"User.{user_id}."):
                os.remove(os.path.join(DATASET_DIR, f))
                deleted += 1

    # Thông báo kết quả
    msg = f"✅ Đã xóa dữ liệu của ID {user_id} trong DB"
    if deleted > 0:
        msg += f"\n✅ Đã xóa {deleted} ảnh trong dataset"
    else:
        msg += "\n⚠️ Không tìm thấy ảnh trong dataset"
    messagebox.showinfo("Hoàn tất", msg)


# =========================================
# 5. Xóa toàn bộ dữ liệu
# =========================================
def resetdata():
    if not messagebox.askyesno("⚠️ Cảnh báo", "Bạn có chắc chắn muốn xóa TOÀN BỘ dữ liệu?"):
        return
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM people")
        conn.commit()
        conn.close()
    except Exception as e:
        messagebox.showerror("Lỗi", f"Lỗi DB: {e}")
    if os.path.exists(DATASET_DIR):
        for f in os.listdir(DATASET_DIR):
            os.remove(os.path.join(DATASET_DIR, f))
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)
    messagebox.showinfo("Hoàn tất", "Đã xóa toàn bộ dữ liệu!")


# =========================================
# Giao diện chính 
# =========================================
def run_gui():
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")

    root = ctk.CTk()
    root.title("Hệ thống nhận diện khuôn mặt")

    # Mở fullscreen mặc định
    root.attributes("-fullscreen", True)

    # Phím tắt fullscreen
    def toggle_fullscreen(event=None):
        state = root.attributes("-fullscreen")
        root.attributes("-fullscreen", not state)

    def end_fullscreen(event=None):
        root.attributes("-fullscreen", False)

    root.bind("<F11>", toggle_fullscreen)
    root.bind("<Escape>", end_fullscreen)

    title = ctk.CTkLabel(root, text="HỆ THỐNG NHẬN DIỆN KHUÔN MẶT", font=("Arial", 28, "bold"))
    title.pack(pady=40)

    ctk.CTkButton(root, text="📷 Thu thập dữ liệu", command=getdata, width=400, height=50).pack(pady=10)
    ctk.CTkButton(root, text="🛠 Huấn luyện dữ liệu", command=lambda: Thread(target=traindata, args=(progress,)).start(), width=400, height=50).pack(pady=10)
    ctk.CTkButton(root, text="👤 Nhận diện khuôn mặt", command=lambda: Thread(target=recognizerdata).start(), width=400, height=50).pack(pady=10)
    ctk.CTkButton(root, text="🗑 Xóa dữ liệu theo ID", command=deletedata, fg_color="orange", width=400, height=50).pack(pady=10)
    ctk.CTkButton(root, text="⚠️ Xóa toàn bộ dữ liệu", command=resetdata, fg_color="red", width=400, height=50).pack(pady=10)
    ctk.CTkButton(root, text="❌ Thoát", command=root.quit, fg_color="gray", width=400, height=50).pack(pady=30)

    # Progress bar
    progress = ctk.CTkProgressBar(root, width=400)
    progress.set(0)
    progress.pack(pady=20)

    root.mainloop()



if __name__ == "__main__":
    run_gui()
