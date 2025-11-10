import cv2
import face_recognition
import time
from utils import ensure_dirs, DATA_DIR, load_json, save_json, encoding_to_list, save_thumbnail
import os

ensure_dirs()
EMP_PATH = os.path.join(DATA_DIR, 'employees.json')

def register_employee():
    employees = load_json(EMP_PATH, [])
    emp_id = input('Employee ID (unique): ').strip()
    if any(e['id'] == emp_id for e in employees):
        print('ID đã tồn tại.')
        return
    name = input('Name: ').strip()
    samples = 5
    encodings = []
    cap = cv2.VideoCapture(0)
    print('Nhấn q để hủy. Bắt đầu chụp %d mẫu, giữ khuôn mặt trước camera...' % samples)
    collected = 0

    # keep last good full-size frame and face box for thumbnail
    last_face_box = None
    last_frame = None

    while collected < samples:
        ret, frame = cap.read()
        if not ret:
            break
        small = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb)
        if boxes:
            encs = face_recognition.face_encodings(rgb, boxes)
            if encs:
                encodings.append(encoding_to_list(encs[0]))
                # store latest full-res frame and scaled-up box
                last_frame = frame.copy()
                last_face_box = boxes[0]  # in small-scale coords
                collected += 1
                print(f'Collected {collected}/{samples}')
                time.sleep(0.6)
        cv2.imshow('Register (press q to cancel)', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    if not encodings:
        print('Không thu được encoding nào.')
        return

    # Save thumbnail (crop from last_frame using last_face_box, scale coords back)
    if last_frame is not None and last_face_box is not None:
        top, right, bottom, left = [v*2 for v in last_face_box]  # scale back to full res
        # clamp
        h, w = last_frame.shape[:2]
        top, left = max(0, top), max(0, left)
        bottom, right = min(h, bottom), min(w, right)
        face_crop = last_frame[top:bottom, left:right]
        try:
            thumb_path = save_thumbnail(emp_id, face_crop)
            if thumb_path:
                print('Saved thumbnail:', thumb_path)
        except Exception as ex:
            print('Lỗi khi lưu thumbnail:', ex)

    employees.append({
        'id': emp_id,
        'name': name,
        'encodings': encodings
    })
    save_json(EMP_PATH, employees)
    print('Đã lưu nhân viên:', emp_id, name)

if __name__ == '__main__':
    register_employee()
