import cv2
import face_recognition
import time
import numpy as np
import os
from utils import ensure_dirs, DATA_DIR, load_json, save_json, list_to_encoding, now_iso, today_str

ensure_dirs()
EMP_PATH = os.path.join(DATA_DIR, 'employees.json')
HIST_PATH = os.path.join(DATA_DIR, 'history.json')

def load_known():
    employees = load_json(EMP_PATH, [])
    ids = []
    names = []
    encs = []
    for e in employees:
        for enc_list in e.get('encodings', []):
            ids.append(e['id'])
            names.append(e['name'])
            encs.append(list_to_encoding(enc_list))
    if encs:
        encs = np.stack(encs)
    else:
        encs = np.array([])
    return employees, ids, names, encs

def read_history():
    return load_json(HIST_PATH, [])

def write_history(history):
    save_json(HIST_PATH, history)

def last_action_today(history, emp_id):
    today = today_str()
    for rec in reversed(history):
        if rec['employee_id'] == emp_id and rec.get('date') == today:
            return rec.get('type')
    return None

def main_loop():
    employees, ids_map, names_map, known_encs = load_known()
    if known_encs.size == 0:
        print('Chưa có nhân viên nào. Hãy chạy register.py trước.')
        return
    history = read_history()
    cap = cv2.VideoCapture(0)
    recent_time = {}  # emp_id -> last logged timestamp (seconds)
    print('Bắt đầu nhận dạng. Nhấn q để dừng.')
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        small = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb)
        encs = face_recognition.face_encodings(rgb, boxes)
        for face_loc, face_enc in zip(boxes, encs):
            if known_encs.size == 0:
                continue
            distances = face_recognition.face_distance(known_encs, face_enc)
            idx = np.argmin(distances)
            if distances[idx] < 0.5:
                emp_id = ids_map[idx]
                name = names_map[idx]
                now_s = time.time()
                last_t = recent_time.get(emp_id, 0)
                if now_s - last_t > 10:  # avoid duplicates within 10s
                    # determine in/out
                    last_type = last_action_today(history, emp_id)
                    next_type = 'in' if last_type != 'in' else 'out'
                    rec = {
                        'employee_id': emp_id,
                        'name': name,
                        'timestamp': now_iso(),
                        'type': next_type,
                        'date': today_str()
                    }
                    history.append(rec)
                    write_history(history)
                    recent_time[emp_id] = now_s
                    print(f'Logged {name} ({emp_id}) -> {next_type} at {rec["timestamp"]}')
                top, right, bottom, left = [v*2 for v in face_loc]  # scale back
                cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
                cv2.putText(frame, f'{name} ({emp_id})', (left, top-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            else:
                top, right, bottom, left = [v*2 for v in face_loc]
                cv2.rectangle(frame, (left, top), (right, bottom), (0,0,255), 2)
                cv2.putText(frame, 'Unknown', (left, top-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        cv2.imshow('Recognize (q to quit)', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main_loop()
