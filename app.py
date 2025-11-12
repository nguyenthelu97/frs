# app.py (mới)
import os, io, time, json
import os
from flask_cors import CORS

USE_SSL = os.environ.get('FRS_USE_SSL','0') == '1'
SSL_CERT = os.environ.get('FRS_SSL_CERT','cert.pem')
SSL_KEY = os.environ.get('FRS_SSL_KEY','key.pem')
from datetime import datetime, date
from flask import Flask, request, redirect, url_for, send_file, render_template, flash, jsonify
import cv2
import numpy as np
import face_recognition
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "data")
THUMBS_DIR = os.path.join(DATA_DIR, "thumbs")
PROOFS_DIR = os.path.join(DATA_DIR, "proofs")
EMP_FILE = os.path.join(DATA_DIR, "employees.json")
HIST_FILE = os.path.join(DATA_DIR, "history.json")

os.makedirs(THUMBS_DIR, exist_ok=True)
os.makedirs(PROOFS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

app = Flask(__name__)
CORS(app)  # allow cross-origin requests from frontend (adjust origin in production)
app.config['SECRET_KEY'] = 'dev-secret'  # production: change

# helpers
def load_json(path, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return default

def save_json(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def encoding_to_list(enc):
    return enc.tolist()

def now_iso():
    return datetime.now().isoformat(timespec='seconds')

def today_str():
    return date.today().isoformat()

def save_image_jpg(path, bgr):
    cv2.imwrite(path, bgr)  # bgr image

# load knowns
def load_known():
    emps = load_json(EMP_FILE, [])
    ids = []
    names = []
    encs = []
    for e in emps:
        for enc in e.get('encodings', []):
            ids.append(e['id'])
            names.append(e['name'])
            encs.append(np.array(enc))
    if len(encs) > 0:
        encs = np.stack(encs)
    else:
        encs = np.array([])
    return emps, ids, names, encs

# index
@app.route('/live')
def live_page():
    return render_template('live.html')

@app.route('/')
def index():
    employees = load_json(EMP_FILE, [])
    return render_template('index.html', employees=employees)

# register
@app.route('/register', methods=['GET','POST'])
def register():
    if request.method == 'GET':
        return render_template('register.html')
    emp_id = request.form.get('emp_id','').strip()
    name = request.form.get('name','').strip()
    file = request.files.get('img')
    if not emp_id or not name or not file:
        flash('Missing fields', 'danger')
        return redirect(url_for('register'))
    emps = load_json(EMP_FILE, [])
    if any(e['id']==emp_id for e in emps):
        flash('Employee ID exists', 'warning')
        return redirect(url_for('register'))
    data = file.read()
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        flash('Cannot read image', 'danger')
        return redirect(url_for('register'))
    # detect, encode on scaled image for speed
    small = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb)
    if not boxes:
        flash('No face detected', 'warning')
        return redirect(url_for('register'))
    encs = face_recognition.face_encodings(rgb, boxes)
    if not encs:
        flash('Cannot compute encoding', 'danger')
        return redirect(url_for('register'))
    enc_list = [encoding_to_list(encs[0])]
    emps.append({'id': emp_id, 'name': name, 'encodings': enc_list})
    save_json(EMP_FILE, emps)
    # save thumb (crop from original image using first box scaled back)
    top,right,bottom,left = boxes[0]
    top, right, bottom, left = top*2, right*2, bottom*2, left*2
    h, w = img.shape[:2]
    top,left = max(0,top), max(0,left)
    bottom,right = min(h,bottom), min(w,right)
    crop = img[top:bottom, left:right]
    if crop.size != 0:
        thumb_path = os.path.join(THUMBS_DIR, f"{emp_id}.jpg")
        save_image_jpg(thumb_path, crop)
    flash('Registered', 'success')
    return redirect(url_for('index'))

# serve thumbs & proofs
@app.route('/thumbs/<path:filename>')
def thumbs(filename):
    return send_file(os.path.join(THUMBS_DIR, filename))

@app.route('/proofs/<path:filename>')
def proofs(filename):
    return send_file(os.path.join(PROOFS_DIR, filename))

# history view
@app.route('/history')
def history():
    hist = load_json(HIST_FILE, [])
    # reverse chronological
    hist = list(reversed(hist))
    date_filter = request.args.get('date','')
    emp_filter = request.args.get('emp_id','').strip()
    filtered = []
    for r in hist:
        if date_filter and r.get('date') != date_filter:
            continue
        if emp_filter and r.get('employee_id') != emp_filter:
            continue
        filtered.append(r)
    return render_template('history.html', history=filtered, date_filter=date_filter, emp_filter=emp_filter)

# export
@app.route('/export')
def export():
    """
    Export Excel.
    Query params supported:
      - month=YYYY-MM        -> export that month (all days in month)
      - from_date=YYYY-MM-DD & to_date=YYYY-MM-DD  -> export inclusive range
    If none provided, export all history (original behavior).
    """
    hist = load_json(HIST_FILE, [])

    # helper to parse record date (prefer 'date' field, else try timestamp)
    def rec_date_str(rec):
        d = rec.get('date')
        if d:
            return d  # assumed YYYY-MM-DD
        ts = rec.get('timestamp')
        if ts:
            try:
                # timestamp might be ISO: 2025-11-11T09:47:38
                return ts.split('T')[0]
            except Exception:
                pass
        return None

    # get query params
    month = request.args.get('month', '').strip()            # YYYY-MM
    from_date = request.args.get('from_date', '').strip()    # YYYY-MM-DD
    to_date = request.args.get('to_date', '').strip()        # YYYY-MM-DD

    # determine filter range
    start_date = None
    end_date = None

    try:
        if month:
            # parse month
            start_date = datetime.strptime(month, '%Y-%m').date()
            # compute last day of month: next_month_first - 1 day
            if start_date.month == 12:
                next_month = date(start_date.year + 1, 1, 1)
            else:
                next_month = date(start_date.year, start_date.month + 1, 1)
            end_date = next_month - timedelta(days=1)
        elif from_date or to_date:
            if from_date:
                start_date = datetime.strptime(from_date, '%Y-%m-%d').date()
            if to_date:
                end_date = datetime.strptime(to_date, '%Y-%m-%d').date()
            # if only one provided, set the other same day
            if start_date and not end_date:
                end_date = start_date
            if end_date and not start_date:
                start_date = end_date
    except ValueError as e:
        return f'Invalid date format: {e}', 400

    # filter history if needed
    if start_date and end_date:
        filtered = []
        for r in hist:
            rd = rec_date_str(r)
            if not rd:
                continue
            try:
                rd_date = datetime.strptime(rd, '%Y-%m-%d').date()
            except Exception:
                continue
            if start_date <= rd_date <= end_date:
                filtered.append(r)
        df = pd.DataFrame(filtered) if filtered else pd.DataFrame(columns=['employee_id','name','date','time','timestamp','type','proof','confidence'])
    else:
        # no filter -> export all
        df = pd.DataFrame(hist) if hist else pd.DataFrame(columns=['employee_id','name','date','time','timestamp','type','proof','confidence'])

    out = io.BytesIO()
    with pd.ExcelWriter(out, engine='openpyxl') as writer:
        # write main sheet
        df.to_excel(writer, index=False, sheet_name='history')
    out.seek(0)

    # create filename describing range
    if start_date and end_date:
        filename = f'frs_history_{start_date.isoformat()}_to_{end_date.isoformat()}.xlsx'
    else:
        filename = f'frs_history_{int(time.time())}.xlsx'

    return send_file(out, as_attachment=True, download_name=filename, mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

@app.route('/api/recognize', methods=['POST'])
def api_recognize():
    f = request.files.get('frame')
    if not f:
        return jsonify({'error': 'no frame'}), 400
    data = f.read()
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'error': 'cannot decode'}), 400

    scale = 0.5
    small = cv2.resize(img, (0,0), fx=scale, fy=scale)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb)
    encs = face_recognition.face_encodings(rgb, boxes)

    emps, ids_map, names_map, known_encs = load_known()

    results = []
    ts_iso = datetime.now().isoformat(timespec='seconds')
    date_str = date.today().isoformat()
    time_str = datetime.now().strftime('%H:%M:%S')

    for (top, right, bottom, left), enc in zip(boxes, encs):
        matched_name = 'Unknown'
        matched_id = None
        conf = None

        if known_encs.size != 0:
            dists = face_recognition.face_distance(known_encs, enc)
            idx = int(np.argmin(dists))
            conf = float(dists[idx])
            # điều chỉnh threshold nếu cần (0.5 là mặc định)
            if dists[idx] < 0.5:
                matched_id = ids_map[idx]
                matched_name = names_map[idx]

                # tránh duplicate log trong 10s
                now_s = time.time()
                last = recent_time.get(matched_id, 0)
                if now_s - last > 10:
                    # xác định in/out đơn giản: nếu lần cuối cùng trong ngày không phải 'in' -> in; nếu là 'in' -> out
                    hist = load_json(HIST_FILE, [])
                    last_type = None
                    for rec in reversed(hist):
                        if rec.get('employee_id') == matched_id and rec.get('date') == date_str:
                            last_type = rec.get('type')
                            break
                    next_type = 'in' if last_type != 'in' else 'out'

                    # lưu proof crop (scale lại về ảnh gốc)
                    top_o = int(top / scale); left_o = int(left / scale)
                    bottom_o = int(bottom / scale); right_o = int(right / scale)
                    h, w = img.shape[:2]
                    top_o, left_o = max(0, top_o), max(0, left_o)
                    bottom_o, right_o = min(h, bottom_o), min(w, right_o)
                    proof_filename = None
                    try:
                        proof_crop = img[top_o:bottom_o, left_o:right_o]
                        if proof_crop.size != 0:
                            proof_filename = f"{matched_id}_{int(now_s)}.jpg"
                            save_image_jpg(os.path.join(PROOFS_DIR, proof_filename), proof_crop)
                    except Exception:
                        proof_filename = None

                    # tạo record có fields rõ ràng: name, employee_id, date, time, timestamp, type, proof, confidence
                    record = {
                        'employee_id': matched_id,
                        'name': matched_name,
                        'date': date_str,
                        'time': time_str,
                        'timestamp': ts_iso,
                        'type': next_type,
                        'proof': proof_filename,
                        'confidence': conf
                    }
                    hist.append(record)
                    save_json(HIST_FILE, hist)
                    recent_time[matched_id] = now_s

        # scale box trở lại kích thước client (server dùng scale=0.5 khi detect)
        box = {
            'top': int(top / scale),
            'left': int(left / scale),
            'width': int((right - left) / scale),
            'height': int((bottom - top) / scale)
        }
        results.append({
            'name': matched_name,
            'employee_id': matched_id,
            'confidence': conf,
            'box': box
        })

    return jsonify({'results': results, 'timestamp': ts_iso})

@app.route('/api/employees')
def api_employees():
    emps = load_json(EMP_FILE, [])
    return jsonify(emps)

@app.route('/employee/<emp_id>/edit', methods=['GET', 'POST'])
def edit_employee(emp_id):
    emps = load_json(EMP_FILE, [])
    emp = next((e for e in emps if str(e.get('id')) == str(emp_id)), None)
    if emp is None:
        flash('Employee not found', 'danger')
        return redirect(url_for('index'))

    if request.method == 'GET':
        return render_template('register.html', edit=True, emp=emp)

    # POST: update name and possibly id + image
    new_id = request.form.get('emp_id', '').strip()
    name = request.form.get('name','').strip()
    file = request.files.get('img')

    if not new_id:
        flash('Employee ID is required', 'danger')
        return redirect(url_for('edit_employee', emp_id=emp_id))
    if not name:
        flash('Name is required', 'danger')
        return redirect(url_for('edit_employee', emp_id=emp_id))

    # if new_id differs, check uniqueness
    if str(new_id) != str(emp_id):
        if any(str(e.get('id')) == str(new_id) for e in emps):
            flash('New Employee ID already exists', 'warning')
            return redirect(url_for('edit_employee', emp_id=emp_id))

    # update name
    emp['name'] = name

    # handle new image if provided: recompute encoding + save thumb (replace encodings)
    if file:
        data = file.read()
        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            flash('Cannot read image', 'danger')
            return redirect(url_for('edit_employee', emp_id=emp_id))

        small = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb)
        if not boxes:
            flash('No face detected in new photo', 'warning')
            return redirect(url_for('edit_employee', emp_id=emp_id))
        encs = face_recognition.face_encodings(rgb, boxes)
        if not encs:
            flash('Cannot compute encoding', 'danger')
            return redirect(url_for('edit_employee', emp_id=emp_id))

        emp['encodings'] = [encoding_to_list(encs[0])]

        # save thumb (scale box back to original)
        top,right,bottom,left = boxes[0]
        top, right, bottom, left = top*2, right*2, bottom*2, left*2
        h, w = img.shape[:2]
        top,left = max(0,top), max(0,left)
        bottom,right = min(h,bottom), min(w,right)
        crop = img[top:bottom, left:right]
        if crop.size != 0:
            thumb_path = os.path.join(THUMBS_DIR, f"{emp_id}.jpg")
            save_image_jpg(thumb_path, crop)

    # If ID changed, update emp['id'], rename thumb file and optionally update history/proofs
    if str(new_id) != str(emp_id):
        old_id = str(emp_id)
        # update the emp's id
        emp['id'] = new_id

        # rename thumb file if exists
        old_thumb = os.path.join(THUMBS_DIR, f"{old_id}.jpg")
        new_thumb = os.path.join(THUMBS_DIR, f"{new_id}.jpg")
        try:
            if os.path.exists(old_thumb):
                os.replace(old_thumb, new_thumb)  # atomic where possible
        except Exception:
            # fallback: try copy + remove
            try:
                import shutil
                shutil.copy2(old_thumb, new_thumb)
                os.remove(old_thumb)
            except Exception:
                pass

        # update history entries' employee_id and rename proof files that start with old_id + '_'
        try:
            hist = load_json(HIST_FILE, [])
            changed = False
            for rec in hist:
                if str(rec.get('employee_id')) == old_id:
                    rec['employee_id'] = new_id
                    changed = True
            if changed:
                save_json(HIST_FILE, hist)

            # rename proof files that were named like "<old_id>_<ts>.jpg"
            for fname in os.listdir(PROOFS_DIR):
                if fname.startswith(f"{old_id}_"):
                    old_path = os.path.join(PROOFS_DIR, fname)
                    new_name = fname.replace(f"{old_id}_", f"{new_id}_", 1)
                    new_path = os.path.join(PROOFS_DIR, new_name)
                    try:
                        os.replace(old_path, new_path)
                        # Also update proof references in history if any (already updated by employee_id change, but proof filenames changed too)
                        # If your history stores proof filename strings, you may need to update them:
                        # loop again to fix filenames stored in history
                    except Exception:
                        try:
                            import shutil
                            shutil.copy2(old_path, new_path)
                            os.remove(old_path)
                        except Exception:
                            pass
            # Now update proof filenames inside history records (if stored)
            hist = load_json(HIST_FILE, [])
            changed = False
            for rec in hist:
                pf = rec.get('proof')
                if isinstance(pf, str) and pf.startswith(f"{old_id}_"):
                    rec['proof'] = pf.replace(f"{old_id}_", f"{new_id}_", 1)
                    changed = True
            if changed:
                save_json(HIST_FILE, hist)
        except Exception as e:
            # don't fail whole update on proof/history rename problems
            app.logger.exception("Error while updating history/proofs during id rename: %s", e)

    # finally persist employees
    save_json(EMP_FILE, emps)
    flash('Employee updated', 'success')
    # If ID changed, redirect to index (old edit URL no longer valid)
    return redirect(url_for('index'))


@app.route('/employee/<emp_id>/delete', methods=['POST'])
def delete_employee(emp_id):
    emps = load_json(EMP_FILE, [])
    new_emps = [e for e in emps if str(e.get('id')) != str(emp_id)]
    if len(new_emps) == len(emps):
        flash('Employee not found', 'warning')
        return redirect(url_for('index'))

    # delete thumb file if exists
    thumb_path = os.path.join(THUMBS_DIR, f"{emp_id}.jpg")
    try:
        if os.path.exists(thumb_path):
            os.remove(thumb_path)
    except Exception:
        pass

    save_json(EMP_FILE, new_emps)
    flash('Employee deleted', 'success')
    return redirect(url_for('index'))

@app.route('/api/employee/<emp_id>', methods=['DELETE'])
def api_delete_employee(emp_id):
    emps = load_json(EMP_FILE, [])
    new_emps = [e for e in emps if str(e.get('id')) != str(emp_id)]
    if len(new_emps) == len(emps):
        return jsonify({'error': 'not found'}), 404
    thumb_path = os.path.join(THUMBS_DIR, f"{emp_id}.jpg")
    try:
        if os.path.exists(thumb_path):
            os.remove(thumb_path)
    except Exception:
        pass
    save_json(EMP_FILE, new_emps)
    return jsonify({'ok': True})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', '0') == '1'
    app.run(host='0.0.0.0', port=port, debug=debug)
