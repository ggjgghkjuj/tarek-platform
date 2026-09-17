# -*- coding: utf-8 -*-
"""
يرحّل بيانات faces_db (صور + بصمات) وattendance_log.db المحلية إلى Supabase.
يشغّل مرة واحدة على جهاز المطوّر بعد وضع مفاتيح SUPABASE_* في .streamlit/secrets.toml:
    python migrate_to_cloud.py
"""
import os
import json
import time
import sys

import numpy as np
import cv2
import requests

BASE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE, "faces_db")
LOG_DB = os.path.join(BASE, "attendance_log.db")
PHOTO_BUCKET = "student-photos"


def load_secrets():
    secrets = {}
    p = os.path.join(BASE, ".streamlit", "secrets.toml")
    if not os.path.exists(p):
        print("لا يوجد .streamlit/secrets.toml - ضع SUPABASE_URL وSUPABASE_SERVICE_KEY أولا")
        sys.exit(1)
    try:
        import tomllib
        with open(p, "rb") as f:
            secrets = tomllib.load(f)
    except Exception:
        print("تعذرت قراءة secrets.toml")
        sys.exit(1)
    return secrets


def load_models():
    det = cv2.FaceDetectorYN_create(
        os.path.join(BASE, "models", "face_detection_yunet_2023mar.onnx"), "", (320, 320))
    rec = cv2.FaceRecognizerSF_create(
        os.path.join(BASE, "models", "face_recognition_sface_2021dec.onnx"), "")
    return det, rec


def feature_from_file(path, det, rec):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    det.setInputSize((w, h))
    _, faces = det.detect(img)
    if faces is None or len(faces) == 0:
        return None
    aligned = rec.alignCrop(img, faces[0])
    return rec.feature(aligned)


def headers(key):
    return {
        "apikey": key,
        "Authorization": "Bearer " + key,
        "Content-Type": "application/json",
    }


def main():
    secrets = load_secrets()
    url = secrets.get("SUPABASE_URL")
    key = secrets.get("SUPABASE_SERVICE_KEY")
    if not url or not key:
        print("تأكد من وضع SUPABASE_URL وSUPABASE_SERVICE_KEY في secrets.toml")
        sys.exit(1)

    det, rec = load_models()
    image_exts = (".jpg", ".jpeg", ".png")

    # 1) الطلاب من جذر faces_db (بدون تصنيف) ثم من مجلدات الصفوف
    items = []
    for f in sorted(os.listdir(DB_PATH)):
        full = os.path.join(DB_PATH, f)
        if os.path.isfile(full) and f.lower().endswith(image_exts) and not f.startswith("."):
            items.append((os.path.splitext(f)[0], None, full))
    for d in sorted(os.listdir(DB_PATH)):
        if not os.path.isdir(os.path.join(DB_PATH, d)) or d.startswith("."):
            continue
        for f in sorted(os.listdir(os.path.join(DB_PATH, d))):
            full = os.path.join(DB_PATH, d, f)
            if os.path.isfile(full) and f.lower().endswith(image_exts):
                items.append((os.path.splitext(f)[0], d, full))

    print("عدد الطلاب المحليين:", len(items))
    added = 0
    for name, class_name, full in items:
        feat = feature_from_file(full, det, rec)
        if feat is None:
            print("  تخطي (لا يوجد وجه):", name)
            continue
        # تحقق: هل الطالب موجود سابقا في نفس الصف؟
        params = {"select": "name", "name": "eq." + name}
        if class_name:
            params["class_name"] = "eq." + class_name
        else:
            params["class_name"] = "is.null"
        try:
            r = requests.get(url + "/rest/v1/students", params=params, headers=headers(key), timeout=10)
            if r.status_code == 200 and r.json():
                print("  موجود مسبقا:", name, "- تخطي")
                continue
        except Exception:
            pass

        if class_name:
            try:
                requests.post(url + "/rest/v1/classes",
                              json={"name": class_name},
                              headers={**headers(key), "Prefer": "resolution=ignore-duplicates,return=minimal"},
                              timeout=10)
            except Exception:
                pass

        img_bytes = open(full, "rb").read()
        ext = os.path.splitext(full)[1].lower()
        object_path = str(int(time.time() * 1000)) + "_" + name.replace(" ", "_") + ext
        hdrs = headers(key)
        hdrs["Content-Type"] = "image/jpeg"
        try:
            r = requests.post(url + "/storage/v1/object/" + PHOTO_BUCKET + "/" + object_path,
                              data=img_bytes, headers=hdrs, timeout=30)
            if r.status_code not in (200, 201, 204):
                print("  فشل رفع الصورة:", name, r.status_code)
                continue
            payload = {
                "name": name,
                "class_name": class_name or None,
                "photo": object_path,
                "feature": json.dumps(np.asarray(feat, dtype=float).tolist()),
            }
            r2 = requests.post(url + "/rest/v1/students", json=payload,
                               headers={**headers(key), "Prefer": "return=representation"}, timeout=15)
            if r2.status_code in (200, 201):
                added += 1
                print("  تم الترحيل:", name, (class_name or "(بدون تصنيف)"))
            else:
                print("  فشل الإدراج:", name, r2.status_code)
        except Exception as e:
            print("  خطأ:", name, e)

    # 2) ترحيل سجل الحضور إن وُجد
    if os.path.exists(LOG_DB):
        import sqlite3
        conn = sqlite3.connect(LOG_DB)
        rows = conn.execute("SELECT student_name, time, date FROM attendance ORDER BY date, time").fetchall()
        conn.close()
        n = 0
        for name, t, date in rows:
            try:
                requests.post(url + "/rest/v1/attendance",
                              params={"on_conflict": "student_name,date"},
                              json={"student_name": name, "time": t, "date": date},
                              headers={**headers(key), "Prefer": "resolution=ignore-duplicates,return=minimal"},
                              timeout=10)
                n += 1
            except Exception:
                pass
        print("تم ترحيل سجلات الحضور:", n)

    print("\nاكتمل الترحيل - أضيف:", added, "طالبا")


if __name__ == "__main__":
    main()