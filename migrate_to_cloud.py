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
import uuid

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


def items_from_manifest():
    """يقرأ students.json الجديد (يدعم عدة صور ورقم قيد)."""
    p = os.path.join(DB_PATH, "students.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        m = json.load(f)
    photos_dir = os.path.join(DB_PATH, "photos")
    items = []
    for s in m.get("students", []):
        for ph in s.get("photos", []):
            items.append({
                "name": s.get("name"),
                "class": s.get("class"),
                "student_id": s.get("student_id"),
                "file": os.path.join(photos_dir, ph.get("file", "")),
                "feature": ph.get("feature"),
            })
    return items


def items_from_legacy():
    """النمط القديم: صور في الجذر أو مجلدات صفوف."""
    image_exts = (".jpg", ".jpeg", ".png")
    items = []
    for f in sorted(os.listdir(DB_PATH)):
        full = os.path.join(DB_PATH, f)
        if os.path.isfile(full) and f.lower().endswith(image_exts) and not f.startswith("."):
            items.append({"name": os.path.splitext(f)[0], "class": None,
                          "student_id": None, "file": full, "feature": None})
    for d in sorted(os.listdir(DB_PATH)):
        if not os.path.isdir(os.path.join(DB_PATH, d)) or d.startswith("."):
            continue
        for f in sorted(os.listdir(os.path.join(DB_PATH, d))):
            full = os.path.join(DB_PATH, d, f)
            if os.path.isfile(full) and f.lower().endswith(image_exts):
                items.append({"name": os.path.splitext(f)[0], "class": d,
                              "student_id": None, "file": full, "feature": None})
    return items


def main():
    secrets = load_secrets()
    url = secrets.get("SUPABASE_URL")
    key = secrets.get("SUPABASE_SERVICE_KEY")
    if not url or not key:
        print("تأكد من وضع SUPABASE_URL وSUPABASE_SERVICE_KEY في secrets.toml")
        sys.exit(1)

    det, rec = load_models()

    if not os.path.isdir(DB_PATH):
        print("لا يوجد مجلد faces_db")
        sys.exit(1)
    items = items_from_manifest()
    if items is None:
        items = items_from_legacy()

    print("عدد الصور المحلية:", len(items))
    added = 0
    skipped = set()
    for it in items:
        name = it["name"]
        class_name = it["class"]
        key_pair = (name, class_name or "")
        if key_pair in skipped:
            continue
        if it.get("feature"):
            feat = np.asarray(it["feature"], dtype=float)
        else:
            feat = feature_from_file(it["file"], det, rec)
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
                skipped.add(key_pair)
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

        img_bytes = open(it["file"], "rb").read()
        ext = os.path.splitext(it["file"])[1].lower()
        object_path = str(int(time.time() * 1000)) + "_" + uuid.uuid4().hex[:12] + ext
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
                "student_id": it.get("student_id") or None,
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