import streamlit as st
import cv2
import os
import io
import base64
import json
import hmac
import time
import uuid
import shutil
import threading
import numpy as np
from datetime import datetime, timedelta
import requests
import streamlit.components.v1 as components

# ============================================================
# إعدادات المنصة
# ============================================================
DB_PATH = "faces_db"
LOG_DB = "attendance_log.db"
TODAY = datetime.now().strftime("%Y-%m-%d")

os.makedirs(DB_PATH, exist_ok=True)

PLATFORM_NAME = "Tarek Platform"
PLATFORM_NAME_AR = "منصة طارق"

# ============================================================
# طبقة التخزين: Supabase (سحابي) + مخزن محلي (students.json + photos)
# كل طالب قد يملك عدة صور (بصمات) + رقم قيد + صف.
# ============================================================
SUPABASE_URL = None
SUPABASE_KEY = None
PHOTO_BUCKET = "student-photos"
MANIFEST_PATH = os.path.join(DB_PATH, "students.json")
PHOTOS_DIR = os.path.join(DB_PATH, "photos")
_manifest_lock = threading.Lock()


def get_secret(name, default=None):
    """يقرأ سرّاً من Secrets أو من متغيرات البيئة (للنشر السحابي مثل Render)."""
    try:
        val = st.secrets.get(name)
    except Exception:
        val = None
    if val is None:
        val = os.environ.get(name, default)
    return val


def init_storage():
    """يقرأ مفاتيح Supabase من Secrets/البيئة؛ إن غابت يعمل المخزن المحلي."""
    global SUPABASE_URL, SUPABASE_KEY
    SUPABASE_URL = get_secret("SUPABASE_URL")
    SUPABASE_KEY = get_secret("SUPABASE_SERVICE_KEY")


def cloud_enabled():
    return bool(SUPABASE_URL and SUPABASE_KEY)


def _rest_headers():
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
    }


def public_photo_url(object_path):
    if not (cloud_enabled() and object_path):
        return None
    return f"{SUPABASE_URL}/storage/v1/object/public/{PHOTO_BUCKET}/{object_path}"


# ------------------------------------------------------------------
# المخزن المحلي
# ------------------------------------------------------------------
def _empty_manifest():
    return {"classes": [], "students": []}


def _load_manifest():
    try:
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            m = json.load(f)
        m.setdefault("classes", [])
        m.setdefault("students", [])
        return m
    except Exception:
        return _empty_manifest()


def _log_mem(tag):
    """يطبع استهلاك الذاكرة في سجل Render لتشخيص OOM (Linux فقط)."""
    try:
        with open("/proc/self/status") as f:
            vals = {}
            for line in f:
                if line.startswith(("VmRSS:", "VmHWM:")):
                    k, v = line.split(":", 1)
                    vals[k] = v.strip()
        if vals:
            print(f"[MEM] {tag}: RSS={vals.get('VmRSS', '?')} PEAK={vals.get('VmHWM', '?')}", flush=True)
    except Exception:
        pass


_log_mem("startup")


def _save_manifest(m):
    os.makedirs(DB_PATH, exist_ok=True)
    tmp = MANIFEST_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=2)
    os.replace(tmp, MANIFEST_PATH)


def _save_photo_bytes(image_bytes, ext):
    os.makedirs(PHOTOS_DIR, exist_ok=True)
    fname = f"{uuid.uuid4().hex[:12]}{ext or '.jpg'}"
    with open(os.path.join(PHOTOS_DIR, fname), "wb") as f:
        f.write(image_bytes)
    return fname


def _delete_photo_file(fname):
    try:
        os.remove(os.path.join(PHOTOS_DIR, os.path.basename(fname)))
    except Exception:
        pass


def _find_local_student(m, name, class_name):
    for s in m["students"]:
        if s.get("name") == name and (s.get("class") or None) == (class_name or None):
            return s
    return None


def _extract_features(files):
    """files: [(bytes, ext)] -> (features, error)."""
    feats = []
    for data, ext in files:
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None, "cv2_fail"
        feat = face_feature_from_image(img)
        if feat is None:
            return None, "no_face"
        feats.append(feat)
    return feats, None


def local_add_student(name, class_name, student_id, files):
    feats, err = _extract_features(files)
    if err:
        return False, err
    with _manifest_lock:
        m = _load_manifest()
        if _find_local_student(m, name, class_name) is not None:
            return False, "duplicate"
        photos = []
        for (data, ext), feat in zip(files, feats):
            photos.append({
                "file": _save_photo_bytes(data, ext),
                "feature": np.asarray(feat, dtype=float).tolist(),
            })
        m["students"].append({
            "name": name,
            "class": class_name or None,
            "student_id": (student_id or "").strip() or None,
            "photos": photos,
        })
        if class_name and class_name not in m["classes"]:
            m["classes"].append(class_name)
        _save_manifest(m)
    return True, "ok"


def local_add_photos(name, class_name, files):
    feats, err = _extract_features(files)
    if err:
        return False, err
    with _manifest_lock:
        m = _load_manifest()
        s = _find_local_student(m, name, class_name)
        if s is None:
            return False, "missing"
        for (data, ext), feat in zip(files, feats):
            s.setdefault("photos", []).append({
                "file": _save_photo_bytes(data, ext),
                "feature": np.asarray(feat, dtype=float).tolist(),
            })
        _save_manifest(m)
    return True, "ok"


def local_update_student(old_name, old_class, new_name, new_class, student_id):
    with _manifest_lock:
        m = _load_manifest()
        s = _find_local_student(m, old_name, old_class)
        if s is None:
            return False, "missing"
        s["name"] = new_name
        s["class"] = new_class or None
        s["student_id"] = (student_id or "").strip() or None
        if new_class and new_class not in m["classes"]:
            m["classes"].append(new_class)
        _save_manifest(m)
    return True, "ok"


def local_delete_student(name, class_name):
    with _manifest_lock:
        m = _load_manifest()
        s = _find_local_student(m, name, class_name)
        if s is None:
            return False
        for p in s.get("photos", []):
            _delete_photo_file(p.get("file"))
        m["students"] = [x for x in m["students"] if x is not s]
        _save_manifest(m)
    return True


def local_delete_photo(name, class_name, file_name):
    with _manifest_lock:
        m = _load_manifest()
        s = _find_local_student(m, name, class_name)
        if s is None:
            return False
        s["photos"] = [p for p in s.get("photos", []) if p.get("file") != file_name]
        _save_manifest(m)
    _delete_photo_file(file_name)
    return True


# ------------------------------------------------------------------
# المخزن السحابي (Supabase)
# ------------------------------------------------------------------
def cloud_add_student(name, class_name, student_id, files):
    feats, err = _extract_features(files)
    if err:
        return False, err
    if class_name:
        try:
            requests.post(
                f"{SUPABASE_URL}/rest/v1/classes",
                json={"name": class_name},
                headers={**_rest_headers(), "Prefer": "resolution=ignore-duplicates,return=minimal"},
                timeout=10,
            )
        except Exception:
            pass
    for (data, ext), feat in zip(files, feats):
        object_path = f"{uuid.uuid4().hex[:12]}{ext or '.jpg'}"
        hdrs = _rest_headers()
        mime = "image/jpeg"
        if ext and ext.lower() in (".png",):
            mime = "image/png"
        elif ext and ext.lower() in (".webp",):
            mime = "image/webp"
        hdrs["Content-Type"] = mime
        try:
            r = requests.post(
                f"{SUPABASE_URL}/storage/v1/object/{PHOTO_BUCKET}/{object_path}",
                data=data, headers=hdrs, timeout=30,
            )
            if r.status_code not in (200, 201, 204):
                return False, "upload"
        except Exception:
            return False, "upload"
        payload = {
            "name": name,
            "class_name": class_name or None,
            "student_id": (student_id or "").strip() or None,
            "photo": object_path,
            "feature": json.dumps(np.asarray(feat, dtype=float).tolist()),
        }
        try:
            r2 = requests.post(
                f"{SUPABASE_URL}/rest/v1/students",
                json=payload,
                headers={**_rest_headers(), "Prefer": "return=representation"},
                timeout=15,
            )
            if r2.status_code not in (200, 201):
                if "student_id" in payload:
                    payload.pop("student_id", None)
                    r2 = requests.post(
                        f"{SUPABASE_URL}/rest/v1/students",
                        json=payload,
                        headers={**_rest_headers(), "Prefer": "return=representation"},
                        timeout=15,
                    )
                if r2.status_code not in (200, 201):
                    return False, "insert"
        except Exception:
            return False, "insert"
    return True, "ok"


def cloud_add_photos(name, class_name, files):
    return cloud_add_student(name, class_name, None, files)


def cloud_update_student(old_name, old_class, new_name, new_class, student_id):
    params = {"name": f"eq.{old_name}"}
    if old_class:
        params["class_name"] = f"eq.{old_class}"
    payload = {
        "name": new_name,
        "class_name": new_class or None,
        "student_id": (student_id or "").strip() or None,
    }
    try:
        r = requests.patch(f"{SUPABASE_URL}/rest/v1/students", params=params, json=payload,
                           headers=_rest_headers(), timeout=15)
        if r.status_code not in (200, 204) and "student_id" in payload:
            payload.pop("student_id", None)
            r = requests.patch(f"{SUPABASE_URL}/rest/v1/students", params=params, json=payload,
                               headers=_rest_headers(), timeout=15)
        return r.status_code in (200, 204), "ok"
    except Exception:
        return False, "error"


def cloud_delete_student(name, class_name):
    params = {"name": f"eq.{name}"}
    if class_name:
        params["class_name"] = f"eq.{class_name}"
    try:
        r = requests.delete(f"{SUPABASE_URL}/rest/v1/students", params=params,
                            headers=_rest_headers(), timeout=15)
        return r.status_code in (200, 204)
    except Exception:
        return False


# ------------------------------------------------------------------
# واجهة موحّدة (تختار السحابة أو المحلي تلقائياً)
# ------------------------------------------------------------------
def add_student(name, class_name, student_id, files):
    if cloud_enabled():
        return cloud_add_student(name, class_name, student_id, files)
    return local_add_student(name, class_name, student_id, files)


def add_student_photos(name, class_name, files):
    if cloud_enabled():
        return cloud_add_photos(name, class_name, files)
    return local_add_photos(name, class_name, files)


def update_student(old_name, old_class, new_name, new_class, student_id):
    if cloud_enabled():
        return cloud_update_student(old_name, old_class, new_name, new_class, student_id)
    return local_update_student(old_name, old_class, new_name, new_class, student_id)


def delete_student(name, class_name):
    if cloud_enabled():
        return cloud_delete_student(name, class_name)
    return local_delete_student(name, class_name)


def delete_student_photo(name, class_name, file_name):
    if cloud_enabled():
        try:
            requests.delete(f"{SUPABASE_URL}/rest/v1/students",
                            params={"name": f"eq.{name}", "photo": f"eq.{file_name}"},
                            headers=_rest_headers(), timeout=15)
            return True
        except Exception:
            return False
    return local_delete_photo(name, class_name, file_name)


def add_class(class_name):
    if cloud_enabled():
        try:
            requests.post(f"{SUPABASE_URL}/rest/v1/classes", json={"name": class_name},
                          headers={**_rest_headers(), "Prefer": "resolution=ignore-duplicates,return=minimal"},
                          timeout=10)
            return True
        except Exception:
            return False
    with _manifest_lock:
        m = _load_manifest()
        if class_name not in m["classes"]:
            m["classes"].append(class_name)
        _save_manifest(m)
    return True


def delete_class(class_name):
    if cloud_enabled():
        try:
            requests.delete(f"{SUPABASE_URL}/rest/v1/students", params={"class_name": f"eq.{class_name}"},
                            headers=_rest_headers(), timeout=15)
            requests.delete(f"{SUPABASE_URL}/rest/v1/classes", params={"name": f"eq.{class_name}"},
                            headers=_rest_headers(), timeout=15)
            return True
        except Exception:
            return False
    with _manifest_lock:
        m = _load_manifest()
        m["classes"] = [c for c in m["classes"] if c != class_name]
        keep = []
        for s in m["students"]:
            if (s.get("class") or None) == class_name:
                for p in s.get("photos", []):
                    _delete_photo_file(p.get("file"))
            else:
                keep.append(s)
        m["students"] = keep
        _save_manifest(m)
    return True


def photo_path_for_local(file_name):
    return os.path.join(PHOTOS_DIR, os.path.basename(file_name or ""))

# ============================================================
# نماذج OpenCV (YuNet للكشف + SFace للتعرف) — بدون حاجة لـ dlib
# ============================================================
MODELS_DIR = "models"
FACE_DETECTOR = None
FACE_RECOGNIZER = None
FACE_MATCH_THRESHOLD = 0.50
# نصف قطر الربط بين إطارين متتاليين (بكسل) لاعتبار الوجه نفس الشخص
FACE_TRACK_RADIUS = 160.0


def ensure_models():
    """يحمّل نماذج الكشف والتعرف مرة واحدة فقط لكل عملية."""
    global FACE_DETECTOR, FACE_RECOGNIZER
    if FACE_DETECTOR is None:
        det_path = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")
        rec_path = os.path.join(MODELS_DIR, "face_recognition_sface_2021dec.onnx")
        FACE_DETECTOR = cv2.FaceDetectorYN_create(det_path, "", (320, 320))
        FACE_RECOGNIZER = cv2.FaceRecognizerSF_create(rec_path, "")
    return FACE_DETECTOR, FACE_RECOGNIZER


def face_feature_from_image(img):
    """يعيد متجه التعرف لأول وجه في الصورة، أو None إن لم يوجد وجه."""
    detector, recognizer = ensure_models()
    h, w = img.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(img)
    if faces is None or len(faces) == 0:
        return None
    aligned = recognizer.alignCrop(img, faces[0])
    return recognizer.feature(aligned)

# ============================================================
# دعم اللغات
# ============================================================
LANGUAGES = {
    "ar": {
        "code": "ar",
        "flag": "🇯🇴",
        "label": "العربية",
        "title": "منصة طارق",
        "subtitle": "نظام الحضور والغياب الذكي بتعرف الوجوه",
        "info": "نظام سحابي للتعرف على الوجوه وكشف الاحتيال، مع تشعيب الطلاب بصفوف وتسجيل الحضور والغياب على ملف Excel.",
        "sidebar": "لوحة التحكم",
        "tab1": "إضافة طالب",
        "tab2": "إدارة الصفوف",
        "class_label": "الصف / القسم",
        "untracked": "بدون تصنيف",
        "name_label": "اسم الطالب",
        "photo_label": "صورة الطالب",
        "add_btn": "إضافة الطالب",
        "add_success": "تمت إضافة {} في {} بنجاح!",
        "add_duplicate": "يوجد طالب بهذا الاسم بالفعل. استخدم اسماً مختلفاً.",
        "add_warning": "الرجاء إدخال الاسم واختيار صورة.",
        "no_face_found": "لم يُعثر على وجه واضح في الصورة — جرّب صورة أوضح ومباشرة.",
        "add_error": "تعذّرت الإضافة — تحقّق من الاتصال بالسحابة وحاول مجدداً.",
        "new_class_label": "اسم الصف الجديد",
        "create_btn": "إنشاء الصف",
        "create_success": "تم إنشاء الصف: {}",
        "create_duplicate": "هذا الصف موجود بالفعل.",
        "create_warning": "أدخل اسم الصف.",
        "existing_classes": "الصفوف الموجودة",
        "no_classes": "لا توجد صفوف بعد.",
        "delete": "حذف",
        "confirm_delete": "تأكيد الحذف؟",
        "today_log": "سجل اليوم",
        "filter_by_class": "تصفية حسب الصف",
        "all_classes": "كل الصفوف",
        "present": "الحاضرون",
        "no_students": "لا يوجد طلاب في هذا التصنيف.",
        "download_excel": "تنزيل ملف Excel لليوم",
        "no_data": "لا توجد بيانات لتصديرها.",
        "download_click": "اضغط لتحميل الملف",
        "students_count": "عدد الطلاب",
        "date": "التاريخ",
        "col_class": "الصف",
        "col_name": "اسم الطالب",
        "col_status": "الحالة",
        "col_time": "وقت الدخول",
        "status_present": "حاضر",
        "status_absent": "غائب",
        "absent_time": "-",
        "live_camera": "بث الكاميرا المباشر",
        "live_hint": "افتح هذا التطبيق من متصفح هاتفك، اضغط Start، ووافق على إذن الكاميرا.",
        "scanning": "Scanning...",
        "live_suffix": "(Live)",
        "photo_text": "PHOTO!",
        "theme_label": "الوضع المظلم",
        "lang_label": "اللغة",
        "motion_threshold": "حساسية كشف الحركة",
        "match_threshold_label": "دقّة مطابقة الوجه (ارفعها إذا تشابهت الوجوه)",
        "processing": "جاري المعالجة...",
        "camera_label": "مصدر الكاميرا",
        "cam_front": "الكاميرا الأمامية",
        "cam_back": "الكاميرا الخلفية",
        "cam_external": "الكاميرا الخارجية (USB)",
        "cam_mode_label": "طريقة الكاميرا",
        "cam_mode_device": "كاميرا الجهاز / الويب كام",
        "cam_mode_url": "كاميرا خارجية بالرابط (IP)",
        "cam_url_label": "رابط الكاميرا (RTSP / HTTP-MJPEG)",
        "cam_url_hint": "الصق رابط الكاميرا، مثال: rtsp://user:pass@192.168.1.10:554/stream  أو  http://192.168.1.10:8080/video",
        "cam_url_start": "▶️ تشغيل بث الكاميرا",
        "cam_url_stop": "⏹ إيقاف البث",
        "cam_url_invalid": "تعذّر الاتصال بالكاميرا — تحقّق من الرابط.",
        "cam_mode_snapshot": "التقاط صورة (موثوق)",
        "cam_mode_live": "بث مباشر (قد لا يعمل على بعض الشبكات)",
        "snapshot_hint": "اضغط زر الكاميرا لالتقاط صورة — سيُسجَّل الحضور تلقائياً خلال لحظات.",
        "snapshot_logged": "تم تسجيل الحضور",
        "snapshot_none": "لم يتم التعرف على أي وجه مسجّل — قرّب الوجه للإطار وحاول مجدداً.",
        "snap_capture": "📸 التقاط وتسجيل",
        "snap_select_cam": "اختر الكاميرا",
        "welcome_title": "أهلاً وسهلاً في منصة طارق",
        "welcome_sub": "قف أمام الكاميرا وسيُسجَّل حضورك تلقائياً — لا حاجة لأي إجراء آخر",
        "about_title": "عن المنصة",
        "about_desc": "منصة ذكية تعتمد تقنية التعرف على الوجوه: تكشف وجه الطالب، تتحقق من هويته، وتُسجّل حضوره فوراً في سجل اليوم، ثم تُصدّر تقرير Excel بالحاضرين والغائبين.",
        "how_works_title": "كيف يعمل النظام؟",
        "how_works": [
            "1️⃣ افتح التطبيق من هاتفك أو حاسوبك وادخل بحسابك",
            "2️⃣ اضغط زر التشغيل واسمح للتطبيق باستخدام الكاميرا",
            "3️⃣ قف أمام الكاميرا — سيظهر صندوق أخضر يتبع وجهك",
            "4️⃣ بمجرد التعرف عليك، يُسجَّل حضورك تلقائياً",
            "5️⃣ شاهد ملخص الحضور ثم صدّر تقرير Excel من الزر أدناه"
        ],
        "cam_external_hint": "اختر هذا الخيار إذا كنت تستخدم حاسوباً مع كاميرا خارجية — ثم اختر الكاميرا المتصلة من شريط أدوات البث.",
        "summary_title": "ملخص الجلسة",
        "summary_present": "حاضر",
        "summary_absent": "غائب",
        "summary_interim": "يتحدّث الملخص عند أي تفاعل — اضغط زر التحديث لرؤية أحدث حالة.",
        "summary_refresh": "🔄 تحديث الملخص",
        "summary_empty": "لم يُسجَّل أي حضور بعد — قف أمام الكاميرا واضغط زر التشغيل.",
        "summary_export_btn": "⬇️ تصدير تقرير Excel لليوم",
        "login_title": "تسجيل الدخول",
        "login_sub": "هذه المنصة محمية — أدخل كلمة سر المطوّر",
        "password_label": "كلمة السر",
        "login_btn": "دخول",
        "wrong_password": "كلمة السر غير صحيحة",
        "auth_not_configured": "لم يتم ضبط كلمة السر بعد — تواصل مع المطوّر",
        "logout": "تسجيل الخروج",
        "developer": "المطوّر",
        "tab_manage": "إدارة الطلاب",
        "student_id_label": "رقم القيد (اختياري)",
        "extra_photos_label": "صور إضافية (اختياري)",
        "select_student": "اختر الطالب",
        "student_photos": "صور الطالب",
        "no_students_yet": "لا يوجد طلاب مسجّلون بعد.",
        "add_photos_btn": "إضافة الصور",
        "save_btn": "حفظ التعديلات",
        "delete_student_btn": "حذف الطالب",
        "confirm_delete_student": "تأكيد حذف الطالب؟ (لا يمكن التراجع)",
        "remove_photo": "حذف الصورة",
        "update_success": "تم تحديث بيانات الطالب.",
        "delete_success": "تم حذف الطالب.",
        "photo_added": "تمت إضافة الصور بنجاح.",
        "photo_removed": "تم حذف الصورة.",
        "manage_hint": "اختر طالباً لعرض بياناته وصوره وتعديلها.",
        "student_exists": "يوجد طالب بهذا الاسم في نفس الصف.",
        "report_range": "تقرير فترة (أسبوعي/شهري)",
        "report_from": "من تاريخ",
        "report_to": "إلى تاريخ",
        "download_range": "تنزيل تقرير الفترة (Excel)",
        "range_empty": "لا توجد سجلات في هذه الفترة.",
        "show_range_report": "عرض تقرير الفترة (اختياري)",
        "col_id": "رقم القيد",
        "col_days_present": "أيام الحضور",
        "col_total_days": "أيام التسجيل",
        "col_percent": "نسبة الحضور %",
        "sync_status": "المزامنة",
        "sync_cloud": "سحابي (Supabase)",
        "sync_local": "محلي",
    },
    "en": {
        "code": "en",
        "flag": "🇬🇧",
        "label": "English",
        "title": "Tarek Platform",
        "subtitle": "Smart Attendance System with Face Recognition",
        "info": "A cloud-based face recognition system with anti-spoofing, class-based student organization, and daily Excel attendance reports.",
        "sidebar": "Control Panel",
        "tab1": "Add Student",
        "tab2": "Manage Classes",
        "class_label": "Class / Section",
        "untracked": "No Classification",
        "name_label": "Student Name",
        "photo_label": "Student Photo",
        "add_btn": "Add Student",
        "add_success": "{} added to {} successfully!",
        "add_duplicate": "A student with this name already exists. Use a different one.",
        "add_warning": "Please enter a name and choose a photo.",
        "no_face_found": "No clear face found in the photo — try a clearer, front-facing image.",
        "add_error": "Add failed — check the cloud connection and try again.",
        "new_class_label": "New Class Name",
        "create_btn": "Create Class",
        "create_success": "Class created: {}",
        "create_duplicate": "This class already exists.",
        "create_warning": "Enter a class name.",
        "existing_classes": "Existing Classes",
        "no_classes": "No classes yet.",
        "delete": "Delete",
        "confirm_delete": "Confirm delete?",
        "today_log": "Today's Log",
        "filter_by_class": "Filter by Class",
        "all_classes": "All Classes",
        "present": "Present",
        "no_students": "No students in this category.",
        "download_excel": "Download Today's Excel",
        "no_data": "No data to export.",
        "download_click": "Click to download file",
        "students_count": "Students",
        "date": "Date",
        "col_class": "Class",
        "col_name": "Student",
        "col_status": "Status",
        "col_time": "Check-in Time",
        "status_present": "Present",
        "status_absent": "Absent",
        "absent_time": "-",
        "live_camera": "Live Camera Feed",
        "live_hint": "Open this app from your phone's browser, press Start, and allow camera access.",
        "scanning": "Scanning...",
        "live_suffix": "(Live)",
        "photo_text": "PHOTO!",
        "theme_label": "Dark Mode",
        "lang_label": "Language",
        "motion_threshold": "Motion Detection Sensitivity",
        "match_threshold_label": "Face match strictness (raise if faces look alike)",
        "processing": "Processing...",
        "camera_label": "Camera Source",
        "cam_front": "Front Camera",
        "cam_back": "Back Camera",
        "cam_mode_label": "Camera Mode",
        "cam_mode_device": "Device / Webcam",
        "cam_mode_url": "External camera by URL (IP)",
        "cam_url_label": "Camera URL (RTSP / HTTP-MJPEG)",
        "cam_url_hint": "Paste the camera URL, e.g. rtsp://user:pass@192.168.1.10:554/stream or http://192.168.1.10:8080/video",
        "cam_url_start": "▶️ Start camera stream",
        "cam_url_stop": "⏹ Stop stream",
        "cam_url_invalid": "Could not connect to the camera — check the URL.",
        "cam_mode_snapshot": "Snapshot (reliable)",
        "cam_mode_live": "Live stream (may fail on some networks)",
        "snapshot_hint": "Press the camera button to take a photo — attendance is logged automatically.",
        "snapshot_logged": "Attendance logged",
        "snapshot_none": "No registered face recognized — move closer and try again.",
        "snap_capture": "📸 Capture & log",
        "snap_select_cam": "Select camera",
        "login_title": "Login",
        "login_sub": "This platform is protected — enter the developer password",
        "password_label": "Password",
        "login_btn": "Login",
        "wrong_password": "Wrong password",
        "auth_not_configured": "Developer password is not configured yet — contact the developer",
        "logout": "Log out",
        "developer": "Developer",
        "tab_manage": "Manage Students",
        "student_id_label": "Student ID (optional)",
        "extra_photos_label": "Additional photos (optional)",
        "select_student": "Select a student",
        "student_photos": "Student photos",
        "no_students_yet": "No students registered yet.",
        "add_photos_btn": "Add photos",
        "save_btn": "Save changes",
        "delete_student_btn": "Delete student",
        "confirm_delete_student": "Confirm student deletion? (irreversible)",
        "remove_photo": "Remove photo",
        "update_success": "Student updated.",
        "delete_success": "Student deleted.",
        "photo_added": "Photos added successfully.",
        "photo_removed": "Photo removed.",
        "manage_hint": "Pick a student to view and edit their data and photos.",
        "student_exists": "A student with this name already exists in the same class.",
        "report_range": "Period Report (weekly/monthly)",
        "report_from": "From",
        "report_to": "To",
        "download_range": "Download Period Report (Excel)",
        "range_empty": "No records in this period.",
        "show_range_report": "Show period report (optional)",
        "col_id": "Student ID",
        "col_days_present": "Days Present",
        "col_total_days": "Recorded Days",
        "col_percent": "Attendance %",
        "sync_status": "Sync",
        "sync_cloud": "Cloud (Supabase)",
        "sync_local": "Local",
    },
}

# ============================================================
# إعدادات الوضع المظلم + اللون الأساسي
# ============================================================
def inject_css(is_dark, lang_code="ar"):
    """حقن CSS مخصص للواجهة العصرية مع دعم الوضع المظلم واتجاه النص."""
    if is_dark:
        bg = "#0f1117"
        card = "#1a1e2c"
        border = "#2a2f42"
        text = "#e6e9f0"
        accent = "#6366f1"
        sub = "#9aa3b5"
    else:
        bg = "#f5f7fb"
        card = "#ffffff"
        border = "#e3e8f0"
        text = "#1f2430"
        accent = "#4f46e5"
        sub = "#5b6478"

    direction = "rtl" if lang_code == "ar" else "ltr"

    st.markdown(f"""
    <style>
    :root {{
        --bg: {bg};
        --card: {card};
        --border: {border};
        --text: {text};
        --accent: {accent};
        --sub: {sub};
    }}
    .stApp {{
        background: var(--bg);
    }}
    /* الحاوية الرئيسية */
    .block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
    }}
    /* الرأس / الشعار */
    .hero {{
        background: linear-gradient(135deg, {accent}, #8b5cf6);
        border-radius: 18px;
        padding: 1.8rem 2.2rem;
        margin-bottom: 1.5rem;
        color: #fff;
        box-shadow: 0 8px 24px rgba(99,102,241,0.25);
        display: flex;
        align-items: center;
        gap: 1.2rem;
    }}
    .hero-logo {{
        font-size: 2.6rem;
        line-height: 1;
    }}
    .hero-title {{
        font-size: 1.9rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: 0.5px;
    }}
    .hero-sub {{
        font-size: 0.95rem;
        opacity: 0.9;
        margin: 0;
    }}
    .badge {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(255,255,255,0.15);
        padding: 4px 12px;
        border-radius: 30px;
        font-size: 0.8rem;
        margin-top: 0.6rem;
    }}
    /* البطاقات */
    .card {{
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }}
    .metric-card {{
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 1rem;
        text-align: center;
    }}
    .metric-value {{
        font-size: 2.2rem;
        font-weight: 800;
        color: var(--accent);
    }}
    .metric-label {{
        color: var(--sub);
        font-size: 0.85rem;
        margin-top: 0.1rem;
    }}
    /* أزرار مخصصة */
    .stButton>button {{
        border-radius: 10px;
        font-weight: 600;
    }}
    /* القوائم والحقول */
    .stTextInput>div>div>input, .stSelectbox>div>div {{
        border-radius: 10px;
    }}
    h1, h2, h3, h4, .stTitle, .stMarkdown, p, label, .stCaption {{
        color: var(--text);
    }}
    /* الشريط الجانبي */
    [data-testid="stSidebar"] {{
        background: var(--card);
        border-right: 1px solid var(--border);
    }}
    /* إزالة الخط/الإطار الطويل حول مشغّل الكاميرا */
    iframe[title="streamlit_webrtc.component"],
    div[data-testid="stCustomComponentV1"],
    div[data-testid="stCustomComponentV1"] > div {{
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
    }}
    iframe[title="streamlit_webrtc.component"] {{
        border-bottom: 0 !important;
    }}
    /* إعداد اتجاه النص حسب اللغة */
    .stApp, .stSidebar, .stMarkdown {{
        direction: {direction};
        text-align: {("right" if lang_code == "ar" else "left")};
    }}
    .hero {{
        direction: {direction};
    }}
    /* Mobile Responsive */
    @media (max-width: 768px) {{
        .block-container {{
            padding-top: 1rem;
            padding-bottom: 1rem;
            padding-left: 0.5rem;
            padding-right: 0.5rem;
        }}
        .hero {{
            padding: 1.2rem 1rem;
            flex-direction: column;
            text-align: center;
            border-radius: 12px;
        }}
        .hero-title {{
            font-size: 1.4rem;
        }}
        .hero-sub {{
            font-size: 0.85rem;
        }}
        .hero-logo {{
            font-size: 2rem;
        }}
        .card {{
            padding: 0.8rem;
            border-radius: 10px;
        }}
        .metric-card {{
            padding: 0.7rem;
        }}
        .metric-value {{
            font-size: 1.6rem;
        }}
        .stButton>button {{
            padding: 0.6rem 1rem;
            font-size: 0.9rem;
        }}
        .stTextInput>div>div>input, .stSelectbox>div>div {{
            font-size: 1rem;
        }}
        [data-testid="stSidebar"] {{
            min-width: 100%;
        }}
        .stTabs [data-baseweb="tab-list"] {{
            gap: 0.3rem;
        }}
        .stTabs [data-baseweb="tab"] {{
            padding: 0.5rem 0.8rem;
            font-size: 0.85rem;
        }}
        video {{
            width: 100% !important;
            height: auto !important;
        }}
        .stDataFrame {{
            font-size: 0.8rem;
        }}
    }}
    /* Touch-friendly */
    @media (hover: none) and (pointer: coarse) {{
        .stButton>button {{
            min-height: 44px;
        }}
        .stSelectbox>div>div, .stTextInput>div>div>input {{
            min-height: 44px;
        }}
    }}
    /* Landscape phone */
    @media (max-width: 768px) and (orientation: landscape) {{
        .hero {{
            flex-direction: row;
            padding: 1rem 1.5rem;
        }}
    }}
    </style>
    """, unsafe_allow_html=True)


# ============================================================
# دوال مساعدة للتعامل مع المجلدات (التشعيب)
# ============================================================
def get_classes():
    """يعيد قائمة الصفوف — من Supabase أو من students.json محلياً."""
    if cloud_enabled():
        try:
            r = requests.get(f"{SUPABASE_URL}/rest/v1/classes",
                             params={"select": "name", "order": "name.asc"},
                             headers=_rest_headers(), timeout=10)
            if r is not None and r.status_code == 200:
                return sorted({row.get("name") for row in r.json() if row.get("name")})
        except Exception:
            pass
        return []
    m = _load_manifest()
    return sorted(set(m.get("classes", [])))


def get_student_records(class_name=None):
    """سجلات كاملة (اسم، صف، رقم قيد، صور) للعرض والإدارة."""
    if cloud_enabled():
        try:
            r = requests.get(f"{SUPABASE_URL}/rest/v1/students",
                             params={"select": "name,class_name,student_id,photo", "order": "name.asc"},
                             headers=_rest_headers(), timeout=15)
            if r.status_code != 200:
                r = requests.get(f"{SUPABASE_URL}/rest/v1/students",
                                 params={"select": "name,class_name,photo", "order": "name.asc"},
                                 headers=_rest_headers(), timeout=15)
            if r is not None and r.status_code == 200:
                groups = {}
                for row in r.json():
                    cl = row.get("class_name") or None
                    if class_name is not None and class_name != "__root__" and cl != class_name:
                        continue
                    if class_name == "__root__" and cl is not None:
                        continue
                    key = (row.get("name"), cl)
                    g = groups.setdefault(key, {"name": row.get("name"), "class": cl,
                                                "student_id": row.get("student_id"), "photos": []})
                    if row.get("photo"):
                        g["photos"].append({"file": row["photo"], "url": public_photo_url(row["photo"])})
                return sorted(groups.values(), key=lambda g: (g["class"] or "", g["name"] or ""))
        except Exception:
            pass
        return []
    out = []
    for s in _load_manifest().get("students", []):
        cl = s.get("class") or None
        if class_name is not None and class_name != "__root__" and cl != class_name:
            continue
        if class_name == "__root__" and cl is not None:
            continue
        out.append({"name": s.get("name"), "class": cl, "student_id": s.get("student_id"),
                    "photos": [{"file": p.get("file")} for p in s.get("photos", [])]})
    return sorted(out, key=lambda g: (g["class"] or "", g["name"] or ""))


def get_students_in_class(class_name=None):
    """قائمة (اسم, صف) — None لكل الطلاب، __root__ لغير المصنفين."""
    return [(r["name"], r["class"]) for r in get_student_records(class_name)]


def all_students_names():
    return sorted(set(name for name, _ in get_students_in_class()))


# ============================================================
# قاعدة بيانات الحضور
# ============================================================
def sqlite3_connect():
    import sqlite3
    return sqlite3.connect(LOG_DB)


def init_log_db():
    conn = sqlite3_connect()
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_name TEXT,
            time TEXT,
            date TEXT
        )
    ''')
    conn.commit()
    conn.close()


def log_attendance(name):
    """يسجّل الحضور مرة واحدة لكل طالب في اليوم (لا تكرار أبداً)."""
    now = datetime.now()
    today = now.strftime("%Y-%m-%d")
    if cloud_enabled():
        try:
            r = requests.post(
                f"{SUPABASE_URL}/rest/v1/attendance",
                params={"on_conflict": "student_name,date"},
                json={"student_name": name, "time": now.strftime("%H:%M:%S"), "date": today},
                headers={**_rest_headers(), "Prefer": "resolution=ignore-duplicates,return=minimal"},
                timeout=10,
            )
            if r.status_code not in (200, 201, 204):
                print(f"Attendance log failed: {r.status_code} {r.text}")
        except Exception as e:
            print(f"Attendance log error: {e}")
        return
    conn = sqlite3_connect()
    c = conn.cursor()
    c.execute("SELECT id FROM attendance WHERE student_name = ? AND date = ?", (name, today))
    if not c.fetchone():
        c.execute("INSERT INTO attendance (student_name, time, date) VALUES (?, ?, ?)",
                  (name, now.strftime("%H:%M:%S"), today))
        conn.commit()
    conn.close()


def get_today_present():
    """يعيد قائمة (الاسم, وقت الدخول) للحاضرين اليوم."""
    if cloud_enabled():
        try:
            r = requests.get(f"{SUPABASE_URL}/rest/v1/attendance",
                             params={"select": "student_name,time", "date": f"eq.{TODAY}", "order": "time.asc"},
                             headers=_rest_headers(), timeout=10)
            if r is not None and r.status_code == 200:
                return [(row.get("student_name"), row.get("time") or "") for row in r.json()]
        except Exception:
            pass
        return []
    conn = sqlite3_connect()
    c = conn.cursor()
    c.execute("SELECT student_name, time FROM attendance WHERE date = ? ORDER BY time", (TODAY,))
    rows = c.fetchall()
    conn.close()
    return rows


# ============================================================
# توليد التقارير (صف + رقم قيد + حالة + فترة)
# ============================================================
def _present_map_for_date(date_str):
    """dict: اسم -> وقت الدخول لتاريخ معيّن (سحابة أو محلي)."""
    present = {}
    if cloud_enabled():
        try:
            r = requests.get(f"{SUPABASE_URL}/rest/v1/attendance",
                             params={"select": "student_name,time", "date": f"eq.{date_str}"},
                             headers=_rest_headers(), timeout=15)
            if r is not None and r.status_code == 200:
                for row in r.json():
                    present[row.get("student_name")] = row.get("time") or ""
        except Exception:
            pass
        return present
    conn = sqlite3_connect()
    c = conn.cursor()
    c.execute("SELECT student_name, time FROM attendance WHERE date = ?", (date_str,))
    for name, t in c.fetchall():
        present[name] = t
    conn.close()
    return present


def get_attendance_records(start_date, end_date):
    """قائمة (اسم, تاريخ, وقت) بين تاريخين شاملين."""
    if cloud_enabled():
        try:
            r = requests.get(f"{SUPABASE_URL}/rest/v1/attendance",
                             params={"select": "student_name,date,time",
                                     "date": [f"gte.{start_date}", f"lte.{end_date}"],
                                     "order": "date.asc"},
                             headers=_rest_headers(), timeout=20)
            if r is not None and r.status_code == 200:
                return [(row.get("student_name"), row.get("date"), row.get("time") or "") for row in r.json()]
        except Exception:
            pass
        return []
    conn = sqlite3_connect()
    c = conn.cursor()
    c.execute("SELECT student_name, date, time FROM attendance WHERE date BETWEEN ? AND ? ORDER BY date",
              (start_date, end_date))
    rows = c.fetchall()
    conn.close()
    return rows


def build_today_rows(date, lang):
    """يعيد صفوف سجل اليوم كقائمة قواميس (بلا pandas لخفض الذاكرة)."""
    present = _present_map_for_date(date)
    L = LANGUAGES[lang]
    rows = []
    for s in get_student_records():
        name = s.get("name")
        cl = s.get("class")
        rows.append({
            L["col_id"]: s.get("student_id") or "",
            L["col_class"]: cl if cl else L["untracked"],
            L["col_name"]: name,
            L["col_status"]: L["status_present"] if name in present else L["status_absent"],
            L["col_time"]: present.get(name, L["absent_time"]),
        })
    rows.sort(key=lambda r: (r[L["col_class"]], r[L["col_name"]]))
    return rows


def generate_excel(date, lang):
    """DataFrame للتنزيل فقط؛ pandas يُحمَّل عند الطلب فقط."""
    import pandas as pd
    L = LANGUAGES[lang]
    rows = build_today_rows(date, lang)
    return pd.DataFrame(rows, columns=[L["col_id"], L["col_class"], L["col_name"], L["col_status"], L["col_time"]])


def generate_excel_for_class(date, class_name, lang):
    df = generate_excel(date, lang)
    L = LANGUAGES[lang]
    class_display = L["untracked"] if class_name in ("__root__", None) else class_name
    return df[df[L["col_class"]] == class_display].reset_index(drop=True)


def generate_range_report(start_date, end_date, lang):
    """تقرير فترة: أيام الحضور لكل طالب + نسبة الحضور."""
    import pandas as pd
    records = get_attendance_records(start_date, end_date)
    L = LANGUAGES[lang]
    days = sorted({d for _, d, _ in records})
    total_days = len(days)
    present_days = {}
    for name, d, _ in records:
        present_days.setdefault(name, set()).add(d)
    rows = []
    for s in get_student_records():
        name = s.get("name")
        cl = s.get("class")
        cnt = len(present_days.get(name, set()))
        pct = round(cnt * 100.0 / total_days, 1) if total_days else 0.0
        rows.append({
            L["col_id"]: s.get("student_id") or "",
            L["col_class"]: cl if cl else L["untracked"],
            L["col_name"]: name,
            L["col_days_present"]: cnt,
            L["col_total_days"]: total_days,
            L["col_percent"]: pct,
        })
    rows.sort(key=lambda r: (r[L["col_class"]], r[L["col_name"]]))
    return pd.DataFrame(rows, columns=[L["col_id"], L["col_class"], L["col_name"],
                                       L["col_days_present"], L["col_total_days"], L["col_percent"]])


# ============================================================
# تهيئة قاعدة البيانات
# ============================================================
init_log_db()

# ============================================================
# تفضيلات المستخدم (اللغة + الوضع المظلم) — في جلسة واحدة
# ============================================================
if "lang_code" not in st.session_state:
    st.session_state.lang_code = "ar"
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

LANG = LANGUAGES[st.session_state.lang_code]

# تخطيط الصفحة (يجب أن يكون أول استدعاء لكل عناصر Streamlit)
st.set_page_config(page_title=f"{LANG['title']} - {LANG['subtitle']}", layout="wide")

# حقن CSS (يُعاد بعد ضبط الوضع المظلم واللغة)
inject_css(st.session_state.dark_mode, st.session_state.lang_code)

# ============================================================
# الحماية بكلمة السر (شاشة الدخول للمطوّر)
# ============================================================
AUTH_PASSWORD = get_secret("DEVELOPER_PASSWORD")

init_storage()


def render_login_gate():
    """يعرض شاشة الدخول ويوقف التطبيق حتى إدخال كلمة السر الصحيحة."""
    if st.session_state.get("auth_role") == "developer":
        return
    if not AUTH_PASSWORD:
        st.markdown(f"""
        <div class="hero">
            <div class="hero-logo">🔐</div>
            <div>
                <p class="hero-title">{LANG['login_title']}</p>
                <p class="hero-sub">{LANG['auth_not_configured']}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.stop()
    st.markdown(f"""
    <div class="hero">
        <div class="hero-logo">🔐</div>
        <div>
            <p class="hero-title">{LANG['login_title']}</p>
            <p class="hero-sub">{LANG['login_sub']}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        pwd = st.text_input(LANG["password_label"], type="password")
        if st.button(LANG["login_btn"], use_container_width=True):
            if pwd and hmac.compare_digest(pwd, AUTH_PASSWORD):
                st.session_state.auth_role = "developer"
                st.rerun()
            else:
                st.error(LANG["wrong_password"])
    st.stop()


render_login_gate()

# ============================================================
# الرأس / اسم المنصة
# ============================================================
st.markdown(f"""
<div class="hero">
    <div class="hero-logo">🎓</div>
    <div>
        <p class="hero-title">{LANG['title']}</p>
        <p class="hero-sub">{LANG['subtitle']}</p>
        <span class="badge">🟢 Online &nbsp;|&nbsp; v2.0 &nbsp;|&nbsp; {TODAY}</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown(f"<p style='color:var(--sub);'>{LANG['info']}</p>", unsafe_allow_html=True)

# ============================================================
# الشريط الجانبي
# ============================================================
with st.sidebar:
    st.markdown(f"## {LANG['sidebar']}")

    # المطوّر الحالي + تسجيل الخروج
    if st.session_state.get("auth_role") == "developer":
        st.caption(f"👨‍💻 {LANG['developer']}")
        if st.button(LANG["logout"], use_container_width=True):
            st.session_state.pop("auth_role", None)
            st.rerun()
        st.divider()

    # اللغة
    lang_label = LANG["lang_label"]
    lang_options = {f"{LANGUAGES[k]['flag']} {LANGUAGES[k]['label']}": k for k in LANGUAGES}
    selected_lang = st.selectbox(lang_label, list(lang_options.keys()), index=list(lang_options.values()).index(st.session_state.lang_code))
    if lang_options[selected_lang] != st.session_state.lang_code:
        st.session_state.lang_code = lang_options[selected_lang]
        st.rerun()
    LANG = LANGUAGES[st.session_state.lang_code]

    # الوضع المظلم
    dark = st.toggle(LANG["theme_label"], value=st.session_state.dark_mode)
    if dark != st.session_state.dark_mode:
        st.session_state.dark_mode = dark
        st.rerun()

    st.divider()

    tab1, tab_manage, tab2 = st.tabs([LANG["tab1"], LANG["tab_manage"], LANG["tab2"]])

    # ===== إضافة طالب =====
    with tab1:
        st.subheader(LANG["tab1"])
        classes = get_classes()
        options = [LANG["untracked"]] + classes
        selected_class = st.selectbox(LANG["class_label"], options)

        new_name = st.text_input(LANG["name_label"])
        new_student_id = st.text_input(LANG["student_id_label"])
        uploaded_files = st.file_uploader(LANG["photo_label"], type=["jpg", "jpeg", "png"],
                                          accept_multiple_files=True, key="add_photos")

        if st.button(LANG["add_btn"]):
            if new_name.strip() and uploaded_files:
                target_class = None if selected_class == LANG["untracked"] else selected_class
                existing = [n for n, _ in get_students_in_class("__root__" if target_class is None else target_class)]
                if new_name.strip() in existing:
                    st.warning(LANG["student_exists"])
                else:
                    files = [(f.getbuffer(), os.path.splitext(f.name)[1].lower() or ".jpg") for f in uploaded_files]
                    ok, status = add_student(new_name.strip(), target_class, new_student_id, files)
                    if status == "no_face":
                        st.warning(LANG["no_face_found"])
                    elif ok:
                        st.success(LANG["add_success"].format(new_name.strip(), selected_class))
                        st.cache_resource.clear()
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(LANG["add_error"])
            else:
                st.warning(LANG["add_warning"])

    # ===== إدارة الطلاب =====
    with tab_manage:
        st.subheader(LANG["tab_manage"])
        st.caption(LANG["manage_hint"])
        recs = get_student_records()
        if not recs:
            st.info(LANG["no_students_yet"])
        else:
            labels = [f"{r['name']} — {r['class'] or LANG['untracked']}" for r in recs]
            idx = st.selectbox(LANG["select_student"], range(len(labels)), format_func=lambda i: labels[i])
            rec = recs[idx]
            st.caption(f"{LANG['student_photos']}: {len(rec['photos'])}")

            if rec["photos"]:
                cols = st.columns(min(len(rec["photos"]), 3))
                for i, ph in enumerate(rec["photos"]):
                    with cols[i % len(cols)]:
                        try:
                            if ph.get("url"):
                                st.image(ph["url"], use_container_width=True)
                            else:
                                st.image(photo_path_for_local(ph.get("file")), use_container_width=True)
                        except Exception:
                            st.caption(str(ph.get("file")))
                        if st.button(LANG["remove_photo"], key=f"rm_{idx}_{i}"):
                            delete_student_photo(rec["name"], rec["class"], ph.get("file"))
                            st.cache_resource.clear()
                            st.success(LANG["photo_removed"])
                            time.sleep(0.5)
                            st.rerun()

            st.divider()
            edit_name = st.text_input(LANG["name_label"], value=rec["name"], key=f"en_{idx}")
            edit_id = st.text_input(LANG["student_id_label"], value=rec.get("student_id") or "", key=f"eid_{idx}")
            class_opts = [LANG["untracked"]] + get_classes()
            cur_class = rec["class"] or LANG["untracked"]
            cls_index = class_opts.index(cur_class) if cur_class in class_opts else 0
            edit_class = st.selectbox(LANG["class_label"], class_opts, index=cls_index, key=f"ec_{idx}")
            if st.button(LANG["save_btn"], key=f"save_{idx}"):
                new_class = None if edit_class == LANG["untracked"] else edit_class
                ok, _ = update_student(rec["name"], rec["class"], edit_name.strip() or rec["name"], new_class, edit_id)
                if ok:
                    st.cache_resource.clear()
                    st.success(LANG["update_success"])
                    time.sleep(0.8)
                    st.rerun()
                else:
                    st.error(LANG["add_error"])

            extra = st.file_uploader(LANG["extra_photos_label"], type=["jpg", "jpeg", "png"],
                                     accept_multiple_files=True, key=f"ex_{idx}")
            if st.button(LANG["add_photos_btn"], key=f"addp_{idx}"):
                if extra:
                    files = [(f.getbuffer(), os.path.splitext(f.name)[1].lower() or ".jpg") for f in extra]
                    ok, status = add_student_photos(rec["name"], rec["class"], files)
                    if status == "no_face":
                        st.warning(LANG["no_face_found"])
                    elif ok:
                        st.cache_resource.clear()
                        st.success(LANG["photo_added"])
                        time.sleep(0.8)
                        st.rerun()
                    else:
                        st.error(LANG["add_error"])
                else:
                    st.warning(LANG["add_warning"])

            st.divider()
            dkey = f"dels_{idx}"
            if st.session_state.get(dkey):
                if st.button(LANG["confirm_delete_student"], key=f"yes_s_{idx}"):
                    delete_student(rec["name"], rec["class"])
                    st.session_state.pop(dkey, None)
                    st.cache_resource.clear()
                    st.success(LANG["delete_success"])
                    time.sleep(0.8)
                    st.rerun()
            else:
                if st.button(LANG["delete_student_btn"], key=f"del_s_{idx}"):
                    st.session_state[dkey] = True
                    st.rerun()

    # ===== إدارة الصفوف =====
    with tab2:
        st.subheader(LANG["tab2"])
        new_class = st.text_input(LANG["new_class_label"])
        if st.button(LANG["create_btn"]):
            if new_class.strip():
                if new_class.strip() in get_classes():
                    st.warning(LANG["create_duplicate"])
                elif add_class(new_class.strip()):
                    st.success(LANG["create_success"].format(new_class.strip()))
                    st.cache_resource.clear()
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error(LANG["add_error"])
            else:
                st.warning(LANG["create_warning"])

        st.divider()
        st.write(LANG["existing_classes"])
        classes = get_classes()
        if not classes:
            st.info(LANG["no_classes"])
        else:
            for cl in classes:
                col1, col2, col3 = st.columns([3, 1, 1])
                col1.write(f"📁 {cl}")
                ckey = f"confirm_{cl}"
                if st.session_state.get(ckey, False):
                    if col2.button(LANG["confirm_delete"], key=f"yes_{cl}"):
                        delete_class(cl)
                        st.session_state.pop(ckey, None)
                        st.cache_resource.clear()
                        st.rerun()
                    if col3.button("✖", key=f"no_{cl}"):
                        st.session_state.pop(ckey, None)
                        st.rerun()
                else:
                    if col2.button(LANG["delete"], key=f"del_{cl}"):
                        st.session_state[ckey] = True
                        st.rerun()
                    col3.write("")

    st.divider()

    # ===== سجل اليوم =====
    st.markdown(f"### {LANG['today_log']}")
    classes = get_classes()
    filter_keys = ["all", "untracked"] + classes
    filter_labels = {k: (LANG["all_classes"] if k == "all" else LANG["untracked"] if k == "untracked" else k) for k in filter_keys}
    filter_class_key = st.selectbox(LANG["filter_by_class"], filter_keys, format_func=lambda k: filter_labels[k])

    if filter_class_key == "all":
        rows_today = build_today_rows(TODAY, st.session_state.lang_code)
    else:
        L_today = LANGUAGES[st.session_state.lang_code]
        cls_label = L_today["untracked"] if filter_class_key in ("untracked", None, "__root__") else filter_class_key
        rows_today = [r for r in build_today_rows(TODAY, st.session_state.lang_code) if r[L_today["col_class"]] == cls_label]

    col_status = LANG["col_status"]
    status_p = LANG["status_present"]

    # استخدام metric مخصص
    present_count = sum(1 for r in rows_today if r.get(col_status) == status_p)
    metric_html = f"""
    <div class="metric-card">
        <div class="metric-value">{present_count}</div>
        <div class="metric-label">{LANG['present']}</div>
    </div>
    """
    st.markdown(metric_html, unsafe_allow_html=True)

    if rows_today:
        _headers = [LANG["col_id"], LANG["col_class"], LANG["col_name"], LANG["col_status"], LANG["col_time"]]
        _th = "".join(f'<th style="text-align:right;padding:8px;border-bottom:1px solid #333;color:#9aa4b2">{h}</th>' for h in _headers)
        _body = "".join(
            "<tr>" + "".join(f'<td style="text-align:right;padding:8px;border-bottom:1px solid #262b3a">{r.get(h, "")}</td>' for h in _headers) + "</tr>"
            for r in rows_today
        )
        st.markdown(f'<table style="width:100%;border-collapse:collapse;font-size:0.9rem"><thead><tr>{_th}</tr></thead><tbody>{_body}</tbody></table>', unsafe_allow_html=True)
    else:
        st.info(LANG["no_students"])

    st.divider()

    # زر تنزيل Excel
    if st.button(LANG["download_excel"]):
        if filter_class_key == "all":
            df = generate_excel(TODAY, st.session_state.lang_code)
        else:
            map_val = None if filter_class_key == "untracked" else filter_class_key
            df = generate_excel_for_class(TODAY, map_val, st.session_state.lang_code)
        if df.empty:
            st.warning(LANG["no_data"])
        else:
            buf = io.BytesIO()
            df.to_excel(buf, index=False, engine="openpyxl")
            st.download_button(
                label=LANG["download_click"],
                data=buf.getvalue(),
                file_name=f"attendance_{TODAY}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    st.divider()

    # إعدادات الكاميرا: الحساسية + مصدر الكاميرا
    motion_threshold = st.slider(LANG["motion_threshold"], 1.0, 30.0, 8.0, 0.5)
    match_threshold = st.slider(LANG["match_threshold_label"], 0.30, 0.70, 0.50, 0.01)
    cam_mode_options = [LANG["cam_mode_snapshot"], LANG["cam_mode_live"], LANG["cam_mode_url"]]
    cam_mode_sel = st.radio(LANG["cam_mode_label"], cam_mode_options, horizontal=True)
    camera_url = ""
    camera_mode = "user"
    if cam_mode_sel == LANG["cam_mode_url"]:
        camera_url = st.text_input(LANG["cam_url_label"], value=st.session_state.get("cam_url", ""))
        st.session_state["cam_url"] = camera_url
        st.caption(LANG["cam_url_hint"])
        camera_source_mode = "url"
    elif cam_mode_sel == LANG["cam_mode_live"]:
        cam_options = [LANG["cam_front"], LANG["cam_back"]]
        selected_cam = st.selectbox(LANG["camera_label"], cam_options)
        camera_mode = "user" if selected_cam == LANG["cam_front"] else "environment"
        camera_source_mode = "device"
    else:
        camera_source_mode = "snapshot"

    st.caption(f"{LANG['students_count']}: {len(all_students_names())} | {LANG['date']}: {TODAY}")
    sync_label = LANG["sync_cloud"] if cloud_enabled() else LANG["sync_local"]
    st.caption(f"{LANG['sync_status']}: {sync_label}")


# ============================================================
# تحميل الوجوه المعروفة (بصمات)
# ============================================================
def _add_encoding(path, name, encodings, names):
    try:
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return
        feat = face_feature_from_image(img)
        if feat is not None:
            encodings.append(feat)
            names.append(name)
    except Exception:
        pass


@st.cache_resource
def load_known_encodings():
    encodings = []
    names = []
    if cloud_enabled():
        try:
            r = requests.get(f"{SUPABASE_URL}/rest/v1/students",
                             params={"select": "name,feature"},
                             headers=_rest_headers(), timeout=15)
            if r is not None and r.status_code == 200:
                for row in r.json():
                    if row.get("feature"):
                        encodings.append(np.array(json.loads(row["feature"]), dtype=np.float32))
                        names.append(row["name"])
        except Exception:
            pass
        return encodings, names

    for s in _load_manifest().get("students", []):
        name = s.get("name")
        for p in s.get("photos", []):
            feat = p.get("feature")
            if feat:
                encodings.append(np.array(feat, dtype=np.float32))
                names.append(name)
            else:
                _add_encoding(os.path.join(PHOTOS_DIR, p.get("file", "")), name, encodings, names)
    return encodings, names


known_encodings, known_names = load_known_encodings()


# ============================================================
# معالجة الفيديو (كشف الحركة + تعرف) — API الجديد
# ============================================================
class _SessionState:
    """حالة جلسة الكاميرا، تُشارك بين الإطارات عبر الـ callback."""
    def __init__(self, lang_code, motion_threshold, match_threshold=FACE_MATCH_THRESHOLD):
        self.lang_code = lang_code
        self.motion_threshold = motion_threshold
        self.match_threshold = match_threshold
        self.last_time = 0.0
        self.prev_centers = []
        self.process_interval = 2.0

    def update_threshold(self, new_threshold):
        self.motion_threshold = new_threshold


def analyze_bgr(img, state):
    """يعالج إطار BGR واحد ويعيد الإطار بعد الرسم.

    يعالج كل الوجوه في الإطار (وليس وجهاً واحداً)، مع تتبّع حركة مستقل لكل وجه
    (مضاد الاحتيال). مشتركة بين كاميرا الجهاز (webrtc) والكاميرا الخارجية بالرابط.
    """
    now = time.time()
    if now - state.last_time <= state.process_interval:
        return img
    state.last_time = now

    detector, recognizer = ensure_models()
    h, w = img.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(img)

    lang = LANGUAGES[state.lang_code]
    scanning_text = lang["scanning"]

    if faces is None or len(faces) == 0:
        state.prev_centers = []
        cv2.putText(img, scanning_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        return img

    prev = state.prev_centers
    used = [False] * len(prev)
    cur_centers = []
    live_suffix = lang["live_suffix"]
    photo_text = lang["photo_text"]

    for face in faces:
        fx, fy, fw, fh = face[:4]
        left = int(fx)
        top = int(fy)
        right = int(fx + fw)
        bottom = int(fy + fh)
        center = ((left + right) / 2.0, (top + bottom) / 2.0)
        cur_centers.append(center)

        aligned = recognizer.alignCrop(img, face)
        face_encoding = recognizer.feature(aligned)

        # ربط الوجه بأقرب وجه في الإطار السابق لتقدير حركته
        best_idx, best_d = -1, float("inf")
        for i, pc in enumerate(prev):
            if used[i]:
                continue
            d = ((center[0] - pc[0]) ** 2 + (center[1] - pc[1]) ** 2) ** 0.5
            if d < best_d:
                best_d, best_idx = d, i
        first_appearance = best_idx < 0 or best_d > FACE_TRACK_RADIUS
        if best_idx >= 0:
            used[best_idx] = True
        is_live = (not first_appearance) and best_d > state.motion_threshold

        # التعرف على الوجه
        best_sim = -1.0
        name = "Unknown"
        if len(known_encodings) > 0:
            best_name = "Unknown"
            for known_feat, known_name in zip(known_encodings, known_names):
                sim = recognizer.match(face_encoding, known_feat)
                if sim > best_sim:
                    best_sim = sim
                    best_name = known_name
            name = best_name if best_sim >= state.match_threshold else "Unknown"

        # أول ظهور للوجه: نعرض الاسم بدل اتهامه بأنه صورة (لا تسجيل)
        if first_appearance:
            color = (0, 255, 0) if name != "Unknown" else (255, 255, 255)
            label = name if name != "Unknown" else scanning_text
        elif is_live:
            if name != "Unknown":
                color = (0, 255, 0)
                label = f"{name} {live_suffix}"
                log_attendance(name)
            else:
                color = (0, 255, 255)
                label = f"Unknown {best_sim:.2f}"
        else:
            color = (0, 0, 255)
            label = f"{photo_text} {name}" if name != "Unknown" else f"{photo_text} {best_sim:.2f}"

        cv2.rectangle(img, (left, top), (right, bottom), color, 2)
        cv2.putText(img, label, (left, max(25, top - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    state.prev_centers = cur_centers
    return img


def make_video_processor(lang_code, motion_threshold, match_threshold):
    """يُنشئ دالة callback تعالج الإطارات: كشف وجوه + تعرف + كشف حركة."""
    state = _SessionState(lang_code, motion_threshold, match_threshold)

    def video_frame_callback(frame):
        import av
        img = frame.to_ndarray(format="bgr24")
        img = analyze_bgr(img, state)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

    return video_frame_callback


def analyze_snapshot(img, state):
    """يحلل لقطة واحدة ويسجّل الحضور للوجوه المعروفة مباشرة (بلا شرط الحركة)."""
    detector, recognizer = ensure_models()
    h, w = img.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(img)
    lang = LANGUAGES[state.lang_code]
    recognized = []
    if faces is None or len(faces) == 0:
        cv2.putText(img, lang["scanning"], (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        return img, recognized
    for face in faces:
        fx, fy, fw, fh = face[:4]
        left, top, right, bottom = int(fx), int(fy), int(fx + fw), int(fy + fh)
        aligned = recognizer.alignCrop(img, face)
        face_encoding = recognizer.feature(aligned)
        name = "Unknown"
        best_sim = -1.0
        if len(known_encodings) > 0:
            best_name = "Unknown"
            for known_feat, known_name in zip(known_encodings, known_names):
                sim = recognizer.match(face_encoding, known_feat)
                if sim > best_sim:
                    best_sim, best_name = sim, known_name
            if best_sim >= state.match_threshold:
                name = best_name
        if name != "Unknown":
            color = (0, 255, 0)
            label = name
            log_attendance(name)
            recognized.append(name)
        else:
            color = (0, 0, 255)
            label = f"Unknown {best_sim:.2f}"
        cv2.rectangle(img, (left, top), (right, bottom), color, 2)
        cv2.putText(img, label, (left, max(25, top - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return img, recognized


# ============================================================
# منطقة الكاميرا
# ============================================================
_CAMERA_PICKER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "camera_picker")
_camera_picker_component = components.declare_component("camera_picker", path=_CAMERA_PICKER_DIR)


def camera_device_picker(button_label, select_label, hint, key=None):
    """مكوّن التقاط صورة مع إمكانية اختيار كاميرا محدّدة (USB مثلاً)."""
    return _camera_picker_component(
        button_label=button_label, select_label=select_label, hint=hint, key=key, default=None
    )


def get_ice_servers():
    """STUN + TURN. افتراضياً يستخدم Open Relay المجاني، ويمكن تجاوزه بمتغيرات TURN_* في Secrets/البيئة."""
    servers = [
        {"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]},
    ]
    env_url = get_secret("TURN_URL")
    env_user = get_secret("TURN_USERNAME")
    env_cred = get_secret("TURN_CREDENTIAL")
    if env_url:
        urls = [u.strip() for u in env_url.split(",") if u.strip()]
        servers.append({"urls": urls, "username": env_user or "", "credential": env_cred or ""})
    else:
        servers.append({
            "urls": [
                "turn:openrelay.metered.ca:80",
                "turn:openrelay.metered.ca:80?transport=tcp",
                "turn:openrelay.metered.ca:443",
                "turns:openrelay.metered.ca:443?transport=tcp",
            ],
            "username": env_user or "openrelayproject",
            "credential": env_cred or "openrelayproject",
        })
    return servers


st.markdown(f"## {LANG['live_camera']}")

if camera_source_mode == "url":
    st.caption(LANG["cam_url_hint"])
    col_start, col_stop = st.columns(2)
    if col_start.button(LANG["cam_url_start"], use_container_width=True):
        st.session_state["ext_running"] = True
    if col_stop.button(LANG["cam_url_stop"], use_container_width=True):
        st.session_state["ext_running"] = False
        st.session_state["ext_cap"] = None
        st.session_state["ext_state"] = None

    if st.session_state.get("ext_running") and camera_url:
        holder = st.empty()
        cap = st.session_state.get("ext_cap")
        if cap is None:
            cap = cv2.VideoCapture(camera_url)
            st.session_state["ext_cap"] = cap
        state = st.session_state.get("ext_state")
        if state is None:
            state = _SessionState(st.session_state.lang_code, motion_threshold, match_threshold)
            st.session_state["ext_state"] = state

        got_frame = False
        for _ in range(15):
            ok, frame = cap.read()
            if not ok:
                break
            got_frame = True
            frame = analyze_bgr(frame, state)
            holder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

        if not got_frame:
            st.session_state["ext_running"] = False
            try:
                cap.release()
            except Exception:
                pass
            st.session_state["ext_cap"] = None
            st.session_state["ext_state"] = None
            st.error(LANG["cam_url_invalid"])
        else:
            time.sleep(0.05)
            st.rerun()
else:
    if camera_source_mode == "device":
        from streamlit_webrtc import webrtc_streamer
        st.caption(LANG["live_hint"])
        webrtc_streamer(
            key=f"tarek-face-recognition-{motion_threshold}",
            video_frame_callback=make_video_processor(st.session_state.lang_code, motion_threshold, match_threshold),
            media_stream_constraints={
                "video": {"facingMode": camera_mode},
                "audio": False,
            },
            rtc_configuration={"iceServers": get_ice_servers()},
        )
    else:
        st.caption(LANG["snapshot_hint"])
        result = camera_device_picker(
            LANG["snap_capture"], LANG["snap_select_cam"], LANG["snapshot_hint"], key="snap_camera")
        if isinstance(result, dict) and result.get("image") and result.get("nonce") != st.session_state.get("snap_nonce"):
            st.session_state["snap_nonce"] = result.get("nonce")
            data = result["image"]
            if "," in data:
                data = data.split(",", 1)[1]
            try:
                frame = cv2.imdecode(np.frombuffer(base64.b64decode(data), np.uint8), cv2.IMREAD_COLOR)
            except Exception:
                frame = None
            if frame is not None:
                state = st.session_state.get("snap_state")
                if state is None:
                    state = _SessionState(st.session_state.lang_code, motion_threshold, match_threshold)
                    st.session_state["snap_state"] = state
                processed, recognized = analyze_snapshot(frame, state)
                st.session_state["snap_result"] = (cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), sorted(set(recognized)))
        snap_result = st.session_state.get("snap_result")
        if snap_result:
            st.image(snap_result[0], use_container_width=True)
            if snap_result[1]:
                st.success(f"{LANG['snapshot_logged']}: " + " ، ".join(snap_result[1]))
            else:
                st.warning(LANG["snapshot_none"])

# ============================================================
# تقرير الفترة (أسبوعي/شهري)
# ============================================================
st.divider()
st.markdown(f"## {LANG['report_range']}")

if st.checkbox(LANG["show_range_report"]):
    default_start = (datetime.now() - timedelta(days=6)).strftime("%Y-%m-%d")
    col_from, col_to = st.columns(2)
    with col_from:
        start_date = st.date_input(LANG["report_from"], value=datetime.strptime(default_start, "%Y-%m-%d"))
    with col_to:
        end_date = st.date_input(LANG["report_to"], value=datetime.now())

    df_range = generate_range_report(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"),
                                     st.session_state.lang_code)
    if df_range.empty:
        st.info(LANG["range_empty"])
    else:
        st.dataframe(df_range, use_container_width=True, hide_index=True)
        if st.button(LANG["download_range"]):
            buf_range = io.BytesIO()
            df_range.to_excel(buf_range, index=False, engine="openpyxl")
            st.download_button(
                label=LANG["download_click"],
                data=buf_range.getvalue(),
                file_name=f"report_{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

_log_mem("script-end")
