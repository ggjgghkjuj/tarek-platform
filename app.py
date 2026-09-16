import streamlit as st
import cv2
import os
import io
import hmac
import time
import shutil
import numpy as np
import pandas as pd
from datetime import datetime
import av
from streamlit_webrtc import webrtc_streamer

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
# نماذج OpenCV (YuNet للكشف + SFace للتعرف) — بدون حاجة لـ dlib
# ============================================================
MODELS_DIR = "models"
FACE_DETECTOR = None
FACE_RECOGNIZER = None
FACE_MATCH_THRESHOLD = 0.36


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
        "processing": "جاري المعالجة...",
        "camera_label": "مصدر الكاميرا",
        "cam_front": "الكاميرا الأمامية",
        "cam_back": "الكاميرا الخلفية",
        "login_title": "تسجيل الدخول",
        "login_sub": "هذه المنصة محمية — أدخل كلمة سر المطوّر",
        "password_label": "كلمة السر",
        "login_btn": "دخول",
        "wrong_password": "كلمة السر غير صحيحة",
        "auth_not_configured": "لم يتم ضبط كلمة السر بعد — تواصل مع المطوّر",
        "logout": "تسجيل الخروج",
        "developer": "المطوّر",
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
        "processing": "Processing...",
        "camera_label": "Camera Source",
        "cam_front": "Front Camera",
        "cam_back": "Back Camera",
        "login_title": "Login",
        "login_sub": "This platform is protected — enter the developer password",
        "password_label": "Password",
        "login_btn": "Login",
        "wrong_password": "Wrong password",
        "auth_not_configured": "Developer password is not configured yet — contact the developer",
        "logout": "Log out",
        "developer": "Developer",
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
    /* إعداد اتجاه النص حسب اللغة */
    .stApp, .stSidebar, .stMarkdown {{
        direction: {direction};
        text-align: {("right" if lang_code == "ar" else "left")};
    }}
    .hero {{
        direction: {direction};
    }}
    </style>
    """, unsafe_allow_html=True)


# ============================================================
# دوال مساعدة للتعامل مع المجلدات (التشعيب)
# ============================================================
def get_classes():
    """يعيد قائمة الصفوف (المجلدات) داخل faces_db."""
    classes = [d for d in os.listdir(DB_PATH)
               if os.path.isdir(os.path.join(DB_PATH, d)) and not d.startswith('.')]
    return sorted(classes)


def get_students_in_class(class_name=None):
    """
    class_name=None -> كل الصفوف + الطلاب بدون تصنيف (في الجذر).
    يعيد قائمة tuples: (student_name, class_name_or_None)
    """
    students = []
    image_exts = ('.jpg', '.jpeg', '.png')

    if class_name is None or class_name == "__root__":
        for f in os.listdir(DB_PATH):
            full = os.path.join(DB_PATH, f)
            if os.path.isfile(full) and f.lower().endswith(image_exts) and not f.startswith('.'):
                students.append((os.path.splitext(f)[0], None))

    if class_name is None:
        for cl in get_classes():
            cl_dir = os.path.join(DB_PATH, cl)
            for f in os.listdir(cl_dir):
                full = os.path.join(cl_dir, f)
                if os.path.isfile(full) and f.lower().endswith(image_exts):
                    students.append((os.path.splitext(f)[0], cl))
    elif class_name != "__root__":
        cl_dir = os.path.join(DB_PATH, class_name)
        if os.path.isdir(cl_dir):
            for f in os.listdir(cl_dir):
                full = os.path.join(cl_dir, f)
                if os.path.isfile(full) and f.lower().endswith(image_exts):
                    students.append((os.path.splitext(f)[0], class_name))

    return students


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
    conn = sqlite3_connect()
    c = conn.cursor()
    now = datetime.now()
    today = now.strftime("%Y-%m-%d")
    c.execute("SELECT id FROM attendance WHERE student_name = ? AND date = ?", (name, today))
    if not c.fetchone():
        c.execute("INSERT INTO attendance (student_name, time, date) VALUES (?, ?, ?)",
                  (name, now.strftime("%H:%M:%S"), today))
        conn.commit()
    conn.close()


# ============================================================
# توليد ملف Excel (مع عمود الصف)
# ============================================================
def generate_excel(date, lang):
    """يبني DataFrame يشمل كل الطلاب مع الصف والحالة ووقت الدخول."""
    present = {}
    conn = sqlite3_connect()
    c = conn.cursor()
    c.execute("SELECT student_name, time FROM attendance WHERE date = ?", (date,))
    for name, t in c.fetchall():
        present[name] = t
    conn.close()

    students = get_students_in_class()
    rows = []
    untracked = LANGUAGES[lang]["untracked"]
    status_p = LANGUAGES[lang]["status_present"]
    status_a = LANGUAGES[lang]["status_absent"]
    absent = LANGUAGES[lang]["absent_time"]
    col_class = LANGUAGES[lang]["col_class"]
    col_name = LANGUAGES[lang]["col_name"]
    col_status = LANGUAGES[lang]["col_status"]
    col_time = LANGUAGES[lang]["col_time"]

    for name, cl in students:
        class_display = cl if cl else untracked
        if name in present:
            rows.append({col_class: class_display, col_name: name, col_status: status_p, col_time: present[name]})
        else:
            rows.append({col_class: class_display, col_name: name, col_status: status_a, col_time: absent})

    rows.sort(key=lambda r: (r[col_class], r[col_name]))
    df = pd.DataFrame(rows, columns=[col_class, col_name, col_status, col_time])
    return df


def generate_excel_for_class(date, class_name, lang):
    df = generate_excel(date, lang)
    untracked = LANGUAGES[lang]["untracked"]
    if class_name in ("__root__", None):
        class_display = untracked
    else:
        class_display = class_name
    col_class = LANGUAGES[lang]["col_class"]
    return df[df[col_class] == class_display].reset_index(drop=True)


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
try:
    AUTH_PASSWORD = st.secrets.get("DEVELOPER_PASSWORD")
except Exception:
    AUTH_PASSWORD = None


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
        return
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
        <span class="badge">🟢 Online &nbsp;|&nbsp; {TODAY}</span>
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

    tab1, tab2 = st.tabs([LANG["tab1"], LANG["tab2"]])

    # ===== إضافة طالب =====
    with tab1:
        st.subheader(LANG["tab1"])
        classes = get_classes()
        options = [LANG["untracked"]] + classes
        selected_class = st.selectbox(LANG["class_label"], options)

        new_name = st.text_input(LANG["name_label"])
        uploaded_file = st.file_uploader(LANG["photo_label"], type=["jpg", "jpeg", "png"])

        if st.button(LANG["add_btn"]):
            if new_name.strip() and uploaded_file:
                ext = os.path.splitext(uploaded_file.name)[1].lower()
                if selected_class == LANG["untracked"]:
                    target_dir = DB_PATH
                else:
                    target_dir = os.path.join(DB_PATH, selected_class)
                    os.makedirs(target_dir, exist_ok=True)
                save_path = os.path.join(target_dir, f"{new_name.strip()}{ext}")
                if os.path.exists(save_path):
                    st.warning(LANG["add_duplicate"])
                else:
                    with open(save_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    st.success(LANG["add_success"].format(new_name.strip(), selected_class))
                    st.cache_resource.clear()
                    time.sleep(1)
                    st.rerun()
            else:
                st.warning(LANG["add_warning"])

    # ===== إدارة الصفوف =====
    with tab2:
        st.subheader(LANG["tab2"])
        new_class = st.text_input(LANG["new_class_label"])
        if st.button(LANG["create_btn"]):
            if new_class.strip():
                cd = os.path.join(DB_PATH, new_class.strip())
                if os.path.exists(cd):
                    st.warning(LANG["create_duplicate"])
                else:
                    os.makedirs(cd)
                    st.success(LANG["create_success"].format(new_class.strip()))
                    time.sleep(0.5)
                    st.rerun()
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
                        shutil.rmtree(os.path.join(DB_PATH, cl), ignore_errors=True)
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
    filter_options = [LANG["all_classes"], LANG["untracked"]] + classes
    filter_class = st.selectbox(LANG["filter_by_class"], filter_options)

    if filter_class == LANG["all_classes"]:
        df_today = generate_excel(TODAY, st.session_state.lang_code)
    else:
        map_val = "__root__" if filter_class == LANG["untracked"] else filter_class
        df_today = generate_excel_for_class(TODAY, map_val, st.session_state.lang_code)

    col_status = LANG["col_status"]
    status_p = LANG["status_present"]

    # استخدام metric مخصص
    present_count = (df_today[col_status] == status_p).sum() if not df_today.empty else 0
    metric_html = f"""
    <div class="metric-card">
        <div class="metric-value">{present_count}</div>
        <div class="metric-label">{LANG['present']}</div>
    </div>
    """
    st.markdown(metric_html, unsafe_allow_html=True)

    if not df_today.empty:
        st.dataframe(df_today, use_container_width=True, hide_index=True)
    else:
        st.info(LANG["no_students"])

    st.divider()

    # زر تنزيل Excel
    if st.button(LANG["download_excel"]):
        if filter_class == LANG["all_classes"]:
            df = generate_excel(TODAY, st.session_state.lang_code)
        else:
            map_val = "__root__" if filter_class == LANG["untracked"] else filter_class
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
    cam_options = [LANG["cam_front"], LANG["cam_back"]]
    selected_cam = st.selectbox(LANG["camera_label"], cam_options)
    camera_mode = "user" if selected_cam == LANG["cam_front"] else "environment"

    st.caption(f"{LANG['students_count']}: {len(all_students_names())} | {LANG['date']}: {TODAY}")


# ============================================================
# تحميل الوجوه المعروفة (بصمات)
# ============================================================
def _add_encoding(path, name, encodings, names):
    try:
        img = cv2.imread(path)
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
    image_exts = ('.jpg', '.jpeg', '.png')
    for f in os.listdir(DB_PATH):
        full = os.path.join(DB_PATH, f)
        if os.path.isfile(full) and f.lower().endswith(image_exts) and not f.startswith('.'):
            name = os.path.splitext(f)[0]
            _add_encoding(full, name, encodings, names)
    for cl in get_classes():
        cl_dir = os.path.join(DB_PATH, cl)
        for f in os.listdir(cl_dir):
            full = os.path.join(cl_dir, f)
            if os.path.isfile(full) and f.lower().endswith(image_exts):
                name = os.path.splitext(f)[0]
                _add_encoding(full, name, encodings, names)
    return encodings, names


known_encodings, known_names = load_known_encodings()


# ============================================================
# معالجة الفيديو (كشف الحركة + تعرف) — API الجديد
# ============================================================
class _SessionState:
    """حالة جلسة الكاميرا، تُشارك بين الإطارات عبر الـ callback."""
    def __init__(self, lang_code, motion_threshold):
        self.lang_code = lang_code
        self.motion_threshold = motion_threshold
        self.last_time = 0.0
        self.prev_center = None
        self.first_face = True
        self.process_interval = 2.0


def make_video_processor(lang_code, motion_threshold):
    """يُنشئ دالة callback تعالج الإطارات: كشف وجوه + تعرف + كشف حركة."""
    state = _SessionState(lang_code, motion_threshold)

    def video_frame_callback(frame):
        img = frame.to_ndarray(format="bgr24")

        now = time.time()
        if now - state.last_time > state.process_interval:
            state.last_time = now

            detector, recognizer = ensure_models()
            h, w = img.shape[:2]
            detector.setInputSize((w, h))
            _, faces = detector.detect(img)

            scanning_text = LANGUAGES[state.lang_code]["scanning"]

            if faces is None or len(faces) == 0:
                state.prev_center = None
                state.first_face = True
                cv2.putText(img, scanning_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                return av.VideoFrame.from_ndarray(img, format="bgr24")

            face = faces[0]
            fx, fy, fw, fh = face[:4]
            left = int(fx)
            top = int(fy)
            right = int(fx + fw)
            bottom = int(fy + fh)
            center = ((left + right) / 2, (top + bottom) / 2)

            aligned = recognizer.alignCrop(img, face)
            face_encoding = recognizer.feature(aligned)

            # كشف الحركة (مضاد الاحتيال)
            is_live = False
            if state.prev_center is not None:
                dist = np.sqrt((center[0] - state.prev_center[0]) ** 2 + (center[1] - state.prev_center[1]) ** 2)
                if dist > state.motion_threshold:
                    is_live = True
            state.prev_center = center

            # التعرف على الوجه
            if len(known_encodings) > 0:
                best_sim = -1.0
                best_name = "Unknown"
                for known_feat, known_name in zip(known_encodings, known_names):
                    sim = recognizer.match(face_encoding, known_feat)
                    if sim > best_sim:
                        best_sim = sim
                        best_name = known_name
                name = best_name if best_sim >= FACE_MATCH_THRESHOLD else "Unknown"
            else:
                name = "Unknown"

            live_suffix = LANGUAGES["en"]["live_suffix"]
            photo_text = LANGUAGES["en"]["photo_text"]

            # أول كشف للوجه: نعرض الاسم بدل اتهامه بأنه صورة
            if state.first_face:
                state.first_face = False
                color = (0, 255, 0)
                label = name if name != "Unknown" else scanning_text
            elif is_live:
                if name != "Unknown":
                    color = (0, 255, 0)
                    label = f"{name} {live_suffix}"
                    log_attendance(name)
                else:
                    color = (0, 255, 255)
                    label = f"Unknown {live_suffix}"
            else:
                color = (0, 0, 255)
                label = f"{photo_text} {name}"

            cv2.rectangle(img, (left, top), (right, bottom), color, 2)
            cv2.putText(img, label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    return video_frame_callback


# ============================================================
# منطقة الكاميرا
# ============================================================
st.markdown(f"## {LANG['live_camera']}")
st.caption(LANG["live_hint"])

webrtc_streamer(
    key="tarek-face-recognition",
    video_frame_callback=make_video_processor(st.session_state.lang_code, motion_threshold),
    media_stream_constraints={
        "video": {"facingMode": camera_mode},
        "audio": False,
    },
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
)
