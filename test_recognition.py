# -*- coding: utf-8 -*-
"""
اختبار دقة التعرف: يبني مصفوفة تشابه بين كل الصور المحلية (faces_db)
ويقيّم العتبة الحالية 0.36.
    python test_recognition.py
"""
import os
import numpy as np
import cv2

BASE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE, "faces_db")
M = os.path.join(BASE, "models")


def load_models():
    det = cv2.FaceDetectorYN_create(os.path.join(M, "face_detection_yunet_2023mar.onnx"), "", (320, 320))
    rec = cv2.FaceRecognizerSF_create(os.path.join(M, "face_recognition_sface_2021dec.onnx"), "")
    return det, rec


def feature_of(path, det, rec):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None, "(تعذر قراءة الصورة)"
    h, w = img.shape[:2]
    det.setInputSize((w, h))
    _, faces = det.detect(img)
    if faces is None or len(faces) == 0:
        return None, "(لا يوجد وجه)"
    aligned = rec.alignCrop(img, faces[0])
    feat = rec.feature(aligned)
    fx, fy, fw, fh = faces[0][:4]
    return feat, f"وجه عند {int(fx)},{int(fy)} {int(fw)}x{int(fh)}"


def main():
    det, rec = load_models()
    entries = []
    for f in sorted(os.listdir(DB_PATH)):
        full = os.path.join(DB_PATH, f)
        if os.path.isfile(full) and f.lower().endswith((".jpg", ".jpeg", ".png")):
            entries.append((os.path.splitext(f)[0], "", full))
    for d in sorted(os.listdir(DB_PATH)):
        if not os.path.isdir(os.path.join(DB_PATH, d)):
            continue
        for f in sorted(os.listdir(os.path.join(DB_PATH, d))):
            full = os.path.join(DB_PATH, d, f)
            if os.path.isfile(full) and f.lower().endswith((".jpg", ".jpeg", ".png")):
                entries.append((os.path.splitext(f)[0], d, full))

    feats = []
    print("=== استخراج البصمات ===")
    for name, cl, full in entries:
        feat, meta = feature_of(full, det, rec)
        label = name + ("   [" + cl + "]" if cl else "")
        if feat is None:
            print("  ", name, meta)
        else:
            feats.append((label, feat))
            print("  ", name, meta)

    n = len(feats)
    if n == 0:
        print("لا توجد صور قابلة للمعالجة")
        return

    print("\n=== مصفوفة التشابه (COSINE) ===")
    wider = max(len(l) for l, _ in feats) + 2
    print(" " * wider + "".join(f"{l.split()[0][:10]:>11}" for l, _ in feats))
    crosses = []
    for i in range(n):
        row = f"{feats[i][0]:<{wider}}"
        for j in range(n):
            s = rec.match(feats[i][1], feats[j][1])
            row += f"{s:11.3f}"
            if i != j:
                crosses.append(s)
        print(row)

    best_cross = max(crosses)
    print("\nأعلى تطابق بين أشخاص مختلفين: %.3f" % best_cross)
    thr = 0.36
    if best_cross < thr - 0.05:
        verdict = "آمنة مع هامش جيد"
    elif best_cross < thr:
        verdict = "آمنة لكن بهامش ضيق"
    else:
        verdict = "خطير: قد تختلط الوجوه عند هذا المستوى!"
    print("العتبة الحالية %.2f -> %s" % (thr, verdict))
    suggested = round(max(thr + 0.02, best_cross + 0.10), 3)
    print("عتبة مستحسنة (هامش +0.10): %.3f" % suggested)


if __name__ == "__main__":
    main()