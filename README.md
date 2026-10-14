---
title: Tarek Platform
emoji: 🎓
colorFrom: indigo
colorTo: purple
sdk: streamlit
sdk_version: 1.51.0
app_file: app.py
pinned: false
---

# منصة طارق | Tarek Platform

منصة التعرّف على الوجوه وتسجيل حضور الطلاب، مبنية بـ Streamlit + OpenCV (YuNet + SFace) مع تخزين سحابي على Supabase.

## الميزات
- تسجيل حضور تلقائي بالتعرف على الوجه.
- لقطة كاميرا موثوقة (Snapshot) مع اختيار كاميرا USB، ووضع بث مباشر، وكاميرا IP عبر الرابط.
- إدارة الطلاب والصفوف، وتقارير الحضور اليومية وتقارير الفترة (Excel).
- واجهة عربية/إنجليزية + وضع مظلم، ومحمية بكلمة سر المطوّر.

## الإعداد
تُضبط الأسرار عبر Environment Variables:
- `DEVELOPER_PASSWORD`
- `SUPABASE_URL`
- `SUPABASE_SERVICE_KEY`
