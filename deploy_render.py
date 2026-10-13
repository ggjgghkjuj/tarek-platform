# -*- coding: utf-8 -*-
"""
أتمتة نشر منصة طارق على Render.

يحاج فقط مفتاح Render API (Account Settings → API Keys، يبدأ بـ rnd_).

الاستخدام:
    1) ضع المفتاح في متغير البيئة:
           set RENDER_API_KEY=rnd_xxxxxxxx
       أو في ملف ".render_token" (سطر واحد).
    2) تأكد أن ملف .streamlit/secrets.toml يحتوي مفاتيح Supabase.
    3) شغّل:
           python deploy_render.py

يقوم السكربت بـ:
    - قراءة مفاتيح Supabase من secrets.toml
    - إنشاء خدمة ويب (أو تحديث الموجودة) من المستودع العام
    - حقن متغيرات البيئة وإطلاق النشر
"""
import os
import re
import sys
import time

import requests

API = "https://api.render.com/v1"
BASE = os.path.dirname(os.path.abspath(__file__))
TOKEN_FILE = os.path.join(BASE, ".render_token")
REPO = "https://github.com/ggjgghkjuj/tarek-platform"
NAME = os.environ.get("RENDER_SERVICE_NAME", "tarek-platform")
BRANCH = "master"
REGION = os.environ.get("RENDER_REGION", "frankfurt")
BUILD_CMD = "pip install -r requirements.txt"
START_CMD = (
    "streamlit run app.py --server.port $PORT --server.address 0.0.0.0 "
    "--server.headless true --server.enableCORS false --server.enableXsrfProtection false"
)


def get_token(path, env_names):
    for n in env_names:
        v = os.environ.get(n)
        if v:
            return v.strip()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return f.read().strip()
    return None


def headers(tok):
    return {"Authorization": "Bearer " + tok, "Accept": "application/json"}


def api(tok, method, path, **kw):
    return requests.request(method, API + path, headers=headers(tok), timeout=120, **kw)


def read_secrets():
    p = os.path.join(BASE, ".streamlit", "secrets.toml")
    data = {}
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            for line in f:
                m = re.match(r'\s*([A-Z_]+)\s*=\s*"?(.*?)"?\s*$', line)
                if m:
                    data[m.group(1)] = m.group(2)
    return data


def owners(tok):
    r = api(tok, "GET", "/owners", params={"limit": 50})
    if r.status_code >= 300:
        print("فشل جلب المساحات:", r.status_code, r.text[:300])
        sys.exit(1)
    return [o["owner"]["id"] for o in r.json()]


def existing_service(tok):
    r = api(tok, "GET", "/services", params={"limit": 100})
    if r.status_code >= 300:
        return None
    for s in r.json():
        svc = s.get("service", {})
        if svc.get("name") == NAME:
            return svc
    return None


def main():
    tok = get_token(TOKEN_FILE, ["RENDER_API_KEY", "RENDER_TOKEN"])
    if not tok:
        print("لا يوجد مفتاح Render. ضع RENDER_API_KEY أو ملف .render_token")
        sys.exit(1)

    sec = read_secrets()
    env_vars = [{"key": "PYTHON_VERSION", "value": "3.11.9"}]
    for k in ("DEVELOPER_PASSWORD", "SUPABASE_URL", "SUPABASE_SERVICE_KEY",
              "TURN_URL", "TURN_USERNAME", "TURN_CREDENTIAL"):
        if sec.get(k):
            env_vars.append({"key": k, "value": sec[k]})
    if not sec.get("SUPABASE_URL"):
        print("!! تنبيه: لا يوجد SUPABASE_URL في secrets.toml — شغّل setup_supabase.py أولاً")

    svc = existing_service(tok)
    if svc is None:
        oid = owners(tok)[0]
        body = {
            "type": "web_service",
            "name": NAME,
            "ownerId": oid,
            "repo": REPO,
            "branch": BRANCH,
            "autoDeploy": "no",
            "envVars": env_vars,
            "serviceDetails": {
                "env": "python",
                "plan": "free",
                "region": REGION,
                "healthCheckPath": "/_stcore/health",
                "envSpecificDetails": {
                    "buildCommand": BUILD_CMD,
                    "startCommand": START_CMD,
                },
            },
        }
        r = api(tok, "POST", "/services", json=body)
        if r.status_code >= 300:
            print("فشل إنشاء الخدمة:", r.status_code, r.text[:800])
            sys.exit(1)
        svc = r.json()["service"]
        print("أُنشئت الخدمة:", svc["id"])
    else:
        sid = svc["id"]
        print("الخدمة موجودة:", sid, "- تحديث المتغيرات...")
        api(tok, "PUT", "/services/%s/env-vars" % sid, json=env_vars)
        api(tok, "POST", "/services/%s/deploys" % sid,
            json={"clearCache": "do_not_clear"})

    sid = svc["id"]
    url = svc.get("serviceDetails", {}).get("url") or ("https://%s.onrender.com" % NAME)
    print("\nالرابط:", url)
    print("راقب حالة النشر: https://dashboard.render.com/web/%s" % sid)


if __name__ == "__main__":
    main()
