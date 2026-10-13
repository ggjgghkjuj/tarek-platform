# -*- coding: utf-8 -*-
"""
أتمتة تهيئة Supabase لمنصة طارق.

يحتاج فقط رمز وصول شخصي (PAT) من Supabase (Account → Access Tokens).

الاستخدام:
    1) ضع الرمز في متغير البيئة:
           set SUPABASE_ACCESS_TOKEN=sbp_xxxxxxxx
       أو في ملف ".supabase_token" في جذر المشروع (سطر واحد).
    2) شغّل:
           python setup_supabase.py

يقوم السكربت بـ:
    - إنشاء مشروع باسم tarek-platform (إن لم يكن موجوداً)
    - انتظار جاهزية المشروع
    - تنفيذ supabase_schema.sql + إنشاء مخزن الصور العام student-photos
    - كتابة .streamlit/secrets.toml بالمفاتيح المطلوبة
"""
import os
import sys
import time
import json
import secrets

import requests

API = "https://api.supabase.com/v1"
BASE = os.path.dirname(os.path.abspath(__file__))
TOKEN_FILE = os.path.join(BASE, ".supabase_token")
PROJECT_NAME = os.environ.get("SUPABASE_PROJECT_NAME", "tarek-platform")
REGION = os.environ.get("SUPABASE_REGION", "eu-central-1")
BUCKET = "student-photos"
DEV_PASSWORD = os.environ.get("DEVELOPER_PASSWORD", "tarek1234")


def get_token():
    t = os.environ.get("SUPABASE_ACCESS_TOKEN")
    if not t and os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, encoding="utf-8") as f:
            t = f.read().strip()
    if not t:
        print("لا يوجد رمز وصول. ضع SUPABASE_ACCESS_TOKEN أو ملف .supabase_token")
        sys.exit(1)
    return t


def headers(tok):
    return {"Authorization": "Bearer " + tok, "Content-Type": "application/json"}


def api(tok, method, path, **kw):
    return requests.request(method, API + path, headers=headers(tok), timeout=120, **kw)


def find_or_create_project(tok):
    r = api(tok, "GET", "/projects")
    if r.status_code < 300:
        for p in r.json():
            if p.get("name") == PROJECT_NAME:
                print("المشروع موجود مسبقاً:", p["ref"])
                return p["ref"]

    orgs = api(tok, "GET", "/organizations").json()
    if not orgs:
        print("لا توجد منظمة (organization) في الحساب")
        sys.exit(1)
    org = orgs[0]
    org_id = org.get("id")
    org_slug = org.get("slug") or org.get("id")
    db_pass = secrets.token_urlsafe(24)

    full = {
        "name": PROJECT_NAME,
        "organization_id": org_id,
        "organization_slug": org_slug,
        "db_pass": db_pass,
        "region": REGION,
        "plan": "free",
    }
    r = api(tok, "POST", "/projects", json=full)
    if r.status_code >= 300:
        # بعض الحقول مهجورة في الإصدارات الحديثة — جرّب الحد الأدنى
        minimal = {"name": PROJECT_NAME, "organization_slug": org_slug, "db_pass": db_pass}
        r = api(tok, "POST", "/projects", json=minimal)
    if r.status_code >= 300:
        print("فشل إنشاء المشروع:", r.status_code, r.text)
        sys.exit(1)

    ref = r.json()["ref"]
    print("أُنشئ المشروع:", ref, "- بانتظار الجاهزية (قد يستغرق ~دقيقتين)...")
    return ref


def wait_ready(tok, ref, timeout=360):
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = api(tok, "GET", "/projects/" + ref)
        if r.status_code == 200:
            status = r.json().get("status")
            print("  الحالة:", status)
            if status == "ACTIVE_HEALTHY":
                return True
        time.sleep(12)
    print("!! لم يجهز خلال المهلة — تابِع التهيئة لاحقاً بإعادة التشغيل.")
    return False


def run_sql(tok, ref, sql):
    r = api(tok, "POST", "/projects/" + ref + "/database/query", json={"query": sql})
    if r.status_code >= 300:
        print("تحذير SQL:", r.status_code, r.text[:400])
    return r


def get_keys(tok, ref):
    r = api(tok, "GET", "/projects/" + ref + "/api-keys", params={"reveal": "true"})
    keys = r.json() if r.status_code < 300 else []
    anon = service = None
    for k in keys:
        name = (k.get("name") or "").lower()
        if name == "service_role":
            service = k.get("api_key")
        elif name == "anon":
            anon = k.get("api_key")
    if not service:
        for k in keys:
            if k.get("type") in ("secret", "service_role"):
                service = k.get("api_key")
    return anon, service


def main():
    tok = get_token()
    ref = find_or_create_project(tok)
    wait_ready(tok, ref)

    sql_path = os.path.join(BASE, "supabase_schema.sql")
    with open(sql_path, encoding="utf-8") as f:
        schema = f.read()
    run_sql(tok, ref, schema)
    run_sql(tok, ref,
            "insert into storage.buckets (id, name, public) values "
            "('%s','%s',true) on conflict (id) do nothing;" % (BUCKET, BUCKET))

    url = "https://%s.supabase.co" % ref
    anon, service = get_keys(tok, ref)
    if not service:
        print("!! تعذّر جلب مفتاح الخدمة — خذه يدوياً من Settings → API")
        sys.exit(1)

    secrets_path = os.path.join(BASE, ".streamlit", "secrets.toml")
    os.makedirs(os.path.dirname(secrets_path), exist_ok=True)
    with open(secrets_path, "w", encoding="utf-8") as f:
        f.write('DEVELOPER_PASSWORD = "%s"\n\n' % DEV_PASSWORD)
        f.write('SUPABASE_URL = "%s"\n' % url)
        f.write('SUPABASE_SERVICE_KEY = "%s"\n' % service)

    print("\nتمت تهيئة Supabase بنجاح:")
    print("  URL     :", url)
    print("  anon    :", (anon or "")[:14] + "...")
    print("  service :", service[:14] + "...")
    print("  كُتب الملف: .streamlit/secrets.toml")


if __name__ == "__main__":
    main()
