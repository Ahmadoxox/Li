import json
import os
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import requests
from gtts import gTTS
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
import yfinance as yf

# إعدادات واجهة التطبيق
st.set_page_config(page_title="الوكيل المالي الذكي - فحص رموز MT5", page_icon="🤖", layout="wide")

# جلب الأسرار وإزالة أي مسافات زائدة
metaapi_token = st.secrets.get("METAAPI_TOKEN", os.environ.get("METAAPI_TOKEN", "")).strip()
metaapi_account_id = st.secrets.get("METAAPI_ACCOUNT_ID", os.environ.get("METAAPI_ACCOUNT_ID", "")).strip()
metaapi_region = st.secrets.get("METAAPI_REGION", "london").strip().lower()

# ضبط رابط الخادم بناءً على منطقة الحساب (لندن)
if "london" in metaapi_region:
    API_BASE_URL = "https://mt-client-api-v1.london.agiliumtrade.ai"
else:
    API_BASE_URL = f"https://mt-client-api-v1.{metaapi_region}.agiliumtrade.ai"

@tool
def get_mt5_account_balance() -> str:
    """جلب رصيد الحساب الحقيقي والسيولة مباشرة من منصة MT5 عبر سحابة لندن."""
    if not metaapi_token or not metaapi_account_id:
        return "⚠️ مفاتيح MetaApi غير مضافة."
    
    url = f"{API_BASE_URL}/users/current/accounts/{metaapi_account_id}/account-information"
    headers = {"auth-token": metaapi_token}
    
    try:
        res = requests.get(url, headers=headers, timeout=15)
        if res.status_code == 200:
            data = res.json()
            return f"""📊 معلومات رصيد حسابك الحقيقي على MT5:
- الرصيد (Balance): ${data.get('balance', 0):.2f}
- السيولة المتاحة (Equity): ${data.get('equity', 0):.2f}
- الهامش المجاني (Free Margin): ${data.get('freeMargin', 0):.2f}
"""
        else:
            return f"⚠️ فشل الاتصال بخادم لندن (رمز الاستجابة {res.status_code}): {res.text}"
    except Exception as e:
        return f"خطأ في الاتصال بالخادم: {str(e)}"

@tool
def get_mt5_open_positions() -> str:
    """فحص وجلب كافة الصفقات المفتوحة حالياً فعلياً من خادم MT5."""
    if not metaapi_token or not metaapi_account_id:
        return "⚠️ مفاتيح MetaApi غير مضافة."
        
    url = f"{API_BASE_URL}/users/current/accounts/{metaapi_account_id}/positions"
    headers = {"auth-token": metaapi_token}
    
    try:
        res = requests.get(url, headers=headers, timeout=15)
        if res.status_code == 200:
            positions = res.json()
            if not positions:
                return "📋 لا توجد أي صفقات مفتوحة حالياً على الحساب في منصة MT5."
            
            report = "📋 الصفقات المفتوحة حالياً على منصة MT5:\n"
            for p in positions:
                report += f"- الأصل: {p.get('symbol')} | النوع: {p.get('type')} | الحجم: {p.get('volume')} | سعر الدخول: {p.get('openPrice')} | الربح الحالي: ${p.get('profit', 0):.2f}\n"
            return report
        else:
            return f"⚠️ فشل جلب الصفقات المفتوحة (رمز الاستجابة {res.status_code}): {res.text}"
    except Exception as e:
        return f"خطأ أثناء الاتصال لجلب الصفقات: {str(e)}"

@tool
def get_mt5_symbols() -> str:
    """جلب قائمة الأسماء والرموز الدقيقة المتاحة للتداول على حسابك في منصة MT5 لمعرفة الاسم الصحيح للذهب أو العملات."""
    if not metaapi_token or not metaapi_account_id:
        return "⚠️ مفاتيح MetaApi غير مضافة."
        
    url = f"{API_BASE_URL}/users/current/accounts/{metaapi_account_id}/symbols"
    headers = {"auth-token": metaapi_token}
    
    try:
        res = requests.get(url, headers=headers, timeout=15)
        if res.status_code == 200:
            symbols = res.json()
            if not symbols:
                return "⚠️ لم يتم العثور على رموز متاحة."
            # نعرض عينة من أهم الرموز أو الرموز المتعلقة بالذهب والعملات الرئيسية
            filtered = [s for s in symbols if any(k in s.upper() for k in ["XAU", "GOLD", "EUR", "GBP", "BTC"])]
            return f"📋 الرموز المتاحة في حسابك (أهمها):\n{', '.join(filtered if filtered else symbols[:30])}"
        else:
            return f"⚠️ فشل جلب الرموز (رمز الاستجابة {res.status_code}): {res.text}"
    except Exception as e:
        return f"خطأ: {str(e)}"

tools = [get_mt5_account_balance, get_mt5_open_positions, get_mt5_symbols]

api_key = st.secrets.get("GOOGLE_API_KEY", os.environ.get("GOOGLE_API_KEY", "")).strip()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=api_key,
    system_instruction="أنت مدير محفظة ذكي، تفحص حساب MT5 وتساعد في مطابقة الرموز بدقة."
)
agent_executor = create_agent(llm, tools)

st.title("🤖 الوكيل المالي الذكي - فحص رموز MT5 والسيولة")
st.write("اعرض الرموز المتاحة على منصتك لمعرفة الأسماء الصحيحة التي يقبلها البروكر.")

user_input = st.text_input("💬 اطلب من البوت (مثال: اعرض الرموز المتاحة في حسابي):", placeholder="اكتب أمرك هنا...")

if st.button("🚀 تنفيذ عبر السحابة", type="primary"):
    if user_input:
        with st.spinner("جاري جلب قائمة الرموز والبيانات من خادم MT5..."):
            try:
                res = agent_executor.invoke({"messages": [("user", user_input)]})
                ans = res["messages"][-1].content
            except Exception as e:
                ans = f"حدث خطأ أثناء تنفيذ الطلب: {str(e)}"
            
            st.success("🤖 تقرير الخادم:")
            st.write(ans)
            
            try:
                if ans and isinstance(ans, str) and len(ans.strip()) > 0:
                    audio_file = "ans.mp3"
                    gTTS(text=ans, lang="ar").save(audio_file)
                    st.audio(audio_file)
            except Exception:
                pass
    else:
        st.warning("الرجاء كتابة أمر أولاً.")
