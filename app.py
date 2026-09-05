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
st.set_page_config(page_title="الوكيل المالي - التداول الآلي على MT5", page_icon="🤖", layout="wide")

# جلب أسرار السحابة مع إزالة أي مسافات زائدة تلقائياً (.strip())
metaapi_token = st.secrets.get("METAAPI_TOKEN", os.environ.get("METAAPI_TOKEN", "")).strip()
metaapi_account_id = st.secrets.get("METAAPI_ACCOUNT_ID", os.environ.get("METAAPI_ACCOUNT_ID", "")).strip()

ASSET_MAP = {
    "ذهب": {"yf": "GC=F", "mt5": "XAUUSD"},
    "الذهب": {"yf": "GC=F", "mt5": "XAUUSD"},
    "gold": {"yf": "GC=F", "mt5": "XAUUSD"},
    "يورو دولار": {"yf": "EURUSD=X", "mt5": "EURUSD"},
    "بيتكوين": {"yf": "BTC-USD", "mt5": "BTCUSD"},
    "نفط": {"yf": "CL=F", "mt5": "WTI"},
}

def resolve_asset(asset_name: str):
    clean = asset_name.strip().lower()
    if clean in ASSET_MAP:
        return ASSET_MAP[clean]
    return {"yf": asset_name.upper(), "mt5": asset_name.upper()}

@tool
def get_mt5_account_balance() -> str:
    """جلب رصيد الحساب الحقيقي، السيولة (Equity)، والمارجين مباشرة من منصة MT5 عبر السحابة."""
    if not metaapi_token or not metaapi_account_id:
        return "⚠️ مفاتيح MetaApi غير مضافة في الأسرار أو تحتوي على بيانات فارغة."
    
    url = f"https://mt-client-api-v1.agiliumtrade.ai/users/current/accounts/{metaapi_account_id}/account-information"
    headers = {"auth-token": metaapi_token}
    
    try:
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code == 200:
            data = res.json()
            return f"""📊 معلومات رصيد حسابك على MT5:
- الرصيد (Balance): ${data.get('balance', 0):.2f}
- السيولة المتاحة (Equity): ${data.get('equity', 0):.2f}
- الهامش المجاني (Free Margin): ${data.get('freeMargin', 0):.2f}
- الرافعة المالية: 1:{data.get('leverage', 'N/A')}
- العملة الأساسية: {data.get('currency', 'USD')}
"""
        else:
            return f"⚠️ فشل الاتصال بخادم MetaApi (تأكد من صحة الـ Token و Account ID): {res.text}"
    except Exception as e:
        return f"خطأ في الاتصال بالخادم: {str(e)}"

@tool
def analyze_and_execute_autonomous_trade(timeframe: str, asset_name: str, lot_size: float) -> str:
    """
    يفحص السوق والشموع (مثل 1m, 5m, 1h) باستخدام مؤشر RSI والمتوسطات المتحركة، 
    وإذا تطابقت الشروط الإيجابية، ينفذ صفقة حقيقية (شراء أو بيع) تلقائياً على MT5 عبر سحابة MetaApi.
    """
    symbols = resolve_asset(asset_name)
    yf_symbol = symbols["yf"]
    mt5_symbol = symbols["mt5"]
    
    period_map = {"1m": "1d", "5m": "5d", "15m": "1mo", "1h": "1mo", "1d": "3mo"}
    period = period_map.get(timeframe, "5d")
    
    try:
        df = yf.download(yf_symbol, period=period, interval=timeframe, progress=False)
        if df.empty or len(df) < 15:
            return f"❌ بيانات الشموع غير كافية لـ {asset_name} على الإطار {timeframe}."
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        current_price = float(df['Close'].iloc[-1])
        current_rsi = float(df['RSI'].iloc[-1])
        sma_20 = float(df['SMA_20'].iloc[-1])
        sma_50 = float(df['SMA_50'].iloc[-1])
        
        action_type = None
        action_desc = ""
        
        if current_rsi < 36 or (current_price > sma_20 and current_rsi < 55):
            action_type = "ORDER_TYPE_BUY"
            action_desc = "شراء (BUY)"
        elif current_rsi > 72:
            action_type = "ORDER_TYPE_SELL"
            action_desc = "بيع / جني أرباح (SELL)"
        else:
            return f"""⏸️ وضع المراقبة الآلية (لا توجد صفقة الآن):
- الأصل: {asset_name} ({mt5_symbol})
- السعر الحالي: ${current_price:.2f}
- مؤشر RSI: {current_rsi:.1f} (منطقة حيادية، البوت ينتظر الفرصة الأنسب).
"""

        if metaapi_token and metaapi_account_id and action_type:
            url = f"https://mt-client-api-v1.agiliumtrade.ai/users/current/accounts/{metaapi_account_id}/trade"
            headers = {"auth-token": metaapi_token, "Content-Type": "application/json"}
            payload = {
                "actionType": action_type,
                "symbol": mt5_symbol,
                "volume": float(lot_size),
                "comment": f"AI Auto Trade RSI {current_rsi:.1f}"
            }
            
            res = requests.post(url, json=payload, headers=headers, timeout=15)
            if res.status_code in [200, 201]:
                return f"""🚀 تم تنفيذ الصفقة الآلية بنجاح على MT5!
- الأصل: {asset_name} ({mt5_symbol})
- نوع الصفقة: {action_desc}
- حجم العقد (Lot): {lot_size}
- السعر الحالي عند التنفيذ: ${current_price:.2f} | مؤشر RSI: {current_rsi:.1f}
"""
            else:
                return f"⚠️ فشل تنفيذ الصفقة على MT5: {res.text}"
        else:
            return f"الإشارة المقترحة هي {action_desc} ولكن بيانات MetaApi غير مكتملة."
            
    except Exception as e:
        return f"خطأ في التشغيل الآلي للتحليل: {str(e)}"

tools = [get_mt5_account_balance, analyze_and_execute_autonomous_trade]

api_key = st.secrets.get("GOOGLE_API_KEY", os.environ.get("GOOGLE_API_KEY", "")).strip()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=api_key,
    system_instruction="أنت نظام تداول آلي سحابي خبير بأسواق الفوركس والذهب عبر MetaTrader 5."
)
agent_executor = create_agent(llm, tools)

st.title("🤖 الوكيل المالي الذكي - التداول الآلي على MT5")
st.write("فحص الرصيد، تحليل الشموع، وتنفيذ الصفقات ذاتياً عبر السحابة.")

user_input = st.text_input("💬 اطلب من البوت (مثال: افحص رصيدي في MT5):", placeholder="اكتب أمرك هنا...")

if st.button("🚀 تنفيذ عبر السحابة", type="primary"):
    if user_input:
        with st.spinner("البوت يتصل بسحابة MT5 ويتخذ القرار..."):
            try:
                res = agent_executor.invoke({"messages": [("user", user_input)]})
                ans = res["messages"][-1].content
            except Exception as e:
                ans = f"حدث خطأ أثناء تنفيذ الطلب: {str(e)}"
            
            st.success("🤖 تقرير التنفيذ السحابي:")
            st.write(ans)
            
            # توليد الصوت بشكل آمن يحمي من توقف التطبيق
            try:
                if ans and isinstance(ans, str) and len(ans.strip()) > 0:
                    audio_file = "ans.mp3"
                    gTTS(text=ans, lang="ar").save(audio_file)
                    st.audio(audio_file)
            except Exception:
                pass # تخطي توليد الصوت في حال حدوث أي استثناء لكي لا يتعطل التطبيق
    else:
        st.warning("الرجاء كتابة أمر أولاً.")
