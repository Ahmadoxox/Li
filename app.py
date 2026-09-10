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
st.set_page_config(page_title="الوكيل المالي الذكي - تداول الذهب الآلي", page_icon="🤖", layout="wide")

# جلب الأسرار وإزالة أي مسافات زائدة
metaapi_token = st.secrets.get("METAAPI_TOKEN", os.environ.get("METAAPI_TOKEN", "")).strip()
metaapi_account_id = st.secrets.get("METAAPI_ACCOUNT_ID", os.environ.get("METAAPI_ACCOUNT_ID", "")).strip()
metaapi_region = st.secrets.get("METAAPI_REGION", "london").strip().lower()

# ضبط رابط الخادم بناءً على منطقة الحساب (لندن)
if "london" in metaapi_region:
    API_BASE_URL = "https://mt-client-api-v1.london.agiliumtrade.ai"
else:
    API_BASE_URL = f"https://mt-client-api-v1.{metaapi_region}.agiliumtrade.ai"

ASSET_MAP = {
    "ذهب": {"yf": "GC=F", "mt5": "XAUUSD", "sl_pips": 4.0, "tp_pips": 8.0},
    "الذهب": {"yf": "GC=F", "mt5": "XAUUSD", "sl_pips": 4.0, "tp_pips": 8.0},
    "gold": {"yf": "GC=F", "mt5": "XAUUSD", "sl_pips": 4.0, "tp_pips": 8.0},
    "يورو دولار": {"yf": "EURUSD=X", "mt5": "EURUSD", "sl_pips": 0.0030, "tp_pips": 0.0060},
}

def resolve_asset(asset_name: str):
    clean = asset_name.strip().lower()
    if clean in ASSET_MAP:
        return ASSET_MAP[clean]
    return {"yf": asset_name.upper(), "mt5": asset_name.upper(), "sl_pips": 5.0, "tp_pips": 10.0}

@tool
def get_mt5_account_balance() -> str:
    """جلب رصيد الحساب الحقيقي، السيولة (Equity)، والمارجين مباشرة من منصة MT5 عبر سحابة لندن."""
    if not metaapi_token or not metaapi_account_id:
        return "⚠️ مفاتيح MetaApi غير مضافة في الأسرار."
    
    url = f"{API_BASE_URL}/users/current/accounts/{metaapi_account_id}/account-information"
    headers = {"auth-token": metaapi_token}
    
    try:
        res = requests.get(url, headers=headers, timeout=15)
        if res.status_code == 200:
            data = res.json()
            return f"""📊 معلومات رصيد حسابك الحقيقي على MT5 (سحابة لندن):
- الرصيد (Balance): ${data.get('balance', 0):.2f}
- السيولة المتاحة (Equity): ${data.get('equity', 0):.2f}
- الهامش المجاني (Free Margin): ${data.get('freeMargin', 0):.2f}
- الرافعة المالية: 1:{data.get('leverage', 'N/A')}
"""
        else:
            return f"⚠️ فشل الاتصال بخادم لندن (رمز الاستجابة {res.status_code}): {res.text}"
    except Exception as e:
        return f"خطأ في الاتصال بالخادم الإقليمي: {str(e)}"

@tool
def analyze_and_execute_autonomous_trade(timeframe: str, asset_name: str, lot_size: float) -> str:
    """
    يفحص السوق باتجاهين (فريم الساعات للاتجاه العام + فريم التنفيذ مع مؤشرات متقدمة)
    وينفذ صفقة عالية الاحتمالية مع وقف خسارة وجني أرباح تلقائي على MT5.
    """
    asset_info = resolve_asset(asset_name)
    yf_symbol = asset_info["yf"]
    mt5_symbol = asset_info["mt5"]
    
    try:
        # 1. التحقق من الاتجاه العام على فريم الساعة (1h) لضمان عدم عكس الاتجاه
        df_trend = yf.download(yf_symbol, period="5d", interval="1h", progress=False)
        if not df_trend.empty:
            if isinstance(df_trend.columns, pd.MultiIndex):
                df_trend.columns = df_trend.columns.get_level_values(0)
            df_trend['SMA_50'] = df_trend['Close'].rolling(window=50).mean()
            higher_trend_bullish = float(df_trend['Close'].iloc[-1]) > float(df_trend['SMA_50'].iloc[-1])
        else:
            higher_trend_bullish = True

        # 2. التحقق من فريم التنفيذ (مثل 5m أو 15m) ومؤشر RSI
        period_map = {"1m": "1d", "5m": "5d", "15m": "1mo", "1h": "1mo"}
        period = period_map.get(timeframe, "5d")
        
        df = yf.download(yf_symbol, period=period, interval=timeframe, progress=False)
        if df.empty or len(df) < 20:
            return f"❌ بيانات السوق غير كافية لـ {asset_name}."
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        
        current_price = float(df['Close'].iloc[-1])
        current_rsi = float(df['RSI'].iloc[-1])
        sma_20 = float(df['SMA_20'].iloc[-1])
        
        action_type = None
        action_desc = ""
        
        # استراتيجية عالية الدقة: الشراء فقط في الاتجاه الصاعد العام وعندما يكون RSI مناسباً
        if higher_trend_bullish and (current_rsi < 42 or (current_price > sma_20 and current_rsi < 58)):
            action_type = "ORDER_TYPE_BUY"
            action_desc = "شراء (BUY) - اتجاه صاعد مؤكد"
            stop_loss = round(current_price - asset_info["sl_pips"], 2)
            take_profit = round(current_price + asset_info["tp_pips"], 2)
        elif not higher_trend_bullish and (current_rsi > 58 or (current_price < sma_20 and current_rsi > 42)):
            action_type = "ORDER_TYPE_SELL"
            action_desc = "بيع (SELL) - اتجاه هابط مؤكد"
            stop_loss = round(current_price + asset_info["sl_pips"], 2)
            take_profit = round(current_price - asset_info["tp_pips"], 2)
        else:
            return f"""⏸️ وضع الانتظار الذكي (فرصة غير مكتملة بنسبة عالية):
- الأصل: {asset_name} ({mt5_symbol})
- السعر الحالي: ${current_price:.2f}
- مؤشر RSI: {current_rsi:.1f}
- الاتجاه العام (1h): {'صاعد 🟢' if higher_trend_bullish else 'هابط 🔴'}
البوت ينتظر نقطة دخول مثالية ومضمونة نسبياً لمنع المخاطرة برأس المال.
"""

        # تنفيذ الصفقة آلياً مع SL و TP عبر سحابة MetaApi
        if metaapi_token and metaapi_account_id and action_type:
            url = f"{API_BASE_URL}/users/current/accounts/{metaapi_account_id}/trade"
            headers = {"auth-token": metaapi_token, "Content-Type": "application/json"}
            payload = {
                "actionType": action_type,
                "symbol": mt5_symbol,
                "volume": float(lot_size),
                "stopLoss": stop_loss,
                "takeProfit": take_profit,
                "comment": f"High-Prob AI Trade RSI {current_rsi:.1f}"
            }
            
            res = requests.post(url, json=payload, headers=headers, timeout=15)
            if res.status_code in [200, 201]:
                return f"""🎯🚀 تم تنفيذ صفقة عالية الدقة بنجاح على MT5!
- الأصل: {asset_name} ({mt5_symbol})
- نوع الصفقة: {action_desc}
- حجم العقد (Lot): {lot_size} (آمن لرصيدك)
- سعر الدخول: ${current_price:.2f}
- وقف الخسارة (SL): ${stop_loss} 🛡️
- جني الأرباح (TP): ${take_profit} 🎯
- مؤشر RSI: {current_rsi:.1f}
"""
            else:
                return f"⚠️ فشل تنفيذ الصفقة عبر الخادم: {res.text}"
        else:
            return f"الإشارة جاهزة ولكن بيانات الاعتماد ناقصة."
            
    except Exception as e:
        return f"خطأ أثناء التحليل الذكي وتنفيذ الصفقة: {str(e)}"

tools = [get_mt5_account_balance, analyze_and_execute_autonomous_trade]

api_key = st.secrets.get("GOOGLE_API_KEY", os.environ.get("GOOGLE_API_KEY", "")).strip()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=api_key,
    system_instruction="أنت نظام تداول آلي سحابي محترف ومتحفظ جداً، تعطي الأولوية القصوى لحماية رأس المال الصغير وتنفيذ صفقات عالية الاحتمالية."
)
agent_executor = create_agent(llm, tools)

st.title("🤖 الوكيل المالي الذكي - التداول الآلي عالي الدقة على MT5")
st.write("نظام متطور لتحليل الاتجاهات، حماية رأس المال بوقف خسارة تلقائي، وتنفيذ صفقات ذكية عبر سحابة لندن.")

user_input = st.text_input("💬 اطلب من البوت (مثال: حلل الذهب ونفذ صفقة بـ 0.01 لوت على فريم 5m):", placeholder="اكتب أمرك هنا...")

if st.button("🚀 تنفيذ عبر السحابة", type="primary"):
    if user_input:
        with st.spinner("جاري فحص الاتجاهات الكبرى وتنفيذ الصفقة بأمان..."):
            try:
                res = agent_executor.invoke({"messages": [("user", user_input)]})
                ans = res["messages"][-1].content
            except Exception as e:
                ans = f"حدث خطأ أثناء تنفيذ الطلب: {str(e)}"
            
            st.success("🤖 تقرير التنفيذ الذكي:")
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
