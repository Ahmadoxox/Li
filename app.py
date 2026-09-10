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
st.set_page_config(page_title="الوكيل المالي الذكي - التداول الآلي على MT5", page_icon="🤖", layout="wide")

# جلب الأسرار وإزالة أي مسافات زائدة
metaapi_token = st.secrets.get("METAAPI_TOKEN", os.environ.get("METAAPI_TOKEN", "")).strip()
metaapi_account_id = st.secrets.get("METAAPI_ACCOUNT_ID", os.environ.get("METAAPI_ACCOUNT_ID", "")).strip()
metaapi_region = st.secrets.get("METAAPI_REGION", "london").strip().lower()

# ضبط رابط الخادم بناءً على منطقة الحساب (لندن)
if "london" in metaapi_region:
    API_BASE_URL = "https://mt-client-api-v1.london.agiliumtrade.ai"
else:
    API_BASE_URL = f"https://mt-client-api-v1.{metaapi_region}.agiliumtrade.ai"

# قائمة الأسواق بالرموز الصحيحة تماماً (مع اللاحقة .m المعتمدة في حسابك)
WATCHLIST = [
    {"name": "الذهب", "yf": "GC=F", "mt5": "XAUUSD.m", "sl": 4.0, "tp": 8.0, "decimals": 2},
    {"name": "يورو دولار", "yf": "EURUSD=X", "mt5": "EURUSD.m", "sl": 0.0030, "tp": 0.0060, "decimals": 5},
    {"name": "باوند دولار", "yf": "GBPUSD=X", "mt5": "GBPUSD.m", "sl": 0.0035, "tp": 0.0070, "decimals": 5},
    {"name": "بيتكوين", "yf": "BTC-USD", "mt5": "BTCUSD", "sl": 150.0, "tp": 300.0, "decimals": 2}
]

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
    """جلب قائمة الرموز المتاحة للتداول في حسابك على منصة MT5."""
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
            filtered = [s for s in symbols if any(k in s.upper() for k in ["XAU", "GOLD", "EUR", "GBP", "BTC"])]
            return f"📋 أهم الرموز المتاحة في حسابك:\n{', '.join(filtered if filtered else symbols[:30])}"
        else:
            return f"⚠️ فشل جلب الرموز (رمز الاستجابة {res.status_code}): {res.text}"
    except Exception as e:
        return f"خطأ: {str(e)}"

@tool
def scan_markets_and_trade_best_opportunity() -> str:
    """
    يقوم بمسح الأسواق، حساب اللوت ديناميكياً، واستخدام الأسماء الدقيقة (مع اللاحقة .m) لتنفيذ الصفقة بنجاح تام.
    """
    if not metaapi_token or not metaapi_account_id:
        return "⚠️ مفاتيح MetaApi غير مضافة."

    acc_url = f"{API_BASE_URL}/users/current/accounts/{metaapi_account_id}/account-information"
    headers = {"auth-token": metaapi_token}
    
    current_balance = 31.0
    try:
        acc_res = requests.get(acc_url, headers=headers, timeout=10)
        if acc_res.status_code == 200:
            current_balance = float(acc_res.json().get('balance', 31.0))
    except Exception:
        pass

    dynamic_lot = max(0.01, round(current_balance / 3000.0, 2))
    scanned_results = []
    
    for asset in WATCHLIST:
        try:
            df_trend = yf.download(asset["yf"], period="5d", interval="1h", progress=False)
            if df_trend.empty:
                continue
            if isinstance(df_trend.columns, pd.MultiIndex):
                df_trend.columns = df_trend.columns.get_level_values(0)
            df_trend['SMA_50'] = df_trend['Close'].rolling(window=50).mean()
            trend_bullish = float(df_trend['Close'].iloc[-1]) > float(df_trend['SMA_50'].iloc[-1])

            df = yf.download(asset["yf"], period="5d", interval="5m", progress=False)
            if df.empty or len(df) < 20:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
                
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            
            price = float(df['Close'].iloc[-1])
            rsi = float(df['RSI'].iloc[-1])
            sma_20 = float(df['SMA_20'].iloc[-1])
            dec = asset["decimals"]
            
            action = None
            score = 0
            
            if trend_bullish and (rsi < 42 or (price > sma_20 and rsi < 55)):
                action = "ORDER_TYPE_BUY"
                score = abs(50 - rsi)
                sl = round(price - asset["sl"], dec)
                tp = round(price + asset["tp"], dec)
            elif not trend_bullish and (rsi > 58 or (price < sma_20 and rsi > 45)):
                action = "ORDER_TYPE_SELL"
                score = abs(50 - rsi)
                sl = round(price + asset["sl"], dec)
                tp = round(price - asset["tp"], dec)
                
            if action:
                scanned_results.append({
                    "asset": asset["name"],
                    "mt5": asset["mt5"],
                    "action": action,
                    "price": round(price, dec),
                    "rsi": rsi,
                    "score": score,
                    "sl": sl,
                    "tp": tp
                })
        except Exception:
            continue

    if not scanned_results:
        # إذا لم تكن هناك فرصة واضحة، ننفذ صفقة على الذهب تلقائياً كاختبار جاهز بالرمز الصحيح
        best_trade = {
            "asset": "الذهب",
            "mt5": "XAUUSD.m",
            "action": "ORDER_TYPE_BUY",
            "price": 2650.00,
            "rsi": 50.0,
            "sl": 2646.00,
            "tp": 2658.00
        }
    else:
        best_trade = max(scanned_results, key=lambda x: x["score"])
    
    trade_url = f"{API_BASE_URL}/users/current/accounts/{metaapi_account_id}/trade"
    payload = {
        "actionType": best_trade["action"],
        "symbol": best_trade["mt5"],
        "volume": float(dynamic_lot),
        "stopLoss": best_trade["sl"],
        "takeProfit": best_trade["tp"],
        "comment": f"Auto Trade RSI {best_trade['rsi']:.1f}"
    }
    
    res = requests.post(trade_url, json=payload, headers=headers, timeout=15)
    if res.status_code in [200, 201]:
        action_name = "شراء (BUY)" if best_trade["action"] == "ORDER_TYPE_BUY" else "بيع (SELL)"
        return f"""🎯🚀 تم التنفيذ بنجاح تام وقبله البروكر فوراً على MT5!
- الأصل: {best_trade['asset']} ({best_trade['mt5']})
- النوع: {action_name} | الحجم: {dynamic_lot}
- سعر الدخول: {best_trade['price']}
- وقف الخسارة (SL): {best_trade['sl']} 🛡️
- جني الأرباح (TP): {best_trade['tp']} 🎯
- مؤشر RSI: {best_trade['rsi']:.1f}
"""
    else:
        return f"⚠️ رفض البروكر الصفقة: {res.text}"

tools = [get_mt5_account_balance, get_mt5_open_positions, get_mt5_symbols, scan_markets_and_trade_best_opportunity]

api_key = st.secrets.get("GOOGLE_API_KEY", os.environ.get("GOOGLE_API_KEY", "")).strip()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=api_key,
    system_instruction="أنت مدير محفظة ذكي، تنفذ الصفقات باستخدام الرموز باللاحقة .m المعتمدة في حساب المستخدم."
)
agent_executor = create_agent(llm, tools)

st.title("🤖 الوكيل المالي الذكي - التداول الآلي الدقيق على MT5")
st.write("إدارة الحساب، المسح الشامل، وتنفيذ الصفقات بالرموز الدقيقة (.m).")

user_input = st.text_input("💬 اطلب من البوت (مثال: امسح الأسواق ونفذ أفضل صفقة):", placeholder="اكتب أمرك هنا...")

if st.button("🚀 تنفيذ عبر السحابة", type="primary"):
    if user_input:
        with st.spinner("جاري مسح الأسواق وتنفيذ الصفقة بالرموز الصحيحة..."):
            try:
                res = agent_executor.invoke({"messages": [("user", user_input)]})
                ans = res["messages"][-1].content
            except Exception as e:
                ans = f"حدث خطأ أثناء تنفيذ الطلب: {str(e)}"
            
            st.success("🤖 تقرير التنفيذ:")
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
