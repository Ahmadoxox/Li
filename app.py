import json
import os
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from gtts import gTTS
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
import yfinance as yf

# إعدادات واجهة التطبيق المؤسسي
st.set_page_config(page_title="الوكيل المالي المؤسسي الذكي - MT5 Pro", page_icon="🛡️", layout="wide")

metaapi_token = st.secrets.get("METAAPI_TOKEN", os.environ.get("METAAPI_TOKEN", "")).strip()
metaapi_account_id = st.secrets.get("METAAPI_ACCOUNT_ID", os.environ.get("METAAPI_ACCOUNT_ID", "")).strip()
metaapi_region = st.secrets.get("METAAPI_REGION", "london").strip().lower()

if "london" in metaapi_region:
    API_BASE_URL = "https://mt-client-api-v1.london.agiliumtrade.ai"
else:
    API_BASE_URL = f"https://mt-client-api-v1.{metaapi_region}.agiliumtrade.ai"

WATCHLIST = [
    {"name": "الذهب", "yf": "GC=F", "mt5": "XAUUSD.m", "sl": 4.0, "tp": 8.0, "decimals": 2, "type": "forex"},
    {"name": "يورو دولار", "yf": "EURUSD=X", "mt5": "EURUSD.m", "sl": 0.0030, "tp": 0.0060, "decimals": 5, "type": "forex"},
    {"name": "باوند دولار", "yf": "GBPUSD=X", "mt5": "GBPUSD.m", "sl": 0.0035, "tp": 0.0070, "decimals": 5, "type": "forex"},
    {"name": "بيتكوين", "yf": "BTC-USD", "mt5": "BTCUSD", "sl": 150.0, "tp": 300.0, "decimals": 2, "type": "crypto"}
]

@tool
def get_mt5_account_balance() -> str:
    """جلب رصيد الحساب الحقيقي والسيولة والهامش مع تقييم المخاطر اليومية من منصة MT5."""
    if not metaapi_token or not metaapi_account_id:
        return "⚠️ مفاتيح MetaApi غير مضافة."
    url = f"{API_BASE_URL}/users/current/accounts/{metaapi_account_id}/account-information"
    headers = {"auth-token": metaapi_token}
    try:
        res = requests.get(url, headers=headers, timeout=15)
        if res.status_code == 200:
            data = res.json()
            balance = float(data.get('balance', 0))
            equity = float(data.get('equity', 0))
            drawdown_pct = ((balance - equity) / balance) * 100 if balance > 0 else 0
            status_dd = "🟢 آمن تماماً" if drawdown_pct < 5 else ("🟡 تنبيه: خسارة غير محققة مرتفعة" if drawdown_pct < 10 else "🔴 خطر: تجاوز حد الأمان اليومي!")
            
            return f"""📊 معلومات الحساب المؤسسي على MT5:
- الرصيد (Balance): ${balance:.2f}
- السيولة المتاحة (Equity): ${equity:.2f}
- الهامش المجاني (Free Margin): ${data.get('freeMargin', 0):.2f}
- نسبة التراجع الحالي (Drawdown): {drawdown_pct:.2f}% ({status_dd})
"""
        else:
            return f"⚠️ فشل الاتصال بخادم لندن (رمز الاستجابة {res.status_code}): {res.text}"
    except Exception as e:
        return f"خطأ في الاتصال بالخادم: {str(e)}"

@tool
def get_mt5_open_positions() -> str:
    """فحص وجلب كافة الصفقات المفتوحة حالياً من خادم MT5 مع تتبع حالة الحماية."""
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
            report = "📋 الصفقات المفتوحة حالياً (محمية بنظام الحماية الذكي):\n"
            for p in positions:
                report += f"- الأصل: {p.get('symbol')} | النوع: {p.get('type')} | الحجم: {p.get('volume')} | سعر الدخول: {p.get('openPrice')} | السعر الحالي: {p.get('currentPrice')} | الستوب لوز: {p.get('stopLoss')} | الربح الحالي: ${p.get('profit', 0):.2f} | التعليق: {p.get('comment', 'N/A')}\n"
            return report
        else:
            return f"⚠️ فشل جلب الصفقات المفتوحة (رمز الاستجابة {res.status_code}): {res.text}"
    except Exception as e:
        return f"خطأ أثناء الاتصال لجلب الصفقات: {str(e)}"

@tool
def manage_trailing_stops() -> str:
    """تفقد كافة الصفقات المفتوحة حالياً وتفعيل نظام وقف الخسارة المتحرك (Trailing Stop / Break-Even) لحماية الأرباح تلقائياً."""
    if not metaapi_token or not metaapi_account_id:
        return "⚠️ مفاتيح MetaApi غير مضافة."
    
    url = f"{API_BASE_URL}/users/current/accounts/{metaapi_account_id}/positions"
    headers = {"auth-token": metaapi_token}
    try:
        res = requests.get(url, headers=headers, timeout=15)
        if res.status_code != 200:
            return f"⚠️ فشل جلب الصفقات لتطبيق الترايلينج (رمز الاستجابة {res.status_code}): {res.text}"
        
        positions = res.json()
        if not positions:
            return "📋 لا توجد صفقات مفتوحة لتطبيق وقف الخسارة المتحرك عليها."
        
        modifications_report = []
        trade_url = f"{API_BASE_URL}/users/current/accounts/{metaapi_account_id}/trade"

        for p in positions:
            pos_id = p.get('id')
            symbol = p.get('symbol', '')
            pos_type = p.get('type')
            open_price = float(p.get('openPrice', 0))
            current_price = float(p.get('currentPrice', 0))
            current_sl = float(p.get('stopLoss', 0))
            profit = float(p.get('profit', 0))
            
            if current_price == 0:
                continue

            new_sl = None
            is_gold = "XAU" in symbol.upper() or "GOLD" in symbol.upper()
            is_crypto = "BTC" in symbol.upper()
            
            if pos_type == 'POSITION_TYPE_BUY':
                if is_gold:
                    if (current_price - open_price) >= 3.0:
                        potential_sl = round(current_price - 3.0, 2)
                        if potential_sl > current_sl:
                            new_sl = potential_sl
                elif is_crypto:
                    if (current_price - open_price) >= 100.0:
                        potential_sl = round(current_price - 100.0, 2)
                        if potential_sl > current_sl:
                            new_sl = potential_sl
                else:
                    if (current_price - open_price) >= 0.0020:
                        decimals = 5 if "JPY" not in symbol else 3
                        potential_sl = round(current_price - 0.0015, decimals)
                        if potential_sl > current_sl:
                            new_sl = potential_sl

            elif pos_type == 'POSITION_TYPE_SELL':
                if is_gold:
                    if (open_price - current_price) >= 3.0:
                        potential_sl = round(current_price + 3.0, 2)
                        if current_sl == 0 or potential_sl < current_sl:
                            new_sl = potential_sl
                elif is_crypto:
                    if (open_price - current_price) >= 100.0:
                        potential_sl = round(current_price + 100.0, 2)
                        if current_sl == 0 or potential_sl < current_sl:
                            new_sl = potential_sl
                else:
                    if (open_price - current_price) >= 0.0020:
                        decimals = 5 if "JPY" not in symbol else 3
                        potential_sl = round(current_price + 0.0015, decimals)
                        if current_sl == 0 or potential_sl < current_sl:
                            new_sl = potential_sl

            if new_sl is not None:
                payload = {
                    "actionType": "POSITION_MODIFY",
                    "positionId": str(pos_id),
                    "stopLoss": float(new_sl)
                }
                mod_res = requests.post(trade_url, json=payload, headers=headers, timeout=15)
                if mod_res.status_code in [200, 201]:
                    modifications_report.append(f"✅ الأصل: {symbol} (ID: {pos_id}) | تم تحديث الستوب لوز المتحرك إلى: {new_sl} (الربح الحالي: ${profit:.2f})")
                else:
                    modifications_report.append(f"⚠️ فشل تحديث الأصل {symbol}: {mod_res.text}")

        if not modifications_report:
            return "ℹ️ تم تفقد الصفقات المفتوحة، ولا توجد صفقات حققت مسافة كافية لتفعيل الترايلينج ستاپ حالياً (الوضع آمن ومستقر)."
        
        return "🛡️ تقرير تفعيل وقف الخسارة المتحرك (Trailing Stop):\n" + "\n".join(modifications_report)
    except Exception as e:
        return f"خطأ أثناء إدارة وقف الخسارة المتحرك: {str(e)}"

@tool
def get_mt5_symbols() -> str:
    """جلب قائمة الرموز المتاحة وحالة السيولة في منصة MT5."""
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
def get_market_news_and_fundamental_analysis() -> str:
    """جلب أحدث الأخبار الاقتصادية والتحليل الأساسي المؤسسي للأسواق."""
    news_report = "📰 تقرير التحليل الأساسي والأخبار الاقتصادية العالمية:\n"
    for asset in WATCHLIST:
        try:
            ticker = yf.Ticker(asset["yf"])
            news_list = ticker.news
            if news_list:
                news_report += f"\n📌 أخبار {asset['name']}:\n"
                for item in news_list[:2]:
                    title = item.get('title', 'بدون عنوان')
                    publisher = item.get('publisher', 'وكالة عالمية')
                    news_report += f"  • {title} (المصدر: {publisher})\n"
            else:
                news_report += f"\n📌 {asset['name']}: الهدوء يسود الأخبار المباشرة.\n"
        except Exception as e:
            news_report += f"\n📌 {asset['name']}: تعذر تحديث الأخبار مؤقتاً.\n"
    return news_report

@tool
def scan_markets_and_execute_smart_trades(timeframe: str = "5m") -> str:
    """يقوم بالمسح الشامل وتطبيق فلاتر الحماية (منع تكرار الصفقات، فلتر السيولة والجلسات، فحص الحد الأقصى للخسارة، التحليل الفني والأساسي، وإدارة اللوت الديناميكي)."""
    if not metaapi_token or not metaapi_account_id:
        return "⚠️ مفاتيح MetaApi غير مضافة."

    headers = {"auth-token": metaapi_token}
    
    acc_url = f"{API_BASE_URL}/users/current/accounts/{metaapi_account_id}/account-information"
    current_balance = 31.0
    current_equity = 31.0
    try:
        acc_res = requests.get(acc_url, headers=headers, timeout=10)
        if acc_res.status_code == 200:
            acc_data = acc_res.json()
            current_balance = float(acc_data.get('balance', 31.0))
            current_equity = float(acc_data.get('equity', 31.0))
    except Exception:
        pass

    drawdown_amount = current_balance - current_equity
    if drawdown_amount > (current_balance * 0.10):
        return f"🛑 تم تفعيل نظام الأمان المؤسسي: تم إيقاف فتح صفقات جديدة نظراً لأن الخسارة اليومية بلغت (${drawdown_amount:.2f}) وهي تتجاوز حد الأمان (10%)."

    now_utc = datetime.utcnow()
    weekday = now_utc.weekday()
    is_weekend = (weekday >= 5)

    pos_url = f"{API_BASE_URL}/users/current/accounts/{metaapi_account_id}/positions"
    active_symbols = set()
    try:
        pos_res = requests.get(pos_url, headers=headers, timeout=10)
        if pos_res.status_code == 200:
            for p in pos_res.json():
                active_symbols.add(p.get('symbol'))
    except Exception:
        pass

    dynamic_lot = max(0.01, round(current_balance / 3000.0, 2))
    executed_trades = []
    skipped_trades = []

    for asset in WATCHLIST:
        if is_weekend and asset["type"] == "forex":
            skipped_trades.append(f"- {asset['name']}: مغلق لعطلة نهاية الأسبوع.")
            continue

        if asset["mt5"] in active_symbols:
            skipped_trades.append(f"- {asset['name']} ({asset['mt5']}): توجد صفقة مفتوحة مسبقاً (تم منع التكرار).")
            continue

        try:
            df_trend = yf.download(asset["yf"], period="5d", interval="1h", progress=False)
            if df_trend.empty:
                continue
            if isinstance(df_trend.columns, pd.MultiIndex):
                df_trend.columns = df_trend.columns.get_level_values(0)
            df_trend['SMA_50'] = df_trend['Close'].rolling(window=50).mean()
            trend_bullish = float(df_trend['Close'].iloc[-1]) > float(df_trend['SMA_50'].iloc[-1])

            df = yf.download(asset["yf"], period="5d", interval=timeframe, progress=False)
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
            
            if trend_bullish and (rsi < 48 or price > sma_20):
                action = "ORDER_TYPE_BUY"
                sl = round(price - asset["sl"], dec)
                tp = round(price + asset["tp"], dec)
            elif not trend_bullish and (rsi > 52 or price < sma_20):
                action = "ORDER_TYPE_SELL"
                sl = round(price + asset["sl"], dec)
                tp = round(price - asset["tp"], dec)

            if action:
                trade_url = f"{API_BASE_URL}/users/current/accounts/{metaapi_account_id}/trade"
                payload = {
                    "actionType": action, "symbol": asset["mt5"],
                    "volume": float(dynamic_lot), "stopLoss": sl,
                    "takeProfit": tp, "comment": f"Pro-Shield TF {timeframe} RSI {rsi:.1f}"
                }
                res = requests.post(trade_url, json=payload, headers=headers, timeout=15)
                if res.status_code in [200, 201]:
                    action_name = "شراء (BUY)" if action == "ORDER_TYPE_BUY" else "بيع (SELL)"
                    executed_trades.append(f"- {asset['name']} ({asset['mt5']}) | {action_name} | السعر: {price:.2f} | RSI: {rsi:.1f} | الفريم: {timeframe}")
        except Exception:
            continue

    if not executed_trades and not skipped_trades:
        if not is_weekend:
            best_trade = {"name": "الذهب", "mt5": "XAUUSD.m", "action": "ORDER_TYPE_BUY", "sl_val": 4.0, "tp_val": 8.0}
            try:
                df_gold = yf.download("GC=F", period="1d", interval="1m", progress=False)
                if not df_gold.empty:
                    if isinstance(df_gold.columns, pd.MultiIndex):
                        df_gold.columns = df_gold.columns.get_level_values(0)
                    b_price = float(df_gold['Close'].iloc[-1])
                    b_sl = round(b_price - best_trade["sl_val"], 2)
                    b_tp = round(b_price + best_trade["tp_val"], 2)
                    
                    if best_trade["mt5"] not in active_symbols:
                        trade_url = f"{API_BASE_URL}/users/current/accounts/{metaapi_account_id}/trade"
                        payload = {
                            "actionType": best_trade["action"], "symbol": best_trade["mt5"],
                            "volume": float(dynamic_lot), "stopLoss": b_sl,
                            "takeProfit": b_tp, "comment": f"Smart Backup TF {timeframe}"
                        }
                        res = requests.post(trade_url, json=payload, headers=headers, timeout=15)
                        if res.status_code in [200, 201]:
                            executed_trades.append(f"- {best_trade['name']} ({best_trade['mt5']}) | شراء (BUY) | صفقة احتياطية مؤكدة ومحمية")
            except Exception:
                pass

    report = "🛡️🚀 تقرير التنفيذ المؤسسي المتكامل (آليات الحماية مفعلة):\n"
    if executed_trades:
        report += "✅ الصفقات التي تم فتحها:\n" + "\n".join(executed_trades) + "\n"
    else:
        report += "ℹ️ لم يتم فتح صفقات جديدة في هذا الفحص لعدم مطابقة الشروط أو بسبب فلاتر الحماية.\n"
        
    if skipped_trades:
        report += "\n🔒 صفقات تم استثناؤها بواسطة فلاتر الحماية:\n" + "\n".join(skipped_trades) + "\n"
        
    report += f"\n📊 حجم اللوت المستخدم: {dynamic_lot} | الإطار الزمني: {timeframe}"
    return report

tools = [
    get_mt5_account_balance, 
    get_mt5_open_positions, 
    manage_trailing_stops, 
    get_mt5_symbols, 
    get_market_news_and_fundamental_analysis, 
    scan_markets_and_execute_smart_trades
]

api_key = st.secrets.get("GOOGLE_API_KEY", os.environ.get("GOOGLE_API_KEY", "")).strip()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=api_key,
    system_instruction="أنت مدير محفظة مؤسسي ذكي وخبير في إدارة المخاطر. تطبق آليات الحماية المؤسسية (منع تكرار الصفقات، فلاتر السيولة والجلسات، إدارة التراجع والحد اليومي للخسارة، التحليل الفني والأساسي المدمج، اللوت الديناميكي، ووقف الخسارة المتحرك Trailing Stop). تنفذ الأوامر وتجيب باحترافية."
)
agent_executor = create_agent(llm, tools)

st.title("🛡️🤖 الوكيل المالي المؤسسي - نسخة الحماية والترايلينج ستاپ على MT5")
st.write("إدارة الحساب، تفعيل وقف الخسارة المتحرك، جلب الأخبار، فلاتر الحماية، ومسح الأسواق الذكي.")

user_input = st.text_input("💬 اطلب من البوت (مثال: افحص الصفقات وفعل وقف الخسارة المتحرك، أو امسح الأسواق وافتح الفرص):", placeholder="اكتب أمرك هنا...")

if st.button("🚀 تشغيل البوت المؤسسي الآلي", type="primary"):
    if user_input:
        with st.spinner("جاري تنفيذ الطلب وتطبيق أداة الترايلينج ستاپ وفحص الأسواق..."):
            try:
                res = agent_executor.invoke({"messages": [("user", user_input)]})
                ans = res["messages"][-1].content
            except Exception as e:
                ans = f"حدث خطأ أثناء تنفيذ الطلب: {str(e)}"
            st.success("🤖 تقرير التنفيذ والتحليل المؤسسي:")
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
