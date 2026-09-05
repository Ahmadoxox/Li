import json
import os
import streamlit as st
import matplotlib.pyplot as plt
from gtts import gTTS
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
import yfinance as yf

# إعدادات الصفحة والتصميم
st.set_page_config(page_title="الوكيل المالي الذكي", page_icon="📈", layout="wide")

PORTFOLIO_FILE = "portfolio.json"

ASSET_MAP = {
    "ذهب": "GC=F", "الذهب": "GC=F", "gold": "GC=F",
    "نفط": "CL=F", "النفط": "CL=F", "برنت": "BZ=F",
    "فضة": "SI=F", "الفضة": "SI=F",
    "غاز": "NG=F", "الغاز": "NG=F",
    "بيتكوين": "BTC-USD", "البيتكوين": "BTC-USD",
    "إيثريوم": "ETH-USD",
    "نازداك": "^IXIC", "اس اند بي": "^GSPC",
    "أبل": "AAPL", "تيسلا": "TSLA", "أنفيديا": "NVDA", "انفيديا": "NVDA",
    "يورو دولار": "EURUSD=X"
}

def resolve_ticker(asset_name: str) -> str:
    clean_name = asset_name.strip().lower()
    return ASSET_MAP.get(clean_name, asset_name.upper())

def load_portfolio():
    if os.path.exists(PORTFOLIO_FILE):
        with open(PORTFOLIO_FILE, "r") as f:
            return json.load(f)
    initial_state = {
        "cash_usd": 10000.0,
        "holdings": {"GC=F": 1.0, "BTC-USD": 0.02}
    }
    save_portfolio(initial_state)
    return initial_state

def save_portfolio(data):
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(data, f, indent=4)

portfolio = load_portfolio()

@tool
def get_market_price(asset_name: str) -> str:
    """جلب السعر اللحظي لأي أصل (ذهب، نفط، أسهم، معادن، فوركس، كريبتو)."""
    ticker = resolve_ticker(asset_name)
    try:
        data = yf.Ticker(ticker)
        price = data.fast_info.last_price
        return f"سعر {asset_name} ({ticker}) الحالي هو: ${price:.2f}"
    except Exception as e:
        return f"خطأ في جلب السعر: {str(e)}"

@tool
def get_market_news(asset_name: str) -> str:
    """جلب أحدث الأخبار والتحليلات المؤثرة على سوق معين."""
    ticker = resolve_ticker(asset_name)
    try:
        data = yf.Ticker(ticker)
        news_items = data.news
        if not news_items:
            return f"لا توجد أخبار عاجلة متوفرة حالياً لـ {asset_name}."
        
        summary = f"📰 أحدث الأخبار المؤثرة على سوق {asset_name}:\n"
        for item in news_items[:3]:
            summary += f"- {item.get('title', '')} (المصدر: {item.get('publisher', '')})\n"
        return summary
    except Exception as e:
        return f"خطأ في جلب الأخبار: {str(e)}"

@tool
def buy_asset_by_percentage(asset_name: str, percentage: float) -> str:
    """شراء أصل بنسبة مئوية من الكاش المتاح."""
    global portfolio
    ticker = resolve_ticker(asset_name)
    try:
        if percentage <= 0 or percentage > 100:
            return "النسبة يجب أن تكون بين 1 و 100."
        
        amount = (portfolio["cash_usd"] * percentage) / 100.0
        if amount < 1:
            return "السيولة المتاحة غير كافية."

        data = yf.Ticker(ticker)
        price = data.fast_info.last_price
        units = amount / price

        portfolio["cash_usd"] -= amount
        portfolio["holdings"][ticker] = portfolio["holdings"].get(ticker, 0) + units
        save_portfolio(portfolio)

        return f"تم شراء {units:.4f} وحدة من {asset_name} بقيمة ${amount:.2f} ({percentage}% من الكاش). المتبقي: ${portfolio['cash_usd']:.2f}"
    except Exception as e:
        return f"خطأ في العملية: {str(e)}"

@tool
def sell_asset_by_percentage(asset_name: str, percentage: float) -> str:
    """بيع نسبة مئوية من أصل مملوك."""
    global portfolio
    ticker = resolve_ticker(asset_name)
    try:
        current_units = portfolio["holdings"].get(ticker, 0)
        if current_units <= 0:
            return f"أنت لا تمتلك رصيداً من {asset_name}."

        units_to_sell = (current_units * percentage) / 100.0
        data = yf.Ticker(ticker)
        price = data.fast_info.last_price
        sale_val = units_to_sell * price

        portfolio["holdings"][ticker] -= units_to_sell
        if portfolio["holdings"][ticker] <= 1e-6:
            del portfolio["holdings"][ticker]

        portfolio["cash_usd"] += sale_val
        save_portfolio(portfolio)

        return f"تم بيع {units_to_sell:.4f} وحدة من {asset_name} بقيمة ${sale_val:.2f} ({percentage}% من الممتلكات). الكاش الجديد: ${portfolio['cash_usd']:.2f}"
    except Exception as e:
        return f"خطأ في البيع: {str(e)}"

@tool
def get_portfolio_valuation() -> str:
    """عرض تقييم المحفظة المالي الكامل والسيولة المتاحة."""
    total_value = portfolio["cash_usd"]
    report = f"💵 الكاش المتاح: ${portfolio['cash_usd']:.2f}\n📦 الأصول المملوكة:\n"

    for ticker, units in portfolio["holdings"].items():
        try:
            price = yf.Ticker(ticker).fast_info.last_price
            val = units * price
            total_value += val
            report += f"- {ticker}: {units:.4f} وحدة | السعر: ${price:.2f} | القيمة: ${val:.2f}\n"
        except:
            report += f"- {ticker}: {units:.4f} وحدة\n"

    report += f"\n💰 صافي قيمة المحفظة: ${total_value:.2f}"
    return report

tools = [
    get_market_price,
    get_market_news,
    buy_asset_by_percentage,
    sell_asset_by_percentage,
    get_portfolio_valuation
]

api_key = os.environ.get("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=api_key,
    system_instruction="أنت خبير تداول ومحلل أسواق مالية. استخدم الأدوات لجلب الأسعار والأخبار الحقيقية وتصرف بدقة في صفقات الشراء والبيع والتحليل."
)
agent_executor = create_agent(llm, tools)

st.title("🌐 الوكيل المالي الذكي")
st.write("تداول + تحليل أخبار + رسم بياني للمحفظة")

user_input = st.text_input("💬 اكتب الأمر أو الاستفسار هنا:", placeholder="مثال: اشترِ بـ 20% كاش ذهب...")

if st.button("🚀 تنفيذ فوراً", type="primary"):
    if user_input:
        with st.spinner("جاري تحليل الطلب والتنفيذ..."):
            res = agent_executor.invoke({"messages": [("user", user_input)]})
            ans = res["messages"][-1].content
            
            st.success("🤖 التقرير والرد:")
            st.write(ans)
            
            # الرد الصوتي
            audio_file = "ans.mp3"
            gTTS(text=ans, lang="ar").save(audio_file)
            st.audio(audio_file)
            
            # الرسم البياني
            labels = ["الكاش"]
            values = [portfolio["cash_usd"]]
            for ticker, units in portfolio["holdings"].items():
                try:
                    price = yf.Ticker(ticker).fast_info.last_price
                    values.append(units * price)
                    labels.append(ticker)
                except:
                    pass

            fig, ax = plt.subplots(figsize=(4, 4))
            ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=140)
            ax.set_title("توزيع المحفظة")
            st.pyplot(fig)
    else:
        st.warning("رجاء اكتب أمراً أولاً.")
import json
import os
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from gtts import gTTS
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
import yfinance as yf

# إعدادات الصفحة والتصميم
st.set_page_config(page_title="الوكيل المالي - البوت الآلي بالكامل", page_icon="🤖", layout="wide")

PORTFOLIO_FILE = "portfolio.json"

# قائمة الأصول المتاحة للتداول الذاتي
ASSET_MAP = {
    "ذهب": "GC=F", "الذهب": "GC=F", "gold": "GC=F",
    "نفط": "CL=F", "النفط": "CL=F", "برنت": "BZ=F",
    "فضة": "SI=F", "الفضة": "SI=F",
    "غاز": "NG=F", "الغاز": "NG=F",
    "بيتكوين": "BTC-USD", "البيتكوين": "BTC-USD",
    "إيثريوم": "ETH-USD",
    "نازداك": "^IXIC", "اس اند بي": "^GSPC",
    "أبل": "AAPL", "تيسلا": "TSLA", "أنفيديا": "NVDA", "انفيديا": "NVDA",
    "يورو دولار": "EURUSD=X"
}

def resolve_ticker(asset_name: str) -> str:
    clean_name = asset_name.strip().lower()
    return ASSET_MAP.get(clean_name, asset_name.upper())

def load_portfolio():
    if os.path.exists(PORTFOLIO_FILE):
        with open(PORTFOLIO_FILE, "r") as f:
            return json.load(f)
    initial_state = {
        "cash_usd": 10000.0,
        "holdings": {"GC=F": 1.0, "BTC-USD": 0.02}
    }
    save_portfolio(initial_state)
    return initial_state

def save_portfolio(data):
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(data, f, indent=4)

portfolio = load_portfolio()

@tool
def get_market_price(asset_name: str) -> str:
    """جلب السعر اللحظي لأي أصل."""
    ticker = resolve_ticker(asset_name)
    try:
        data = yf.Ticker(ticker)
        price = data.fast_info.last_price
        return f"سعر {asset_name} ({ticker}) الحالي هو: ${price:.2f}"
    except Exception as e:
        return f"خطأ في جلب السعر: {str(e)}"

@tool
def run_autonomous_trading_bot(timeframe: str, asset_name: str) -> str:
    """
    تشغيل البوت الذاتي بالكامل: يقوم بمسح السوق على الإطار الزمني المطلوب (مثل 1m, 5m, 15m, 1h, 1d)،
    حساب مؤشر RSI والمتوسطات، واتخاذ قرار الدخول (شراء/بيع) أو الانتظار تلقائياً بناءً على تحليل الشموع.
    """
    global portfolio
    ticker = resolve_ticker(asset_name)
    
    # تحديد نطاق البيانات بناءً على الشمعة المطلوبة
    period_map = {
        "1m": "5d",
        "5m": "5d",
        "15m": "1mo",
        "1h": "3mo",
        "1d": "6mo"
    }
    period = period_map.get(timeframe, "1mo")
    
    try:
        df = yf.download(ticker, period=period, interval=timeframe, progress=False)
        if df.empty or len(df) < 20:
            return f"❌ بيانات غير كافية لـ {asset_name} على الإطار الزمني {timeframe}."
        
        # تنظيف أعمدة ياهو فاينانس إذا كانت مالتيايندكس
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # حساب المؤشرات الفنية
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
        
        action_taken = "لا توجد صفقة (انتظار وتتالية مراقبة)"
        result_msg = ""
        
        # استراتيجية البوت الذاتي بناءً على شمعة الإطار الزمني والمؤشرات
        # شروط الشراء الذاتي: تشبع بيع RSI < 35 أو اختراق صاعد للسعر فوق المتوسطات
        if current_rsi < 35 or (current_price > sma_20 and sma_20 > sma_50 and current_rsi < 55):
            # تنفيذ صفقة شراء ذاتية بـ 10% من الكاش المتاح تلقائياً
            percentage = 10.0
            amount = (portfolio["cash_usd"] * percentage) / 100.0
            if amount >= 1:
                units = amount / current_price
                portfolio["cash_usd"] -= amount
                portfolio["holdings"][ticker] = portfolio["holdings"].get(ticker, 0) + units
                save_portfolio(portfolio)
                action_taken = "🚀 تنفيذ صفقة شراء ذاتية (Auto-Buy)"
                result_msg = f"تم شراء {units:.4f} وحدة بقيمة ${amount:.2f} تلقائياً بناءً على شمعة الـ {timeframe}."
            else:
                result_msg = "السيولة لا تكفي لتنفيذ الصفقة الذاتية."
                
        # شروط البيع الذاتي لجني الأرباح: تشبع شراء RSI > 75
        elif current_rsi > 75:
            current_units = portfolio["holdings"].get(ticker, 0)
            if current_units > 0:
                # بيع كامل الكمية أو جزء منها لجني الأرباح
                sale_val = current_units * current_price
                portfolio["cash_usd"] += sale_val
                portfolio["holdings"][ticker] = 0
                del portfolio["holdings"][ticker]
                save_portfolio(portfolio)
                action_taken = "💰 تنفيذ صفقة بيع ذاتية لجني الأرباح (Auto-Sell)"
                result_msg = f"تم إغلاق الصفقة وبيع كامل الكمية لجني الأرباح بقيمة ${sale_val:.2f} على شمعة الـ {timeframe}."
            else:
                action_taken = "انتظار (السعر مرتفع ولا توجد ممتلكات للبيع)"
                result_msg = "السوق في منطقة تشبع شراء، ولا توجد وحدات مملوكة للبيع حالياً."
        else:
            action_taken = "⏸️ وضع الانتظار والمراقبة (Holding/Waiting)"
            result_msg = f"السوق مستقر نسبياً على شمعة الـ {timeframe}، المؤشرات لا تعطي إشارة قوية للدخول الآن."

        return f"""🤖 تقرير البوت الذاتي (الإطار الزمني: {timeframe} - الأصل: {asset_name}):
- السعر الحالي: ${current_price:.2f}
- مؤشر RSI: {current_rsi:.1f}
- متوسط 20: ${sma_20:.2f} | متوسط 50: ${sma_50:.2f}
- القرار الذاتي المتخذ: {action_taken}
- التفاصيل التنفيذية: {result_msg}
- الكاش المتبقي في المحفظة: ${portfolio['cash_usd']:.2f}
"""
    except Exception as e:
        return f"خطأ في تشغيل البوت الذاتي: {str(e)}"

@tool
def get_portfolio_valuation() -> str:
    """عرض تقييم المحفظة المالي الكامل."""
    total_value = portfolio["cash_usd"]
    report = f"💵 الكاش المتاح: ${portfolio['cash_usd']:.2f}\n📦 الأصول المملوكة:\n"
    for ticker, units in portfolio["holdings"].items():
        try:
            price = yf.Ticker(ticker).fast_info.last_price
            val = units * price
            total_value += val
            report += f"- {ticker}: {units:.4f} وحدة | السعر: ${price:.2f} | القيمة: ${val:.2f}\n"
        except:
            report += f"- {ticker}: {units:.4f} وحدة\n"
    report += f"\n💰 صافي قيمة المحفظة: ${total_value:.2f}"
    return report

tools = [
    get_market_price,
    run_autonomous_trading_bot,
    get_portfolio_valuation
]

api_key = os.environ.get("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=api_key,
    system_instruction="""أنت نظام تداول آلي بالكامل (Fully Autonomous Trading AI).
عندما يطلب منك المستخدم تفعيل التداول الذاتي على إطار زمني معين (مثل شمعة دقيقة 1m، أو 5 دقائق 5m، أو ساعة 1h)، يجب عليك استخدام أداة (run_autonomous_trading_bot) لتنفيذ الفحص، واتخاذ القرار الذاتي (شراء، بيع، أو انتظار)، وإدارة المحفظة بالكامل دون تدخل بشري.
كن حاسماً، دقيقاً، واعمل كخبير خوارزميات تداول آلي."""
)
agent_executor = create_agent(llm, tools)

st.title("🤖 الوكيل المالي - منصة التداول الآلي الذاتي")
st.write("التداول التلقائي بناءً على الأطر الزمنية والشموع (1m, 5m, 15m, 1h...) دون تدخل منك")

user_input = st.text_input("💬 اطلب من البوت العمل ذاتياً (مثال: فوت تداول ذاتي على الذهب بشمعة الـ 5 دقائق، أو فحص البيتكوين بشمعة الدقيقة الواحدة):", placeholder="اكتب أمرك هنا...")

if st.button("🚀 تشغيل البوت الذاتي", type="primary"):
    if user_input:
        with st.spinner("البوت يقوم بمسح الشموع، فحص الإطار الزمني، واتخاذ الصفقات ذاتياً..."):
            res = agent_executor.invoke({"messages": [("user", user_input)]})
            ans = res["messages"][-1].content
            
            st.success("🤖 تقرير تنفيذ البوت الذاتي:")
            st.write(ans)
            
            # الرد الصوتي
            audio_file = "ans.mp3"
            gTTS(text=ans, lang="ar").save(audio_file)
            st.audio(audio_file)
            
            # الرسم البياني للمحفظة
            labels = ["الكاش"]
            values = [portfolio["cash_usd"]]
            for ticker, units in portfolio["holdings"].items():
                try:
                    price = yf.Ticker(ticker).fast_info.last_price
                    values.append(units * price)
                    labels.append(ticker)
                except:
                    pass

            fig, ax = plt.subplots(figsize=(4, 4))
            ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=140)
            ax.set_title("توزيع المحفظة بعد التداول الآلي")
            st.pyplot(fig)
    else:
        st.warning("رجاء اكتب أمراً أولاً.")
