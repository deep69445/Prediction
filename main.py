import streamlit as st
from data_loader import fetch_stock_data
from predictor import train_model, predict_next_day
from visualizer import plot_recent_prices, plot_with_moving_averages, plot_actual_vs_predicted
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time

# ------------------------------
# ⚙️ Page Configuration
# ------------------------------
st.set_page_config(page_title="📊 Stock Insight Dashboard", layout="wide")
st.title("📈 Real-Time Stock Dashboard")

API_KEYS = ["UXS0VUB2WCZ5HTYH", "YOUR_SECOND_API_KEY"]
api_counter = 0

def get_next_api_key():
    global api_counter
    key = API_KEYS[api_counter % len(API_KEYS)]
    api_counter += 1
    return key

# ------------------------------
# 🔘 Sidebar Ticker Selector + Manual Refresh
# ------------------------------
st.sidebar.header("📊 Stock Selector")
stock_list = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", ""]
selected_stock = st.sidebar.selectbox("Select a stock", stock_list, index=len(stock_list)-1)
refresh = st.sidebar.button("🔄 Refresh Now")

@st.cache_data(ttl=600)
def load_stock(symbol: str):
    try:
        df = fetch_stock_data(symbol, api_key=get_next_api_key())
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# ------------------------------
# 📊 Display View
# ------------------------------
if selected_stock == "":
    st.subheader("📈 Market Overview")
    st.info("Compare daily performance, peak fluctuations, and trading activity across major stocks.")

    overview_closes = {}
    overview_volumes = {}
    overview_peaks = []
    total_trades = 0

    line_df = pd.DataFrame()

    for symbol in stock_list[:-1]:
        df = load_stock(symbol)
        time.sleep(12)  # Delay to avoid hitting rate limit

        if df is not None and not df.empty:
            df = df.sort_values("Datetime")
            df['Symbol'] = symbol
            today_df = df[df['Datetime'].dt.date == df['Datetime'].max().date()]
            overview_closes[symbol] = df['Close'].iloc[-1]
            volume_today = today_df['Volume'].sum()
            overview_volumes[symbol] = volume_today
            total_trades += volume_today

            if not today_df.empty:
                peak_row = today_df.loc[today_df['High'].idxmax()]
                dip_row = today_df.loc[today_df['Low'].idxmin()]
                overview_peaks.append({
                    'Stock': symbol,
                    'Peak Time': peak_row['Datetime'],
                    'Peak Price': peak_row['High'],
                    'Dip Time': dip_row['Datetime'],
                    'Dip Price': dip_row['Low']
                })

            line_df = pd.concat([line_df, df[['Datetime', 'Close', 'Symbol']].tail(100)])

    if not line_df.empty:
        fig = px.line(
            line_df,
            x="Datetime",
            y="Close",
            color="Symbol",
            title="🔼 Price Fluctuation of Major Stocks (Last 100 Records)"
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        col1.metric("Total Trades Today", f"{total_trades:,}")
        top_stock = max(overview_volumes, key=overview_volumes.get)
        col2.metric("Top Performer (Trades)", top_stock)

        # Enhanced 3D Pie Chart for Volume Distribution
        pie_fig = go.Figure(data=[
            go.Pie(
                labels=list(overview_volumes.keys()),
                values=list(overview_volumes.values()),
                hole=0.3,
                pull=[0.05]*len(overview_volumes),
                marker=dict(line=dict(color='#000000', width=2))
            )
        ])
        pie_fig.update_traces(textinfo='percent+label')
        pie_fig.update_layout(title="🌐 Trading Volume Distribution (3D Effect)", height=500)
        st.plotly_chart(pie_fig, use_container_width=True)

        # Peak Fluctuations
        if overview_peaks:
            st.subheader("🔢 Market Movement Highlights")
            peak_df = pd.DataFrame(overview_peaks)
            st.dataframe(peak_df)

else:
    df = load_stock(selected_stock)
    if refresh:
        st.cache_data.clear()
        df = load_stock(selected_stock)

    if df is not None and not df.empty:
        st.subheader(f"📌 {selected_stock} - Latest Data")

        latest = df.iloc[-1]
        last_updated = latest['Datetime'].strftime('%Y-%m-%d %H:%M')
        df_today = df[df['Datetime'].dt.date == latest['Datetime'].date()]
        peak_price = df_today['High'].max() if not df_today.empty else latest['High']

        model, last_pred, mae, rmse = train_model(df)
        next_close = predict_next_day(model, df)

        col1, col2, col3 = st.columns(3)
        col1.metric("Last Updated", last_updated)
        col2.metric("Peak Price Today", f"${peak_price:.2f}")
        col3.metric("🔮 Predicted Next Close", f"${next_close:.2f}")

        st.plotly_chart(plot_recent_prices(df, selected_stock), use_container_width=True)

        df_ma = df.copy()
        df_ma = df_ma.sort_values("Datetime")
        df_ma = df_ma.tail(100)
        df_ma = df_ma.reset_index(drop=True)
        df_ma['MA_3'] = df_ma['Close'].rolling(3).mean()
        df_ma['MA_7'] = df_ma['Close'].rolling(7).mean()

        st.subheader("🔢 Moving Averages (MA)")
        st.caption("Short-term MA (3-day) helps spot recent price momentum, while the 7-day MA smoothens volatility to show trend direction.")
        st.plotly_chart(plot_with_moving_averages(df_ma, selected_stock), use_container_width=True)

        st.subheader("📉 Actual vs Predicted Close")
        df_feat = df.copy()
        df_feat['MA_3'] = df_feat['Close'].rolling(3).mean()
        df_feat['MA_7'] = df_feat['Close'].rolling(7).mean()
        df_feat['Lag_1'] = df_feat['Close'].shift(1)
        df_feat['Lag_2'] = df_feat['Close'].shift(2)
        df_feat['Lag_3'] = df_feat['Close'].shift(3)

        df_feat = df_feat.dropna()
        X = df_feat[['MA_3', 'MA_7', 'Lag_1', 'Lag_2', 'Lag_3']]
        y_true = df_feat['Close']
        y_pred = model.predict(X)

        st.plotly_chart(plot_actual_vs_predicted(y_true, y_pred), use_container_width=True)