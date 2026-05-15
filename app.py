import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from sklearn.metrics import r2_score

# --- App Configuration ---
st.set_page_config(page_title="Stock Trend & Projection Analyzer", layout="wide")
st.title("📈 Stock Trend & Future Projection Analyzer")

# --- Sidebar Inputs ---
st.sidebar.header("Configuration")
ticker = st.sidebar.text_input("Stock/ETF Ticker (e.g., SPY, AAPL, TSLA, ^GSPC)", value="SPY").upper()
p_years = st.sidebar.slider("P: Historical Years to analyze", min_value=1, max_value=20, value=5)

st.sidebar.markdown("---")
model_type = st.sidebar.radio("Fitting Model", options=["Polynomial", "Exponential"])
if model_type == "Polynomial":
    poly_order = st.sidebar.slider("n: Polynomial Order", min_value=1, max_value=5, value=2)
else:
    poly_order = 1 

st.sidebar.markdown("---")
y_years = st.sidebar.slider("Y: Future Years to predict", min_value=1, max_value=10, value=2)

st.sidebar.markdown("---")
st.sidebar.subheader("Bottom Panel Metrics")
show_fundamentals = st.sidebar.checkbox("Show Current Fundamentals Summary", value=True)

# ADDED "Closing Price" to the options here:
selected_indicators = st.sidebar.multiselect(
    "Select Historical Indicators to Plot:",
    ["Closing Price", "Volume", "30-Day Rolling Volatility", "RSI (14-Day)", "MACD"],
    default=["Closing Price", "Volume"]
)

@st.cache_data
def load_data(ticker, years):
    end_date = datetime.now()
    # Always pull a little extra for the 52-week calculations if P is small
    start_date = end_date - timedelta(days=max(years, 1) * 365.25 + 30) 
    df = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
    return df

@st.cache_data
def load_fundamentals(ticker):
    stock = yf.Ticker(ticker)
    return stock.info

if ticker:
    with st.spinner(f"Loading data for {ticker}..."):
        raw_df = load_data(ticker, p_years)
        
    if raw_df.empty:
        st.error(f"Could not load data for {ticker}. Please check the ticker symbol.")
    else:
        if isinstance(raw_df.columns, pd.MultiIndex):
            raw_df.columns = raw_df.columns.get_level_values(0)
            
        # Filter dataframe strictly to the requested P years for the main plot
        cutoff_date = pd.to_datetime(datetime.now() - timedelta(days=p_years * 365.25))
        df = raw_df[raw_df.index >= cutoff_date].copy()
        
        # 1. Prepare Historical Data
        df = df.reset_index()
        df['Days'] = (df['Date'] - df['Date'].min()).dt.days
        
        x_hist = df['Days'].values
        y_hist = df['Close'].values
        dates_hist = df['Date']
        current_price = y_hist[-1]
        
        # --- CALCULATE NEW INDICATORS ---
        df['Daily_Return'] = df['Close'].pct_change()
        df['Rolling_Vol'] = df['Daily_Return'].rolling(window=30).std() * np.sqrt(252) * 100
        
        delta = df['Close'].diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        ema_up = up.ewm(com=13, adjust=False).mean()
        ema_down = down.ewm(com=13, adjust=False).mean()
        rs = ema_up / ema_down
        df['RSI'] = 100 - (100 / (1 + rs))
        
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

        # 2. Fit the Model
        if model_type == "Polynomial":
            coeffs = np.polyfit(x_hist, y_hist, poly_order)
            poly_eq = np.poly1d(coeffs)
            y_fit = poly_eq(x_hist)
            model_name = f'Polynomial Fit (n={poly_order})'
        elif model_type == "Exponential":
            coeffs = np.polyfit(x_hist, np.log(y_hist), 1)
            B = coeffs[0]
            A = np.exp(coeffs[1])
            y_fit = A * np.exp(B * x_hist)
            model_name = 'Exponential Fit'

        r2 = r2_score(y_hist, y_fit)
        
        normalized_residuals = (y_hist - y_fit) / y_fit
        norm_std = np.std(normalized_residuals)
        
        hist_upper_bound = y_fit * (1 + norm_std)
        hist_lower_bound = y_fit * (1 - norm_std)
        
        st.subheader(f"Historical Analysis: {ticker}")
        col1, col2 = st.columns(2)
        col1.metric("$R^2$ (Fit Quality)", f"{r2:.4f}", help="Measures how closely the fitted line matches actual price.")
        col2.metric("Normalized $\sigma$ (Relative Volatility)", f"{(norm_std * 100):.2f}%", help="Standard deviation relative to the trendline.")

        # 3. Future Prediction
        last_date = df['Date'].max()
        last_day_val = x_hist[-1]
        
        future_days = y_years * 365
        x_future = np.linspace(last_day_val + 1, last_day_val + future_days, future_days)
        dates_future = [pd.to_datetime(last_date) + timedelta(days=int(d)) for d in range(1, future_days + 1)]
        
        if model_type == "Polynomial":
            average_scenario = poly_eq(x_future)
        elif model_type == "Exponential":
            average_scenario = A * np.exp(B * x_future)
            
        time_in_years = np.linspace(1/365, y_years, future_days)
        expanding_uncertainty = norm_std * np.sqrt(time_in_years)
        
        best_scenario = average_scenario * (1 + expanding_uncertainty)
        worst_scenario = average_scenario * (1 - expanding_uncertainty)

        # --- Plotting Main Chart ---
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=dates_hist, y=hist_upper_bound, mode='lines', line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=dates_hist, y=y_fit, mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(255, 0, 0, 0.2)', name='Red Shadow (+σ)'))

        fig.add_trace(go.Scatter(x=dates_hist, y=hist_lower_bound, mode='lines', line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=dates_hist, y=y_fit, mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0, 255, 0, 0.2)', name='Green Shadow (-σ)'))

        fig.add_trace(go.Scatter(x=dates_hist, y=y_hist, mode='lines', name='Actual Price', line=dict(color='black', width=1.5)))
        fig.add_trace(go.Scatter(x=dates_hist, y=y_fit, mode='lines', name=model_name, line=dict(color='blue', dash='dash')))

        fig.add_trace(go.Scatter(x=[last_date] + dates_future, y=[y_fit[-1]] + list(best_scenario), mode='lines', name='Best Scenario (+1σ)', line=dict(color='green', dash='dot', width=2)))
        fig.add_trace(go.Scatter(x=[last_date] + dates_future, y=[y_fit[-1]] + list(worst_scenario), mode='lines', name='Worst Scenario (-1σ)', line=dict(color='red', dash='dot', width=2)))
        fig.add_trace(go.Scatter(x=[last_date] + dates_future, y=[y_fit[-1]] + list(average_scenario), mode='lines', name='Average/Expected Trend', line=dict(color='grey', dash='dash', width=2)))

        # fig.add_vline(x=last_date.strftime('%Y-%m-%d'), line_width=2, line_dash="solid", line_color="black", annotation_text="TODAY", annotation_position="top left")

        fig.update_layout(title=f"{ticker} Price Projection ({model_type} Model)", xaxis_title="Date", yaxis_title="Price", height=600, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

        # --- Panel 2: Fundamentals Summary & 52-Week Range ---
        if show_fundamentals:
            st.markdown("---")
            st.subheader(f"Current Fundamental Snapshot: {ticker}")
            with st.spinner("Fetching fundamentals..."):
                info = load_fundamentals(ticker)
            
            st.markdown("##### 52-Week Range")
            low_52w = info.get('fiftyTwoWeekLow') or raw_df['Close'].tail(252).min()
            high_52w = info.get('fiftyTwoWeekHigh') or raw_df['Close'].tail(252).max()
            
            if low_52w and high_52w and low_52w != high_52w:
                range_percent = (current_price - low_52w) / (high_52w - low_52w)
                range_percent = max(0.0, min(1.0, range_percent))
                
                col_min, col_slider, col_max = st.columns([1, 8, 1])
                with col_min:
                    st.write(f"**MIN:** \n${low_52w:.2f}")
                with col_slider:
                    st.progress(range_percent)
                    st.markdown(f"<div style='text-align: center; color: gray; margin-top: -10px;'><b>Today:</b> ${current_price:.2f}</div>", unsafe_allow_html=True)
                with col_max:
                    st.write(f"**MAX:** \n${high_52w:.2f}")
            else:
                st.write("*52-Week range data unavailable.*")

            st.write("")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Beta", round(info.get('beta', 0), 2) if info.get('beta') else "N/A", help="Systemic risk vs market.")
            col2.metric("Trailing P/E", round(info.get('trailingPE', 0), 2) if info.get('trailingPE') else "N/A", help="Current price / past 12m EPS.")
            col3.metric("Forward P/E", round(info.get('forwardPE', 0), 2) if info.get('forwardPE') else "N/A", help="Current price / next 12m estimated EPS.")
            col4.metric("Price-to-Book (P/B)", round(info.get('priceToBook', 0), 2) if info.get('priceToBook') else "N/A", help="Market value vs book value.")
            
            raw_yield = info.get('dividendYield') or info.get('trailingAnnualDividendYield') or info.get('yield')
            if raw_yield is not None:
                actual_yield = raw_yield * 100 if raw_yield < 0.5 else raw_yield
                div_yield_display = f"{round(actual_yield, 2)}%"
            else:
                div_yield_display = "N/A"
            col5.metric("Dividend Yield", div_yield_display, help="Annual dividend payout / stock price.")

        # --- Panel 3: Dynamic Historical Indicators ---
        if selected_indicators:
            st.markdown("---")
            st.subheader("Historical Indicators")
            
            num_plots = len(selected_indicators)
            fig_ind = make_subplots(rows=num_plots, cols=1, shared_xaxes=True, vertical_spacing=0.05, subplot_titles=selected_indicators)
            
            colors = ['green' if row['Close'] >= row['Open'] else 'red' for index, row in df.iterrows()]

            for i, indicator in enumerate(selected_indicators, start=1):
                
                # ADDED Closing Price logic here
                if indicator == "Closing Price":
                    fig_ind.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', line=dict(color='black'), name="Close"), row=i, col=1)

                elif indicator == "Volume":
                    fig_ind.add_trace(go.Bar(x=df['Date'], y=df['Volume'], marker_color=colors, name="Volume"), row=i, col=1)
                
                elif indicator == "30-Day Rolling Volatility":
                    fig_ind.add_trace(go.Scatter(x=df['Date'], y=df['Rolling_Vol'], mode='lines', line=dict(color='purple'), name="Volatility %"), row=i, col=1)
                
                elif indicator == "RSI (14-Day)":
                    fig_ind.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], mode='lines', line=dict(color='orange'), name="RSI"), row=i, col=1)
                    fig_ind.add_hline(y=70, line_dash="dot", line_color="red", row=i, col=1)
                    fig_ind.add_hline(y=30, line_dash="dot", line_color="green", row=i, col=1)
                
                elif indicator == "MACD":
                    fig_ind.add_trace(go.Scatter(x=df['Date'], y=df['MACD'], mode='lines', line=dict(color='blue'), name="MACD"), row=i, col=1)
                    fig_ind.add_trace(go.Scatter(x=df['Date'], y=df['Signal_Line'], mode='lines', line=dict(color='red'), name="Signal Line"), row=i, col=1)
                    fig_ind.add_trace(go.Bar(x=df['Date'], y=df['MACD'] - df['Signal_Line'], marker_color='gray', name="Histogram"), row=i, col=1)

            plot_height = 250 * num_plots
            fig_ind.update_layout(height=plot_height, template="plotly_white", showlegend=False, margin=dict(t=30, b=10, l=10, r=10))
            st.plotly_chart(fig_ind, use_container_width=True)

        # --- Panel 4: Indicator Glossary ---
        st.markdown("---")
        with st.expander("📚 Glossary of Technical Indicators"):
            st.markdown("""
            * **Closing Price:** The standard, unadjusted raw closing price of the asset.
            * **Volume:** The number of shares traded during a given timeframe. Green bars indicate the closing price was higher than the opening price; red indicates it was lower.
            * **30-Day Rolling Volatility:** Measures how wildly the stock price swings over a 30-day window, annualized. A spike means the stock is becoming riskier/more unpredictable.
            * **RSI (Relative Strength Index):** A momentum oscillator ranging from 0 to 100. 
                * *Above 70 (Red Line):* The stock may be "overbought" and due for a pullback.
                * *Below 30 (Green Line):* The stock may be "oversold" and due for a bounce.
            * **MACD (Moving Average Convergence Divergence):** Shows the relationship between two moving averages (usually 12-day and 26-day).
                * When the blue MACD line crosses *above* the red signal line, it is generally considered a bullish (buy) signal.
                * When it crosses *below*, it is a bearish (sell) signal. The grey histogram shows the distance between the two lines.
            """)