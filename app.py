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
ticker = st.sidebar.text_input("Stock/ETF Ticker (e.g., SPY, AAPL, TSLA)", value="SPY").upper()
p_years = st.sidebar.slider("P: Historical Years to analyze", min_value=1, max_value=20, value=5)

st.sidebar.markdown("---")
model_type = st.sidebar.radio("Fitting Model", options=["Polynomial", "Exponential"])
# Only show polynomial order slider if Polynomial is selected
if model_type == "Polynomial":
    poly_order = st.sidebar.slider("n: Polynomial Order", min_value=1, max_value=5, value=2)
else:
    poly_order = 1 # Not used for exponential, but kept for variable safety

st.sidebar.markdown("---")
y_years = st.sidebar.slider("Y: Future Years to predict", min_value=1, max_value=10, value=2)
show_fundamentals = st.sidebar.checkbox("Show 2nd Panel (PE, P/B & Volume)", value=True)

@st.cache_data
def load_data(ticker, years):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365.25)
    df = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
    return df

@st.cache_data
def load_fundamentals(ticker):
    stock = yf.Ticker(ticker)
    return stock.info

if ticker:
    with st.spinner(f"Loading data for {ticker}..."):
        df = load_data(ticker, p_years)
        
    if df.empty:
        st.error(f"Could not load data for {ticker}. Please check the ticker symbol.")
    else:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # 1. Prepare Historical Data
        df = df.reset_index()
        df['Days'] = (df['Date'] - df['Date'].min()).dt.days
        
        x_hist = df['Days'].values
        y_hist = df['Close'].values
        dates_hist = df['Date']
        
        # 2. Fit the Model (Polynomial vs Exponential)
        if model_type == "Polynomial":
            coeffs = np.polyfit(x_hist, y_hist, poly_order)
            poly_eq = np.poly1d(coeffs)
            y_fit = poly_eq(x_hist)
            model_name = f'Polynomial Fit (n={poly_order})'
            
        elif model_type == "Exponential":
            # To fit y = A * e^(Bx), we fit a line to ln(y) = ln(A) + Bx
            # This requires y_hist to be strictly positive (which stock prices are)
            coeffs = np.polyfit(x_hist, np.log(y_hist), 1)
            B = coeffs[0]
            A = np.exp(coeffs[1])
            y_fit = A * np.exp(B * x_hist)
            model_name = 'Exponential Fit'

        r2 = r2_score(y_hist, y_fit)
        
        # --- THE FIX: Normalized Volatility ---
        # Calculate the relative (percentage) deviation from the fitted line
        normalized_residuals = (y_hist - y_fit) / y_fit
        norm_std = np.std(normalized_residuals) # This is the percentage standard deviation
        
        # Calculate the upper and lower shadows by scaling the fit day-by-day
        hist_upper_bound = y_fit * (1 + norm_std)
        hist_lower_bound = y_fit * (1 - norm_std)
        
        st.subheader(f"Historical Analysis: {ticker}")
        col1, col2 = st.columns(2)
        col1.metric("$R^2$ (Fit Quality)", f"{r2:.4f}")
        col2.metric("Normalized $\sigma$ (Relative Volatility)", f"{(norm_std * 100):.2f}%")

        # 3. Future Prediction
        last_date = df['Date'].max()
        last_day_val = x_hist[-1]
        
        future_days = y_years * 365
        x_future = np.linspace(last_day_val + 1, last_day_val + future_days, future_days)
        dates_future = [pd.to_datetime(last_date) + timedelta(days=int(d)) for d in range(1, future_days + 1)]
        
        # Extrapolate the base future trend using the chosen model
        if model_type == "Polynomial":
            base_future = poly_eq(x_future)
        elif model_type == "Exponential":
            base_future = A * np.exp(B * x_future)
            
        # Create the diverging cone using the normalized standard deviation
        # We scale the variance by sqrt(t) where t is time in years to replicate an expanding uncertainty cone
        time_in_years = np.linspace(1/365, y_years, future_days)
        expanding_uncertainty = norm_std * np.sqrt(time_in_years)
        
        best_scenario = base_future * (1 + expanding_uncertainty)
        worst_scenario = base_future * (1 - expanding_uncertainty)

        # --- Plotting Main Chart ---
        fig = go.Figure()

        # Upper shadow (+sigma, Red) using normalized bounds
        fig.add_trace(go.Scatter(x=dates_hist, y=hist_upper_bound, mode='lines', line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=dates_hist, y=y_fit, mode='lines', line=dict(width=0),
            fill='tonexty', fillcolor='rgba(255, 0, 0, 0.2)', name='Red Shadow (+σ)'))

        # Lower shadow (-sigma, Green) using normalized bounds
        fig.add_trace(go.Scatter(x=dates_hist, y=hist_lower_bound, mode='lines', line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=dates_hist, y=y_fit, mode='lines', line=dict(width=0),
            fill='tonexty', fillcolor='rgba(0, 255, 0, 0.2)', name='Green Shadow (-σ)'))

        # Actual Price & Fit
        fig.add_trace(go.Scatter(x=dates_hist, y=y_hist, mode='lines', name='Actual Price', line=dict(color='black', width=1.5)))
        fig.add_trace(go.Scatter(x=dates_hist, y=y_fit, mode='lines', name=model_name, line=dict(color='blue', dash='dash')))

        # Future Projections
        fig.add_trace(go.Scatter(x=[last_date] + dates_future, y=[y_fit[-1]] + list(best_scenario),
            mode='lines', name='Best Scenario (+1σ)', line=dict(color='green', dash='dot', width=2)))
        fig.add_trace(go.Scatter(x=[last_date] + dates_future, y=[y_fit[-1]] + list(worst_scenario),
            mode='lines', name='Worst Scenario (-1σ)', line=dict(color='red', dash='dot', width=2)))
        fig.add_trace(go.Scatter(x=[last_date] + dates_future, y=[y_fit[-1]] + list(base_future),
            mode='lines', name='Expected Trend', line=dict(color='grey', dash='dash', width=1)))

        # fig.add_vline(x=last_date.strftime('%Y-%m-%d'), line_width=2, line_dash="solid", line_color="black", 
        #               annotation_text="TODAY", annotation_position="top left")

        fig.update_layout(title=f"{ticker} Price Projection ({model_type} Model)", xaxis_title="Date", yaxis_title="Price", height=600, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

        # --- Panel 2: Fundamentals & Volume ---
        if show_fundamentals:
            st.markdown("---")
            st.subheader(f"Fundamental Analysis & Volume: {ticker}")
            
            with st.spinner("Fetching fundamentals..."):
                info = load_fundamentals(ticker)
                
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Trailing P/E", round(info.get('trailingPE', 0), 2) if info.get('trailingPE') else "N/A")
            col2.metric("Forward P/E", round(info.get('forwardPE', 0), 2) if info.get('forwardPE') else "N/A")
            col3.metric("Price-to-Book (P/B)", round(info.get('priceToBook', 0), 2) if info.get('priceToBook') else "N/A")
            col4.metric("Dividend Yield", f"{round(info.get('dividendYield', 0) * 100, 2)}%" if info.get('dividendYield') else "N/A")
            
            fig_vol = go.Figure()
            colors = ['green' if row['Close'] >= row['Open'] else 'red' for index, row in df.iterrows()]
            
            fig_vol.add_trace(go.Bar(x=df['Date'], y=df['Volume'], marker_color=colors, name="Volume"))
            fig_vol.update_layout(title="Historical Trading Volume", xaxis_title="Date", yaxis_title="Volume", height=300, template="plotly_white", margin=dict(t=40, b=10, l=10, r=10))
            
            st.plotly_chart(fig_vol, use_container_width=True)