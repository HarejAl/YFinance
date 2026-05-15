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
st.title("Quantitative Trend Analyzer & Backtester")

# --- Sidebar Inputs ---
st.sidebar.header("Configuration")
ticker = st.sidebar.text_input("Stock/ETF Ticker (e.g., SPY, AAPL, NVDA)", value="NVDA").upper()

st.sidebar.markdown("---")
st.sidebar.subheader("Model Parameters")
p_years = st.sidebar.slider("P: Historical Years to analyze", min_value=1, max_value=20, value=5)
model_type = st.sidebar.radio("Fitting Model", options=["Polynomial", "Exponential"])
if model_type == "Polynomial":
    poly_order = st.sidebar.slider("n: Polynomial Order", min_value=1, max_value=5, value=2)
else:
    poly_order = 1 
y_years = st.sidebar.slider("Y: Future Years to predict", min_value=0, max_value=10, value=2)

st.sidebar.markdown("---")
st.sidebar.subheader("Live Panel Metrics")
show_fundamentals = st.sidebar.checkbox("Show Current Fundamentals Summary", value=True)

selected_indicators = st.sidebar.multiselect(
    "Select Historical Indicators to Plot:",
    ["Closing Price", "Volume", "30-Day Rolling Volatility", "RSI (14-Day)", "MACD"],
    default=["Closing Price", "Volume"]
)

show_sp500_comp = st.sidebar.checkbox("Compare vs S&P 500", value=True)

@st.cache_data
def load_data(ticker, years):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=max(years, 1) * 365.25 + 30) 
    df = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    return df

@st.cache_data
def load_benchmark(start_date, end_date):
    df = yf.download('SPY', start=start_date, end=end_date)
    return df

@st.cache_data
def load_fundamentals(ticker):
    stock = yf.Ticker(ticker)
    return stock.info

# --- THE CORE MATH & PLOTTING ENGINE ---
def generate_projection_chart(df, ticker, simulated_today, p_years, y_years, model_type, poly_order, is_backtest=False):
    start_train_date = simulated_today - timedelta(days=p_years * 365.25)
    train_df = df[(df['Date'] >= start_train_date) & (df['Date'] <= simulated_today)].copy()
    
    if train_df.empty or len(train_df) < 50:
        st.error("Not enough historical data for this date range.")
        return None

    end_test_date = simulated_today + timedelta(days=max(y_years, 1) * 365.25)
    future_actual_df = df[(df['Date'] > simulated_today) & (df['Date'] <= end_test_date)].copy()

    train_df['Days'] = (train_df['Date'] - train_df['Date'].min()).dt.days
    x_hist = train_df['Days'].values
    y_hist = train_df['Close'].values
    dates_hist = train_df['Date']
    
    if model_type == "Polynomial":
        coeffs = np.polyfit(x_hist, y_hist, poly_order)
        poly_eq = np.poly1d(coeffs)
        y_fit = poly_eq(x_hist)
        model_name = f'Poly Fit (n={poly_order})'
    elif model_type == "Exponential":
        coeffs = np.polyfit(x_hist, np.log(y_hist), 1)
        B = coeffs[0]
        A = np.exp(coeffs[1])
        y_fit = A * np.exp(B * x_hist)
        model_name = 'Exponential Fit'

    r2_train = r2_score(y_hist, y_fit)
    normalized_residuals = (y_hist - y_fit) / y_fit
    norm_std = np.std(normalized_residuals)
    
    hist_upper_bound = y_fit * (1 + norm_std)
    hist_lower_bound = y_fit * (1 - norm_std)
    
    st.subheader(f"Historical Training Stats ({start_train_date.strftime('%Y')} to {simulated_today.strftime('%Y')})")
    col1, col2 = st.columns(2)
    col1.metric("Training R-Squared (Fit Quality)", f"{r2_train:.4f}")
    col2.metric("Historical Volatility", f"{(norm_std * 100):.2f}%")

    if y_years > 0:
        future_days = y_years * 365
        last_day_val = x_hist[-1]
        x_future = np.linspace(last_day_val + 1, last_day_val + future_days, future_days)
        dates_future = [simulated_today + timedelta(days=int(d)) for d in range(1, future_days + 1)]
        
        if model_type == "Polynomial":
            average_scenario = poly_eq(x_future)
        elif model_type == "Exponential":
            average_scenario = A * np.exp(B * x_future)
            
        time_in_years = np.linspace(1/365, y_years, future_days)
        expanding_uncertainty = norm_std * np.sqrt(time_in_years)
        
        best_scenario = average_scenario * (1 + expanding_uncertainty)
        worst_scenario = average_scenario * (1 - expanding_uncertainty)

        if is_backtest and not future_actual_df.empty:
            pred_df = pd.DataFrame({
                'Date': pd.to_datetime(dates_future),
                'Expected': average_scenario,
                'Upper': best_scenario,
                'Lower': worst_scenario
            })
            eval_df = pd.merge(future_actual_df[['Date', 'Close']], pred_df, on='Date', how='inner')
            
            if not eval_df.empty:
                inside_cone = ((eval_df['Close'] <= eval_df['Upper']) & (eval_df['Close'] >= eval_df['Lower'])).sum()
                coverage_pct = (inside_cone / len(eval_df)) * 100
                rmse = np.sqrt(np.mean((eval_df['Close'] - eval_df['Expected'])**2))
                future_r2 = r2_score(eval_df['Close'], eval_df['Expected'])
                
                st.markdown("---")
                st.markdown("#### Backtest Prediction Accuracy (Future Data)")
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Coverage (±1σ Cone)", f"{coverage_pct:.1f}%")
                col_b.metric("RMSE (Avg Dollar Error)", f"${rmse:.2f}")
                r2_color = "green" if future_r2 > 0 else "red"
                col_c.markdown(f"""
                    <div style='padding-top: 1rem;'>
                        <p style='margin: 0; font-size: 0.8rem; color: gray;'>Future R² Score</p>
                        <h3 style='margin: 0; color: {r2_color};'>{future_r2:.4f}</h3>
                    </div>
                """, unsafe_allow_html=True)
                st.markdown("---")

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=dates_hist, y=hist_upper_bound, mode='lines', line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=dates_hist, y=y_fit, mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(255, 0, 0, 0.2)', name='Red Shadow (+σ)'))
    fig.add_trace(go.Scatter(x=dates_hist, y=hist_lower_bound, mode='lines', line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=dates_hist, y=y_fit, mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0, 255, 0, 0.2)', name='Green Shadow (-σ)'))

    fig.add_trace(go.Scatter(x=dates_hist, y=y_hist, mode='lines', name='Actual Price', line=dict(color='black', width=1.5)))
    fig.add_trace(go.Scatter(x=dates_hist, y=y_fit, mode='lines', name=model_name, line=dict(color='blue', dash='dash')))

    if y_years > 0:
        fig.add_trace(go.Scatter(x=[simulated_today] + dates_future, y=[y_fit[-1]] + list(best_scenario), mode='lines', name='Best Scenario (+1σ)', line=dict(color='green', dash='dot', width=2)))
        fig.add_trace(go.Scatter(x=[simulated_today] + dates_future, y=[y_fit[-1]] + list(worst_scenario), mode='lines', name='Worst Scenario (-1σ)', line=dict(color='red', dash='dot', width=2)))
        fig.add_trace(go.Scatter(x=[simulated_today] + dates_future, y=[y_fit[-1]] + list(average_scenario), mode='lines', name='Expected Trend', line=dict(color='grey', dash='dash', width=2)))

        if is_backtest and not future_actual_df.empty:
            fig.add_trace(go.Scatter(x=future_actual_df['Date'], y=future_actual_df['Close'], mode='lines', name='ACTUAL FUTURE PRICE', line=dict(color='purple', width=2.5)))


    title_prefix = "Backtest Simulation" if is_backtest else "Live Projection"
    chart_type = "Projection" if y_years > 0 else "Historical Fit"
    fig.update_layout(title=f"{title_prefix}: {ticker} {chart_type} ({model_type} Model)", xaxis_title="Date", yaxis_title="Price", height=600, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)


# --- APP LAYOUT (TABS) ---
if ticker:
    with st.spinner("Fetching data from market..."):
        master_df = load_data(ticker, 30).copy()

    if master_df.empty:
        st.error(f"Could not retrieve data for {ticker}.")
    else:
        tab1, tab2 = st.tabs(["Live Market Analysis", "Time Machine (Backtester)"])

        # ==========================================
        # TAB 1: LIVE PROJECTION & INDICATORS
        # ==========================================
        with tab1:
            st.markdown("### Current Market Projection")
            real_today = master_df['Date'].max()
            generate_projection_chart(master_df, ticker, real_today, p_years, y_years, model_type, poly_order, is_backtest=False)

            # --- Panel: Fundamentals Summary & 52-Week Range ---
            if show_fundamentals:
                st.markdown("---")
                st.subheader(f"Current Fundamental Snapshot: {ticker}")
                with st.spinner("Fetching fundamentals..."):
                    info = load_fundamentals(ticker)
                
                current_price = master_df['Close'].iloc[-1]
                st.markdown("##### 52-Week Range")
                low_52w = info.get('fiftyTwoWeekLow') or master_df['Close'].tail(252).min()
                high_52w = info.get('fiftyTwoWeekHigh') or master_df['Close'].tail(252).max()
                
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
                col1.metric("Beta", round(info.get('beta', 0), 2) if info.get('beta') else "N/A")
                col2.metric("Trailing P/E", round(info.get('trailingPE', 0), 2) if info.get('trailingPE') else "N/A")
                col3.metric("Forward P/E", round(info.get('forwardPE', 0), 2) if info.get('forwardPE') else "N/A")
                col4.metric("Price-to-Book (P/B)", round(info.get('priceToBook', 0), 2) if info.get('priceToBook') else "N/A")
                
                raw_yield = info.get('dividendYield') or info.get('trailingAnnualDividendYield') or info.get('yield')
                if raw_yield is not None:
                    actual_yield = raw_yield * 100 if raw_yield < 0.5 else raw_yield
                    div_yield_display = f"{round(actual_yield, 2)}%"
                else:
                    div_yield_display = "N/A"
                col5.metric("Dividend Yield", div_yield_display)

            # --- Panel: Dynamic Historical Indicators ---
            if selected_indicators:
                st.markdown("---")
                st.subheader("Historical Indicators")
                
                cutoff_date = pd.to_datetime(datetime.now() - timedelta(days=p_years * 365.25))
                df_ind = master_df[master_df['Date'] >= cutoff_date].copy()
                
                df_ind['Daily_Return'] = df_ind['Close'].pct_change()
                df_ind['Rolling_Vol'] = df_ind['Daily_Return'].rolling(window=30).std() * np.sqrt(252) * 100
                
                delta = df_ind['Close'].diff()
                up = delta.clip(lower=0)
                down = -1 * delta.clip(upper=0)
                ema_up = up.ewm(com=13, adjust=False).mean()
                ema_down = down.ewm(com=13, adjust=False).mean()
                rs = ema_up / ema_down
                df_ind['RSI'] = 100 - (100 / (1 + rs))
                
                exp1 = df_ind['Close'].ewm(span=12, adjust=False).mean()
                exp2 = df_ind['Close'].ewm(span=26, adjust=False).mean()
                df_ind['MACD'] = exp1 - exp2
                df_ind['Signal_Line'] = df_ind['MACD'].ewm(span=9, adjust=False).mean()
                
                num_plots = len(selected_indicators)
                
                # --- APPLY THE FIXES HERE ---
                fig_ind = make_subplots(
                    rows=num_plots, 
                    cols=1, 
                    shared_xaxes=True, 
                    vertical_spacing=0.02,  # Shrink vertical spacing
                    subplot_titles=selected_indicators
                )
                
                colors = ['green' if row['Close'] >= row['Open'] else 'red' for index, row in df_ind.iterrows()]

                for i, indicator in enumerate(selected_indicators, start=1):
                    if indicator == "Closing Price":
                        fig_ind.add_trace(go.Scatter(x=df_ind['Date'], y=df_ind['Close'], mode='lines', line=dict(color='black'), name="Close"), row=i, col=1)
                    elif indicator == "Volume":
                        fig_ind.add_trace(go.Bar(x=df_ind['Date'], y=df_ind['Volume'], marker_color=colors, name="Volume"), row=i, col=1)
                    elif indicator == "30-Day Rolling Volatility":
                        fig_ind.add_trace(go.Scatter(x=df_ind['Date'], y=df_ind['Rolling_Vol'], mode='lines', line=dict(color='purple'), name="Volatility %"), row=i, col=1)
                    elif indicator == "RSI (14-Day)":
                        fig_ind.add_trace(go.Scatter(x=df_ind['Date'], y=df_ind['RSI'], mode='lines', line=dict(color='orange'), name="RSI"), row=i, col=1)
                        fig_ind.add_hline(y=70, line_dash="dot", line_color="red", row=i, col=1)
                        fig_ind.add_hline(y=30, line_dash="dot", line_color="green", row=i, col=1)
                    elif indicator == "MACD":
                        fig_ind.add_trace(go.Scatter(x=df_ind['Date'], y=df_ind['MACD'], mode='lines', line=dict(color='blue'), name="MACD"), row=i, col=1)
                        fig_ind.add_trace(go.Scatter(x=df_ind['Date'], y=df_ind['Signal_Line'], mode='lines', line=dict(color='red'), name="Signal Line"), row=i, col=1)
                        fig_ind.add_trace(go.Bar(x=df_ind['Date'], y=df_ind['MACD'] - df_ind['Signal_Line'], marker_color='gray', name="Histogram"), row=i, col=1)

                plot_height = 250 * num_plots
                
            

                fig_ind.update_layout(
                    height=plot_height,
                    template="plotly_white",
                    showlegend=True,
                    margin=dict(t=30, b=10, l=10, r=10),
                    hovermode="x unified",
                    hoverdistance=-1,
                    spikedistance=-1
                )

                fig_ind.update_xaxes(
                    matches="x",
                    showspikes=True,
                    spikemode="across",
                    spikesnap="cursor",
                    spikecolor="black",
                    spikedash="dot",
                    spikethickness=1,
                    showline=True
                )

                fig_ind.update_yaxes(showspikes=False)
                        
                # ------------------------------

                st.plotly_chart(fig_ind, use_container_width=True)

            # --- Panel: S&P 500 Comparison ---
            if show_sp500_comp:
                st.markdown("---")
                st.subheader(f"Relative Performance: {ticker} vs S&P 500")
                
                with st.spinner("Fetching S&P 500 benchmark data..."):
                    start_date_comp = (master_df['Date'].max() - timedelta(days=p_years * 365.25)).strftime('%Y-%m-%d')
                    spy_raw = load_benchmark(start_date_comp, master_df['Date'].max().strftime('%Y-%m-%d')).copy()
                    
                if not spy_raw.empty:
                    if isinstance(spy_raw.columns, pd.MultiIndex):
                        spy_raw.columns = spy_raw.columns.get_level_values(0)
                    spy_raw = spy_raw.reset_index()
                    
                    df_comp_base = master_df[master_df['Date'] >= pd.to_datetime(start_date_comp)].copy()
                    
                    if ticker == "SPY":
                        fig_comp = go.Figure()
                        norm_spy = df_comp_base['Close'] / df_comp_base['Close'].iloc[0]
                        fig_comp.add_trace(go.Scatter(x=df_comp_base['Date'], y=norm_spy, mode='lines', name='S&P 500 (SPY)', line=dict(color='blue', width=2)))
                        
                        fig_comp.update_layout(
                            title=f"Growth of $1 Invested ({p_years} Years)",
                            xaxis_title="Date",
                            yaxis_title="Multiplier ($)",
                            yaxis_tickprefix="$",
                            yaxis_tickformat=".2f",
                            height=450,
                            template="plotly_white",
                            hovermode="x unified"
                        )
                        st.plotly_chart(fig_comp, use_container_width=True)
                    else:
                        df_target = df_comp_base[['Date', 'Close']].rename(columns={'Close': 'Close_Target'})
                        df_spy = spy_raw[['Date', 'Close']].rename(columns={'Close': 'Close_SPY'})
                        
                        comp_df = pd.merge(df_target, df_spy, on='Date', how='inner')
                        comp_df['Norm_Target'] = comp_df['Close_Target'] / comp_df['Close_Target'].iloc[0]
                        comp_df['Norm_SPY'] = comp_df['Close_SPY'] / comp_df['Close_SPY'].iloc[0]
                        
                        fig_comp = go.Figure()
                        fig_comp.add_trace(go.Scatter(x=comp_df['Date'], y=comp_df['Norm_Target'], mode='lines', name=ticker, line=dict(color='blue', width=2)))
                        fig_comp.add_trace(go.Scatter(x=comp_df['Date'], y=comp_df['Norm_SPY'], mode='lines', name='S&P 500 (SPY)', line=dict(color='grey', width=2)))
                        
                        fig_comp.update_layout(
                            title=f"Growth of $1 Invested ({p_years} Years)",
                            xaxis_title="Date",
                            yaxis_title="Multiplier ($)",
                            yaxis_tickprefix="$",
                            yaxis_tickformat=".2f",
                            height=450,
                            template="plotly_white",
                            hovermode="x unified",
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                        )
                        st.plotly_chart(fig_comp, use_container_width=True)
                else:
                    st.warning("Could not load benchmark data for comparison.")

            # --- Panel: Indicator Glossary ---
            st.markdown("---")
            with st.expander("Glossary of Technical Indicators"):
                st.markdown("""
                * **Closing Price:** The standard, unadjusted raw closing price of the asset.
                * **Volume:** The number of shares traded. Green bars indicate the closing price was higher than the opening price; red indicates it was lower.
                * **30-Day Rolling Volatility:** Measures how wildly the stock price swings over a 30-day window, annualized. A spike means the stock is becoming riskier/more unpredictable.
                * **RSI (Relative Strength Index):** A momentum oscillator ranging from 0 to 100. *Above 70:* potentially "overbought". *Below 30:* potentially "oversold".
                * **MACD (Moving Average Convergence Divergence):** When the blue MACD line crosses *above* the red signal line, it is generally considered a bullish signal. When it crosses *below*, it is bearish.
                """)

        # ==========================================
        # TAB 2: HISTORICAL BACKTEST SIMULATOR
        # ==========================================
        with tab2:
            st.markdown("### Historical Backtest Simulator")
            
            col_date, col_empty = st.columns([1, 2])
            with col_date:
                default_backtest_date = master_df['Date'].max() - timedelta(days=2 * 365)
                min_allowed_date = master_df['Date'].min() + timedelta(days=p_years * 365)
                
                simulated_date = st.date_input(
                    "Select a simulated 'Today' date:", 
                    value=default_backtest_date,
                    min_value=min_allowed_date,
                    max_value=master_df['Date'].max() - timedelta(days=30)
                )
            
            simulated_date = pd.to_datetime(simulated_date)
            generate_projection_chart(master_df, ticker, simulated_date, p_years, y_years, model_type, poly_order, is_backtest=True)