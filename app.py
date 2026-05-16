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
st.title("Quantitative Trend Analyzer")

# --- Sidebar Inputs ---
st.sidebar.header("Configuration")
ticker = st.sidebar.text_input("Stock/ETF Ticker (e.g., SPY, AAPL, NVDA)", value="NVDA").upper()

st.sidebar.markdown("---")
st.sidebar.subheader("Model Parameters")
p_years = st.sidebar.slider("P: Historical Years to analyze", min_value=1, max_value=20, value=5)
v_pct = st.sidebar.slider("V: Validation Split (%)", min_value=0, max_value=50, value=20, help="Percentage of recent historical data to hold out for validation.")

model_type = st.sidebar.radio("Fitting Model", options=["Polynomial", "Exponential"])
if model_type == "Polynomial":
    poly_order = st.sidebar.slider("n: Polynomial Order", min_value=1, max_value=5, value=2)
else:
    poly_order = 1 
y_years = st.sidebar.slider("Y: Future Years to predict", min_value=0, max_value=10, value=2)

st.sidebar.markdown("---")
st.sidebar.subheader("Panel Metrics")
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

# --- THE MAIN SCRIPT ---
if ticker:
    with st.spinner("Fetching data from market..."):
        master_df = load_data(ticker, 30).copy()

    if master_df.empty:
        st.error(f"Could not retrieve data for {ticker}.")
    else:
        st.markdown(f"### Market Projection: {ticker}")
        
        real_today = master_df['Date'].max()
        start_hist_date = real_today - timedelta(days=p_years * 365.25)
        
        # 1. Train/Validation Time Split Logic
        val_days_count = int((p_years * 365.25) * (v_pct / 100.0))
        split_date = real_today - timedelta(days=val_days_count)
        
        full_hist_df = master_df[(master_df['Date'] >= start_hist_date) & (master_df['Date'] <= real_today)].copy()
        
        if len(full_hist_df) < 50:
            st.error("Not enough historical data for this date range.")
        else:
            full_hist_df['Days'] = (full_hist_df['Date'] - full_hist_df['Date'].min()).dt.days
            
            train_mask = full_hist_df['Date'] <= split_date
            val_mask = full_hist_df['Date'] > split_date
            
            x_train = full_hist_df.loc[train_mask, 'Days'].values
            y_train = full_hist_df.loc[train_mask, 'Close'].values
            dates_train = full_hist_df.loc[train_mask, 'Date']
            
            x_val = full_hist_df.loc[val_mask, 'Days'].values
            y_val = full_hist_df.loc[val_mask, 'Close'].values
            dates_val = full_hist_df.loc[val_mask, 'Date']
            
            x_full = full_hist_df['Days'].values
            y_full = full_hist_df['Close'].values
            
            # 2. Fit the Model (STRICTLY ON TRAINING DATA)
            if model_type == "Polynomial":
                coeffs = np.polyfit(x_train, y_train, poly_order)
                poly_eq = np.poly1d(coeffs)
                y_fit_train = poly_eq(x_train)
                y_fit_val = poly_eq(x_val) if v_pct > 0 else []
                model_name = f'Poly Fit (n={poly_order})'
            elif model_type == "Exponential":
                coeffs = np.polyfit(x_train, np.log(y_train), 1)
                B = coeffs[0]
                A = np.exp(coeffs[1])
                y_fit_train = A * np.exp(B * x_train)
                y_fit_val = A * np.exp(B * x_val) if v_pct > 0 else []
                model_name = 'Exponential Fit'

            # Calculate R^2 Metrics
            r2_train = r2_score(y_train, y_fit_train)
            r2_val = r2_score(y_val, y_fit_val) if v_pct > 0 and len(y_val) > 1 else None
            
            # Calculate historical volatility (Q process noise) from training set residuals
            normalized_residuals = (y_train - y_fit_train) / y_fit_train
            norm_std = np.std(normalized_residuals)
            
            hist_upper_train = y_fit_train * (1 + norm_std)
            hist_lower_train = y_fit_train * (1 - norm_std)

            # 3. Projection Phase (Kalman Covariance covers Validation + Unknown Future)
            unknown_future_days = int(y_years * 365)
            total_proj_days = len(x_val) + unknown_future_days
            
            final_expected, final_upper, final_lower = 0, 0, 0
            coverage_pct, rmse = "N/A", "N/A"
            
            if total_proj_days > 0:
                last_train_day_val = x_train[-1]
                last_train_date = dates_train.iloc[-1]
                
                x_proj = np.linspace(last_train_day_val + 1, last_train_day_val + total_proj_days, total_proj_days)
                dates_proj = [last_train_date + timedelta(days=int(d)) for d in range(1, total_proj_days + 1)]
                
                if model_type == "Polynomial":
                    proj_expected = poly_eq(x_proj)
                elif model_type == "Exponential":
                    proj_expected = A * np.exp(B * x_proj)
                    
                # Kalman Filter Covariance Update
                P_t_pct = norm_std ** 2 
                Q = np.var(np.diff(y_train))
                T_days = np.arange(1, total_proj_days + 1)
                
                proj_variance = (P_t_pct * (proj_expected ** 2)) + (Q * T_days)
                proj_uncertainty_dollars = np.sqrt(proj_variance)
                
                proj_upper = proj_expected + proj_uncertainty_dollars
                proj_lower = proj_expected - proj_uncertainty_dollars
                
                final_expected = proj_expected[-1]
                final_upper = proj_upper[-1]
                final_lower = proj_lower[-1]

                # Evaluate over the Validation Set if it exists
                if v_pct > 0 and len(y_val) > 0:
                    pred_df = pd.DataFrame({
                        'Date': pd.to_datetime(dates_proj),
                        'Expected': proj_expected,
                        'Upper': proj_upper,
                        'Lower': proj_lower
                    })
                    val_df_merge = full_hist_df.loc[val_mask, ['Date', 'Close']]
                    eval_df = pd.merge(val_df_merge, pred_df, on='Date', how='inner')
                    
                    if not eval_df.empty:
                        inside_cone = ((eval_df['Close'] <= eval_df['Upper']) & (eval_df['Close'] >= eval_df['Lower'])).sum()
                        coverage_pct = f"{(inside_cone / len(eval_df)) * 100:.1f}%"
                        rmse = f"${np.sqrt(np.mean((eval_df['Close'] - eval_df['Expected'])**2)):.2f}"

            # 4. Plotting Main Chart
            fig = go.Figure()

            # Train Shadows
            fig.add_trace(go.Scatter(x=dates_train, y=hist_upper_train, mode='lines', line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=dates_train, y=y_fit_train, mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(255, 0, 0, 0.2)', name='Red Shadow (+σ)'))
            fig.add_trace(go.Scatter(x=dates_train, y=hist_lower_train, mode='lines', line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=dates_train, y=y_fit_train, mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0, 255, 0, 0.2)', name='Green Shadow (-σ)'))

            # Actual Train Price & Fit
            fig.add_trace(go.Scatter(x=dates_train, y=y_train, mode='lines', name='Actual Price (Train)', line=dict(color='black', width=1.5)))
            fig.add_trace(go.Scatter(x=dates_train, y=y_fit_train, mode='lines', name=model_name, line=dict(color='blue', dash='dash')))

            # Actual Validation Price
            if v_pct > 0 and len(y_val) > 0:
                conn_x = [dates_train.iloc[-1], dates_val.iloc[0]]
                conn_y = [y_train[-1], y_val[0]]
                fig.add_trace(go.Scatter(x=conn_x, y=conn_y, mode='lines', showlegend=False, line=dict(color='orange', width=1.5)))
                fig.add_trace(go.Scatter(x=dates_val, y=y_val, mode='lines', name='Actual Price (Val)', line=dict(color='orange', width=2.0)))
                fig.add_vrect(x0=split_date.strftime('%Y-%m-%d'), x1=real_today.strftime('%Y-%m-%d'), fillcolor="rgba(255, 165, 0, 0.1)", layer="below", line_width=0, annotation_text="Validation Phase", annotation_position="top left")
                fig.add_vline(x=split_date.strftime('%Y-%m-%d'), line_width=2, line_dash="dot", line_color="orange")

            # Projected Cone (Validation + Unknown Future)
            if total_proj_days > 0:
                proj_x_plot = [dates_train.iloc[-1]] + dates_proj
                proj_expected_plot = [y_fit_train[-1]] + list(proj_expected)
                proj_upper_plot = [hist_upper_train[-1]] + list(proj_upper)
                proj_lower_plot = [hist_lower_train[-1]] + list(proj_lower)

                fig.add_trace(go.Scatter(x=proj_x_plot, y=proj_upper_plot, mode='lines', name='Best Scenario (+1σ)', line=dict(color='green', dash='dot', width=2)))
                fig.add_trace(go.Scatter(x=proj_x_plot, y=proj_lower_plot, mode='lines', name='Worst Scenario (-1σ)', line=dict(color='red', dash='dot', width=2)))
                fig.add_trace(go.Scatter(x=proj_x_plot, y=proj_expected_plot, mode='lines', name='Expected Trend', line=dict(color='grey', dash='dash', width=2)))

            # Today Line
            # fig.add_vline(x=real_today.strftime('%Y-%m-%d'), line_width=2, line_dash="solid", line_color="black", annotation_text="TODAY", annotation_position="top left")

            chart_type = "Projection" if y_years > 0 else "Historical Fit"
            fig.update_layout(title=f"{ticker} {chart_type} ({model_type} Model)", xaxis_title="Date", yaxis_title="Price", height=600, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

            # 5. METRICS & SCENARIOS TABLE
            st.markdown("#### Model Performance & Future Scenarios")
            r2_val_display = f"{r2_val:.4f}" if r2_val is not None else "N/A (V=0%)"
            
            # Formatted Scenario Strings with Percentages
            expected_str = "N/A"
            upper_str = "N/A"
            lower_str = "N/A"
            
            if y_years > 0 and total_proj_days > 0:
                base_price = y_full[-1] # The true "Today" price
                
                expected_pct = ((final_expected / base_price) - 1) * 100
                upper_pct = ((final_upper / base_price) - 1) * 100
                lower_pct = ((final_lower / base_price) - 1) * 100
                
                expected_str = f"${final_expected:.2f} ({expected_pct:+.1f}%)"
                upper_str = f"${final_upper:.2f} ({upper_pct:+.1f}%)"
                lower_str = f"${final_lower:.2f} ({lower_pct:+.1f}%)"

            table_data = {
                "Metric / Scenario": [
                    "Training R-Squared",
                    f"Validation R-Squared ({v_pct}% Holdout)",
                    "Validation Coverage (±1σ Cone)",
                    "Validation RMSE (Avg Dollar Error)",
                    "Base Historical Volatility (σ)",
                    f"Expected Target Price (in {y_years}Y)",
                    f"Best Case (+1σ) Scenario",
                    f"Worst Case (-1σ) Scenario"
                ],
                "Value": [
                    f"{r2_train:.4f}",
                    r2_val_display,
                    coverage_pct,
                    rmse,
                    f"{(norm_std * 100):.2f}%",
                    expected_str,
                    upper_str,
                    lower_str
                ]
            }
            metrics_df = pd.DataFrame(table_data)
            st.dataframe(metrics_df, hide_index=True, use_container_width=True)

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
                
                fig_ind = make_subplots(
                    rows=num_plots, 
                    cols=1, 
                    shared_xaxes=True, 
                    vertical_spacing=0.02, 
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
                    showlegend=False, 
                    margin=dict(t=30, b=10, l=10, r=10),
                    hovermode="x unified",
                    hoverdistance=-1,
                    spikedistance=-1
                )
                
                fig_ind.update_xaxes(
                    showspikes=True,
                    spikemode="across",
                    spikesnap="cursor",
                    showline=True,
                    spikedash="solid",
                    spikecolor="darkgray",
                    spikethickness=1
                )
                
                fig_ind.update_yaxes(showspikes=False)

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