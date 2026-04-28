import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Stock Dashboard", page_icon="📈", layout="wide")
st.title("📈 Stock Analytics & Portfolio Dashboard")

tab1, tab2 = st.tabs(["Part 1 — Stock Analysis", "Part 2 — Portfolio"])


# ── Helpers ──────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def fetch(ticker, period):
    df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
    return df[["Close"]].dropna() if not df.empty else pd.DataFrame()

def rsi(series, n=14):
    d = series.diff()
    gain = d.clip(lower=0).rolling(n).mean()
    loss = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + gain / loss)

def ann_vol(returns, n=20):
    return returns.rolling(n).std().iloc[-1] * np.sqrt(252) * 100

def sharpe(returns, rf=0.05):
    r, v = returns.mean() * 252, returns.std() * np.sqrt(252)
    return (r - rf) / v if v else 0.0


# ── Part 1 ───────────────────────────────────────────────────────────────────

with tab1:
    ticker = st.text_input("Ticker", value="AAPL", max_chars=10).upper().strip()

    with st.spinner("Loading…"):
        df = fetch(ticker, "6mo")

    if df.empty:
        st.error(f"No data for {ticker}.")
        st.stop()

    close = df["Close"].squeeze()
    ma20  = close.rolling(20).mean()
    ma50  = close.rolling(50).mean()
    price = float(close.iloc[-1])
    m20   = float(ma20.iloc[-1])
    m50   = float(ma50.iloc[-1])

    if price > m20 > m50:
        trend = "Strong Uptrend"
    elif price < m20 < m50:
        trend = "Strong Downtrend"
    else:
        trend = "Mixed"

    r     = rsi(close)
    r_val = float(r.iloc[-1])
    rsi_sig = "Overbought" if r_val > 70 else ("Oversold" if r_val < 30 else "Neutral")

    rets  = close.pct_change().dropna()
    vol   = ann_vol(rets)
    vol_l = "High" if vol > 40 else ("Medium" if vol >= 25 else "Low")

    buys  = (trend == "Strong Uptrend") + (rsi_sig == "Oversold")
    sells = (trend == "Strong Downtrend") + (rsi_sig == "Overbought")
    rec   = "Buy" if buys > sells else ("Sell" if sells > buys else "Hold")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Price",     f"${price:.2f}")
    c2.metric("20-Day MA", f"${m20:.2f}")
    c3.metric("50-Day MA", f"${m50:.2f}")
    c4.metric("RSI",       f"{r_val:.1f}", rsi_sig)
    c5.metric("Ann. Vol.", f"{vol:.1f}%",  vol_l)

    st.write(f"**Trend:** {trend} &nbsp;|&nbsp; **Recommendation:** {rec}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=close.index, y=close, name="Price",  line=dict(color="#4361ee", width=2)))
    fig.add_trace(go.Scatter(x=ma20.index,  y=ma20,  name="20-MA",  line=dict(color="#f77f00", width=1.5, dash="dot")))
    fig.add_trace(go.Scatter(x=ma50.index,  y=ma50,  name="50-MA",  line=dict(color="#d62828", width=1.5, dash="dash")))
    fig.update_layout(height=300, margin=dict(t=10, b=10, l=10, r=10),
                      legend=dict(orientation="h", y=1.05), yaxis_title="USD")
    st.plotly_chart(fig, use_container_width=True)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=r.index, y=r, name="RSI",
                              line=dict(color="#7209b7", width=2),
                              fill="tozeroy", fillcolor="rgba(114,9,183,0.07)"))
    fig2.add_hline(y=70, line_dash="dash", line_color="#d62828", annotation_text="70")
    fig2.add_hline(y=30, line_dash="dash", line_color="#2dc653", annotation_text="30")
    fig2.update_layout(height=200, margin=dict(t=10, b=10, l=10, r=10),
                       yaxis=dict(range=[0, 100]), yaxis_title="RSI")
    st.plotly_chart(fig2, use_container_width=True)

    with st.expander("Written Interpretation"):
        st.markdown(f"""
**Trend:** Price (${price:.2f}) is {"above" if price > m20 else "below"} the 20-MA (${m20:.2f}) and {"above" if m20 > m50 else "below"} the 50-MA (${m50:.2f}) → **{trend}**.

**RSI:** {r_val:.1f} → **{rsi_sig}**. {"Possible selling pressure." if rsi_sig == "Overbought" else "Possible buying opportunity." if rsi_sig == "Oversold" else "No strong momentum signal."}

**Volatility:** {vol:.1f}% annualized → **{vol_l}** risk.

**Recommendation: {rec}**
        """)


# ── Part 2 ───────────────────────────────────────────────────────────────────

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        tickers_in = st.text_input("5 Tickers (comma-separated)", value="AAPL, MSFT, GOOGL, AMZN, NVDA")
        bench_in   = st.text_input("Benchmark", value="SPY")
    with col2:
        weights_in = st.text_input("Weights (sum to 1.00)", value="0.25, 0.25, 0.20, 0.15, 0.15")

    tickers = [t.strip().upper() for t in tickers_in.split(",") if t.strip()]
    try:
        weights = [float(w.strip()) for w in weights_in.split(",") if w.strip()]
    except ValueError:
        st.error("Invalid weights.")
        st.stop()

    if len(tickers) != 5 or len(weights) != 5:
        st.warning("Enter exactly 5 tickers and 5 weights.")
        st.stop()
    if abs(sum(weights) - 1.0) > 0.01:
        st.warning(f"Weights sum to {sum(weights):.2f}, must be 1.00.")
        st.stop()

    if st.button("Run Analysis", type="primary"):
        with st.spinner("Fetching data…"):
            data, failed = {}, []
            for t in tickers + [bench_in]:
                d = fetch(t, "1y")
                if d.empty: failed.append(t)
                else: data[t] = d["Close"].squeeze()

        if failed:
            st.error(f"No data for: {', '.join(failed)}")
            st.stop()

        prices     = pd.DataFrame({t: data[t] for t in tickers}).dropna()
        bench_p    = data[bench_in].reindex(prices.index).dropna()
        prices     = prices.reindex(bench_p.index).dropna()
        stock_rets = prices.pct_change().dropna()
        bench_rets = bench_p.pct_change().dropna()
        port_rets  = (stock_rets * weights).sum(axis=1)
        port_cum   = (1 + port_rets).cumprod() - 1
        bench_cum  = (1 + bench_rets).cumprod() - 1

        p_ret = float(port_cum.iloc[-1]) * 100
        b_ret = float(bench_cum.iloc[-1]) * 100
        out   = p_ret - b_ret
        p_vol = port_rets.std() * np.sqrt(252) * 100
        b_vol = bench_rets.std() * np.sqrt(252) * 100
        p_sh  = sharpe(port_rets)
        b_sh  = sharpe(bench_rets)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Portfolio Return", f"{p_ret:.2f}%")
        c2.metric("Benchmark Return", f"{b_ret:.2f}%")
        c3.metric("Outperformance",   f"{out:+.2f}%")
        c4.metric("Ann. Volatility",  f"{p_vol:.1f}%", f"vs {b_vol:.1f}%")
        c5.metric("Sharpe Ratio",     f"{p_sh:.2f}",   f"vs {b_sh:.2f}")

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=port_cum.index,  y=port_cum * 100,  name="Portfolio", line=dict(color="#4361ee", width=2.5)))
        fig3.add_trace(go.Scatter(x=bench_cum.index, y=bench_cum * 100, name=bench_in,    line=dict(color="#f77f00", width=2, dash="dot")))
        fig3.update_layout(height=320, margin=dict(t=10, b=10, l=10, r=10),
                           legend=dict(orientation="h", y=1.05), yaxis_title="Cumulative Return (%)")
        st.plotly_chart(fig3, use_container_width=True)

        pc, bc = st.columns(2)
        with pc:
            st.caption("Weights")
            fig4 = px.pie(names=tickers, values=weights,
                          color_discrete_sequence=px.colors.qualitative.Bold)
            fig4.update_layout(height=260, margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig4, use_container_width=True)
        with bc:
            st.caption("1-Year Returns by Stock")
            ind  = [(prices[t].iloc[-1] / prices[t].iloc[0] - 1) * 100 for t in tickers]
            fig5 = go.Figure(go.Bar(
                x=tickers, y=ind,
                marker_color=["#2dc653" if v >= 0 else "#d62828" for v in ind],
                text=[f"{v:.1f}%" for v in ind], textposition="outside"))
            fig5.update_layout(height=260, margin=dict(t=20, b=10, l=10, r=10), yaxis_title="Return (%)")
            st.plotly_chart(fig5, use_container_width=True)

        with st.expander("Written Interpretation"):
            st.markdown(f"""
**Portfolio:** {', '.join(f"{t} ({w*100:.0f}%)" for t, w in zip(tickers, weights))} vs **{bench_in}**

**Performance:** {p_ret:.2f}% vs {b_ret:.2f}% → {"outperformed" if out > 0 else "underperformed"} by {abs(out):.2f}%.

**Risk:** Volatility {p_vol:.1f}% vs {b_vol:.1f}% → {"more" if p_vol > b_vol else "less"} risky than the benchmark.

**Efficiency:** Sharpe {p_sh:.2f} vs {b_sh:.2f} → {"efficient" if p_sh > b_sh else "inefficient"} on a risk-adjusted basis.
            """)
