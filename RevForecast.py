# americas_forecast_dashboard.py
# ------------------------------------------------------------
# Sales Forecasting & Reporting Dashboard (Americas GTM)
# - Synthetic data generator (regions, segments, reps, quotas)
# - Forecast metrics (Commit / Best Case / Pipeline Weighted)
# - Pipeline coverage, product mix, stage views, accuracy trend
# - Drilldowns by region / segment / product / rep
# - Pace to Quota (beside Components), Retention (GRR/NRR), Cohorts
# - Bridge-to-Target REMOVED; Expansion Pipeline KPI REMOVED
# ------------------------------------------------------------

import random
from datetime import datetime, timedelta, date
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
from faker import Faker

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# For custom heatmap palette
from matplotlib.colors import LinearSegmentedColormap

# ---------- Page Configuration ----------
st.set_page_config(
    page_title="Americas GTM | Sales Forecasting & Reporting",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Custom CSS ----------
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.75rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.12);
    }
    .main-title {
        color: white;
        font-size: 2.0rem;
        font-weight: 800;
        text-align: left;
        margin: 0;
        text-shadow: 1px 2px 6px rgba(0,0,0,0.25);
    }
    .subtitle {
        color: rgba(255,255,255,0.95);
        font-size: 1.05rem;
        margin-top: .35rem;
        font-weight: 400;
    }

    /* --- Royal Indigo section headers --- */
    .section-header {
        background: linear-gradient(90deg, #1E1B4B 0%, #312E81 100%);
        padding: .85rem 1.25rem;
        border-radius: 12px;
        margin: 1.25rem 0 .9rem 0;
        border-left: 4px solid #A78BFA;
        box-shadow: 0 2px 10px rgba(0,0,0,0.20);
    }
    .section-title {
        color: #EEF2FF;
        font-size: 1.15rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: .2px;
    }

    div[data-testid="stHorizontalBlock"] > div { gap: .75rem; }
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #ffffff 0%, #f9fafb 100%);
        border: 1px solid #edf1f5;
        padding: .9rem;
        border-radius: 12px;
        box-shadow: 0 2px 14px rgba(0,0,0,0.04);
    }
    .footer-caption { color: #000000 !important; font-weight: 500; }
    .stCaption p { color: #000000 !important; font-weight: 500; }
</style>
""", unsafe_allow_html=True)

# ---------- Helpers ----------
def current_quarter(dt: date) -> str:
    q = (dt.month - 1) // 3 + 1
    return f"Q{q} {dt.year}"

def quarter_bounds(qstr: str):
    q, yr = qstr.split()
    yr = int(yr)
    qi = int(q[1])
    start_month = (qi - 1) * 3 + 1
    start = date(yr, start_month, 1)
    end = start + relativedelta(months=3) - timedelta(days=1)
    return start, end

def prev_quarter(qstr: str) -> str:
    q, y = qstr.split()
    y = int(y); qi = int(q[1])
    if qi == 1:
        return f"Q4 {y-1}"
    return f"Q{qi-1} {y}"

# ---------- Synthetic Data Generation ----------
@st.cache_data(show_spinner=False)
def generate_synthetic_gtm(seed: int = 42,
                           n_accounts: int = 220,
                           n_opps: int = 1600,
                           quarters_back: int = 4,
                           quarters_forward: int = 1):

    random.seed(seed)
    np.random.seed(seed)
    fake = Faker()
    Faker.seed(seed)

    # Dimensions
    regions = ["US", "Canada", "Brazil", "Mexico", "Rest of LATAM"]
    countries_map = {
        "US": ["United States"],
        "Canada": ["Canada"],
        "Brazil": ["Brazil"],
        "Mexico": ["Mexico"],
        "Rest of LATAM": ["Chile", "Colombia", "Argentina", "Peru"]
    }
    segments = ["SMB", "Mid-Market", "Enterprise"]
    products = ["CTV", "Display", "Mobile", "OpenWrap"]

    # Stage probabilities
    stage_probs = {
        "Prospecting": 0.05,
        "Discovery": 0.10,
        "Qualification": 0.20,
        "Proposal": 0.40,
        "Negotiation": 0.65,
        "Legal": 0.80,
        "Commit": 0.90,
        "Closed Won": 1.00,
        "Closed Lost": 0.00
    }

    # Reps
    reps = []
    for r in regions:
        num_reps = 8 if r == "US" else 5
        for _ in range(num_reps):
            reps.append({"rep_id": fake.uuid4(), "rep_name": fake.name(), "region": r})
    reps_df = pd.DataFrame(reps)

    # Accounts
    accounts = []
    for i in range(n_accounts):
        region = random.choice(regions)
        country = random.choice(countries_map[region])
        seg = random.choices(segments, weights=[0.45, 0.35, 0.20])[0]
        accounts.append({
            "account_id": f"ACC-{100000 + i}",
            "account_name": fake.company(),
            "region": region,
            "country": country,
            "segment": seg,
            "is_publisher": random.random() < 0.7,
            "is_agency_or_dsp": random.random() < 0.3
        })
    accounts_df = pd.DataFrame(accounts)

    # Timeline
    today = date.today()
    cur_q = current_quarter(today)
    timeline = []
    for i in range(quarters_back, 0, -1):
        past_date = today - relativedelta(months=3 * i)
        timeline.append(current_quarter(past_date))
    timeline.append(cur_q)
    for i in range(1, quarters_forward + 1):
        future_date = today + relativedelta(months=3 * i)
        timeline.append(current_quarter(future_date))

    # Quotas
    quotas = []
    base_quota_map = {"US": 6_000_000, "Canada": 1_200_000, "Brazil": 1_500_000, "Mexico": 1_000_000, "Rest of LATAM": 1_300_000}
    seg_weight = {"SMB": 0.18, "Mid-Market": 0.34, "Enterprise": 0.48}
    for q in timeline:
        for r in regions:
            reg_total = max(int(np.random.normal(base_quota_map[r], base_quota_map[r] * 0.08)), 100000)
            for s in segments:
                quota_amt = max(int(reg_total * seg_weight[s] * np.random.uniform(0.95, 1.05)), 1000)
                quotas.append({"quarter": q, "region": r, "segment": s, "quota": quota_amt})
    quotas_df = pd.DataFrame(quotas)

    # Opportunities
    all_quarter_bounds = {q: quarter_bounds(q) for q in timeline}

    def pick_stage_by_close_proximity(close_dt: date):
        delta_days = (close_dt - today).days
        if delta_days < -5:
            return random.choices(["Closed Won", "Closed Lost"], weights=[0.6, 0.4])[0]
        elif delta_days <= 15:
            return random.choices(["Negotiation", "Legal", "Commit"], weights=[0.35, 0.35, 0.30])[0]
        elif delta_days <= 45:
            return random.choices(["Qualification", "Proposal", "Negotiation"], weights=[0.25, 0.45, 0.30])[0]
        elif delta_days <= 90:
            return random.choices(["Prospecting", "Discovery", "Qualification", "Proposal"],
                                  weights=[0.15, 0.25, 0.35, 0.25])[0]
        else:
            return random.choices(["Prospecting", "Discovery", "Qualification"], weights=[0.50, 0.30, 0.20])[0]

    # Spread close dates across quarters
    fake_local = Faker()
    q_dates = []
    for q in timeline:
        qs, qe = all_quarter_bounds[q]
        weight = 1.5 if q == cur_q else (1.2 if q in timeline[-(quarters_forward+1):] else 1.0)
        quarter_opps = max(int(n_opps * weight / (len(timeline) * 1.2)), 1)
        for _ in range(quarter_opps):
            q_dates.append(fake_local.date_between(qs, qe))
    while len(q_dates) < n_opps:
        cur_qs, cur_qe = all_quarter_bounds[cur_q]
        q_dates.append(fake_local.date_between(cur_qs, cur_qe))
    random.shuffle(q_dates)
    q_dates = q_dates[:n_opps]

    seg_acv_mu = {"SMB": 20_000, "Mid-Market": 60_000, "Enterprise": 180_000}
    prod_lift = {"CTV": 1.4, "Display": 1.0, "Mobile": 0.9, "OpenWrap": 1.2}

    opps = []
    for i in range(n_opps):
        acct = accounts_df.sample(1).iloc[0]
        region = acct["region"]
        segment = acct["segment"]
        rep = reps_df[reps_df["region"] == region].sample(1).iloc[0]
        product = random.choices(products, weights=[0.32, 0.32, 0.20, 0.16])[0]
        acv_mu = seg_acv_mu[segment] * prod_lift[product]
        acv = max(np.random.lognormal(mean=np.log(acv_mu), sigma=0.6), 5_000)
        acv = float(np.clip(acv, 5_000, 600_000))
        created = today - timedelta(days=random.randint(5, 200))
        close_dt = q_dates[i]
        stage = pick_stage_by_close_proximity(close_dt)
        prob = stage_probs[stage]
        if stage in ["Commit", "Legal"]:
            category = "Commit"
        elif stage in ["Negotiation", "Proposal"]:
            category = "Best Case"
        elif stage in ["Prospecting", "Discovery", "Qualification"]:
            category = "Pipeline"
        else:
            category = "Closed"
        is_new_logo = random.random() < 0.55
        is_upsell = (not is_new_logo) and (random.random() < 0.6)
        cur_q_start, cur_q_end = all_quarter_bounds[cur_q]
        slipped = category == "Commit" and random.random() < 0.35 and close_dt > cur_q_end

        opps.append({
            "opp_id": f"OPP-{100000 + i}",
            "account_id": acct["account_id"],
            "account_name": acct["account_name"],
            "region": region,
            "country": acct["country"],
            "segment": segment,
            "product": product,
            "rep_id": rep["rep_id"],
            "rep_name": rep["rep_name"],
            "acv": round(acv, 2),
            "stage": stage,
            "stage_prob": prob,
            "amount_weighted": round(acv * prob, 2),
            "created_date": pd.to_datetime(created),
            "close_date": pd.to_datetime(close_dt),
            "close_quarter": current_quarter(close_dt),
            "category": category,
            "is_new_logo": is_new_logo,
            "is_upsell": is_upsell,
            "slipped": slipped
        })
    opps_df = pd.DataFrame(opps)

    # Historical forecast accuracy
    hist_rows = []
    past_quarters = timeline[:-quarters_forward]
    for q in past_quarters:
        quarter_opps = opps_df[opps_df["close_quarter"] == q]
        if not quarter_opps.empty:
            weighted = quarter_opps["amount_weighted"].sum()
            commit = quarter_opps[quarter_opps["category"] == "Commit"]["acv"].sum()
            best = quarter_opps[quarter_opps["category"] == "Best Case"]["acv"].sum() * 0.65
            snapshot = (commit + best + weighted * 0.15) * np.random.uniform(0.92, 1.08)
            hist_rows.append({"quarter": q, "forecast_snapshot": snapshot})
    hist_df = pd.DataFrame(hist_rows)

    # Actuals by quarter (for accuracy)
    actuals_q = (opps_df[opps_df["stage"].eq("Closed Won")]
                 .groupby("close_quarter", as_index=False)["acv"].sum()
                 .rename(columns={"close_quarter": "quarter", "acv": "actual_bookings"}))
    accuracy = actuals_q.merge(hist_df, on="quarter", how="left").fillna({"forecast_snapshot": 0.0})
    if not accuracy.empty:
        accuracy["abs_error"] = (accuracy["forecast_snapshot"] - accuracy["actual_bookings"]).abs()
        accuracy["mape"] = np.where(
            accuracy["actual_bookings"] > 0,
            accuracy["abs_error"] / accuracy["actual_bookings"],
            np.nan
        )

    return {
        "reps": reps_df,
        "accounts": accounts_df,
        "opps": opps_df,
        "quotas": quotas_df,
        "accuracy": accuracy,
        "timeline": timeline,
        "current_quarter": cur_q
    }

# ---------- Sidebar Controls ----------
with st.sidebar:
    st.header("⚙️ Controls")
    seed = st.number_input("Random seed", min_value=1, value=42, step=1,
                           help="Change to regenerate a brand-new synthetic dataset.")
    quarters_back = st.slider("Past quarters (history)", 3, 6, 4)
    quarters_forward = st.slider("Forecast horizon (quarters ahead)", 1, 2, 1)
    n_accounts = st.slider("Number of Accounts", 100, 500, 220, step=20)
    n_opps = st.slider("Number of Opportunities", 800, 4000, 1600, step=100)
    show_downloads = st.checkbox("Enable CSV downloads", value=False)

# Generate data
try:
    data = generate_synthetic_gtm(
        seed=seed,
        n_accounts=n_accounts,
        n_opps=n_opps,
        quarters_back=quarters_back,
        quarters_forward=quarters_forward
    )
except Exception as e:
    st.error(f"Error generating data: {str(e)}")
    st.stop()

# Extract data components
reps_df = data["reps"]
accounts_df = data["accounts"]
opps_df = data["opps"]
quotas_df = data["quotas"]
accuracy = data["accuracy"]
timeline = data["timeline"]
current_q = data["current_quarter"]

# Ensure datetime dtypes (global safety)
opps_df["close_date"] = pd.to_datetime(opps_df["close_date"], errors="coerce")
opps_df["created_date"] = pd.to_datetime(opps_df["created_date"], errors="coerce")

# ---------- Header ----------
st.markdown(f"""
<div class="main-header">
  <div class="main-title">Americas GTM — Sales Forecasting & Reporting</div>
  <div class="subtitle">Executive roll-up with drill-downs by Region • Segment • Product • Rep</div>
</div>
""", unsafe_allow_html=True)

# ---------- Filters ----------
regions = ["US", "Canada", "Brazil", "Mexico", "Rest of LATAM"]
segments = ["SMB", "Mid-Market", "Enterprise"]
products = ["CTV", "Display", "Mobile", "OpenWrap"]

f1, f2, f3, f4 = st.columns([1.2, 1.2, 1.4, 1])
with f1:
    sel_quarter = st.selectbox("Quarter", options=timeline, index=timeline.index(current_q))
with f2:
    sel_regions = st.multiselect("Regions", regions, default=regions)
with f3:
    sel_segments = st.multiselect("Segments", segments, default=segments)
with f4:
    sel_products = st.multiselect("Products", products, default=products)

# Apply filters
mask = (
    opps_df["close_quarter"].eq(sel_quarter) &
    opps_df["region"].isin(sel_regions) &
    opps_df["segment"].isin(sel_segments) &
    opps_df["product"].isin(sel_products)
)
view = opps_df.loc[mask].copy()
quota_view = quotas_df[
    (quotas_df["quarter"].eq(sel_quarter)) &
    (quotas_df["region"].isin(sel_regions)) &
    (quotas_df["segment"].isin(sel_segments))
].copy()

# ---------- KPI Calculations ----------
commit_amt = view.loc[
    view["category"].eq("Commit") &
    ~view["stage"].isin(["Closed Won", "Closed Lost"]), "acv"
].sum()
best_amt = view.loc[view["category"].eq("Best Case"), "acv"].sum()
weighted_amt = view["amount_weighted"].sum()
booked_amt = view.loc[view["stage"].eq("Closed Won"), "acv"].sum()

quarter_quota = quota_view["quota"].sum()
remaining_quota = max(quarter_quota - booked_amt, 0.0)

pipeline_total = view.loc[~view["stage"].isin(["Closed Won", "Closed Lost"]), "acv"].sum()
coverage_ratio = (pipeline_total / remaining_quota) if remaining_quota > 0 else 0.0

composite_forecast = commit_amt + 0.6 * best_amt + 0.2 * weighted_amt

won = view[view["stage"].eq("Closed Won")].copy()
if not won.empty:
    won["cycle_days"] = (won["close_date"] - won["created_date"]).dt.days
    avg_cycle = float(won["cycle_days"].mean())
    avg_deal = float(won["acv"].mean())
    win_rate = len(won) / max(len(view), 1)
else:
    avg_cycle, avg_deal, win_rate = 0.0, 0.0, 0.0

slipped_val = view.loc[view["slipped"], "acv"].sum()

# ---------- Executive Summary ----------
st.markdown('<div class="section-header"><div class="section-title">Executive Summary</div></div>',
            unsafe_allow_html=True)
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Quota (Quarter)", f"${quarter_quota:,.0f}")
k2.metric("Bookings To-Date", f"${booked_amt:,.0f}")
k3.metric("Composite Forecast", f"${composite_forecast:,.0f}")
k4.metric("Remaining Quota", f"${remaining_quota:,.0f}")
k5.metric("Pipeline Coverage", f"{coverage_ratio:.2f}×")
k6.metric("Slippage (Commit pushed)", f"${slipped_val:,.0f}")

# ---------- Forecast vs. Quota (by Components) & Pace to Quota ----------
left, right = st.columns([1.05, 1.15])

with left:
    st.markdown('<div class="section-header"><div class="section-title">Forecast vs. Quota (by Components)</div></div>',
                unsafe_allow_html=True)
    best_w = 0.6 * best_amt
    weighted_w = 0.2 * weighted_amt
    projected = booked_amt + commit_amt + best_w + weighted_w

    bars = pd.DataFrame({
        "Component": [
            "Closed Won (This Quarter)",
            "Commit / Legal (Must Close)",
            "Negotiation / Proposal",
            "Early Pipeline",
            "Forecast Total"
        ],
        "Value": [booked_amt, commit_amt, best_w, weighted_w, projected]
    })

    fig = go.Figure()
    colors = ["#2ecc71", "#3498db", "#9b59b6", "#95a5a6", "#34495e"]
    for i, row in bars.iterrows():
        fig.add_trace(go.Bar(
            x=[row["Value"]],
            y=[row["Component"]],
            orientation='h',
            marker=dict(color=colors[i]),
            name=row["Component"],
            hovertemplate="<b>%{y}</b><br>$%{x:,.0f}<extra></extra>",
            showlegend=False
        ))
    fig.add_vline(
        x=quarter_quota,
        line_dash="dot",
        line_color="#e74c3c",
        annotation_text="Quarter Quota Target",
        annotation_position="top right"
    )
    fig.update_layout(height=360, template="plotly_white", margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.markdown('<div class="section-header"><div class="section-title">Pace to Quota (Trajectory)</div></div>',
                unsafe_allow_html=True)
    qs, qe = quarter_bounds(sel_quarter)
    qs_ts = pd.Timestamp(qs)
    qe_ts = pd.Timestamp(qe)
    today_ts = pd.Timestamp(date.today())
    today_cutoff = min(today_ts, qe_ts)

    bookings_daily = (
        opps_df
        .loc[(opps_df["stage"].eq("Closed Won")) & (opps_df["close_date"].between(qs_ts, today_cutoff))]
        .groupby(opps_df["close_date"].dt.date)["acv"].sum()
        .reindex(pd.date_range(qs_ts, today_cutoff).date, fill_value=0.0)
        .cumsum()
        .rename("Actual")
        .reset_index()
    )
    bookings_daily.columns = ["date", "Actual"]

    workdays = pd.bdate_range(qs_ts, qe_ts, freq="B")
    expected_dates = pd.date_range(qs_ts, today_cutoff, freq="D")
    counts = np.array([(workdays <= d).sum() for d in expected_dates])
    per_day_linear = quarter_quota / max(len(workdays), 1)
    expected_linear = pd.DataFrame({"date": expected_dates.date, "Expected": per_day_linear * counts})

    pace_df = bookings_daily.merge(expected_linear, on="date", how="outer").sort_values("date")
    pace_df[["Actual", "Expected"]] = pace_df[["Actual", "Expected"]].fillna(method="ffill").fillna(0.0)

    pfig = go.Figure()
    pfig.add_trace(go.Scatter(x=pace_df["date"], y=pace_df["Actual"], mode="lines", name="Actual"))
    pfig.add_trace(go.Scatter(x=pace_df["date"], y=pace_df["Expected"], mode="lines", name="Expected"))
    pfig.update_layout(
        height=360, template="plotly_white", margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Date", yaxis_title="Cumulative Bookings ($)"
    )
    st.plotly_chart(pfig, use_container_width=True)

    pace_delta = (pace_df["Actual"].iloc[-1] - pace_df["Expected"].iloc[-1]) if not pace_df.empty else 0.0
    st.caption(f"Pace delta vs expected: **${pace_delta:,.0f}** ({'ahead' if pace_delta>=0 else 'behind'})")

# ---------- Open Pipeline by Stage & Product Mix (ACV) ----------
c3, c4 = st.columns([1.2, 0.8])

with c3:
    st.markdown('<div class="section-header"><div class="section-title">Open Pipeline by Stage</div></div>',
                unsafe_allow_html=True)
    open_pipe = view[~view["stage"].isin(["Closed Won", "Closed Lost"])]
    if not open_pipe.empty:
        stage_summary = (open_pipe.groupby("stage", as_index=False)["acv"]
                         .sum().sort_values("acv", ascending=True))
        # 25% darker turquoise
        fig2 = go.Figure(go.Bar(
            x=stage_summary["acv"], y=stage_summary["stage"], orientation="h",
            marker=dict(color="#009B9D"),
            hovertemplate="<b>%{y}</b><br>$%{x:,.0f}<extra></extra>"
        ))
        fig2.update_layout(height=360, template="plotly_white", margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No open opportunities in this selection.")

with c4:
    st.markdown('<div class="section-header"><div class="section-title">Product Mix (ACV)</div></div>',
                unsafe_allow_html=True)
    if not view.empty:
        prod_summary = view.groupby("product", as_index=False)["acv"].sum()
        pfig = px.pie(prod_summary, names="product", values="acv", hole=0.45)
        pfig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(pfig, use_container_width=True)
    else:
        st.info("No data for product mix.")

# ---------- Forecast Accuracy & Rep Leaderboard ----------
c7, c8 = st.columns([1.1, 0.9])

with c7:
    st.markdown('<div class="section-header"><div class="section-title">Forecast Accuracy</div></div>',
                unsafe_allow_html=True)
    if not accuracy.empty:
        acc_quarters = accuracy["quarter"].tolist()
        if sel_quarter in acc_quarters:
            cut_idx = acc_quarters.index(sel_quarter)
            acc_view = accuracy.iloc[:cut_idx + 1]
        else:
            acc_view = accuracy.copy()
        fig_acc = go.Figure()
        fig_acc.add_trace(go.Bar(name="Actual Bookings",
                                 x=acc_view["quarter"], y=acc_view["actual_bookings"],
                                 marker_color="#2ecc71"))
        fig_acc.add_trace(go.Scatter(name="Forecast Snapshot",
                                     x=acc_view["quarter"], y=acc_view["forecast_snapshot"],
                                     mode="lines+markers", line=dict(color="#e74c3c"), marker=dict(size=8)))
        fig_acc.update_layout(template="plotly_white", height=360,
                              legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                              margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_acc, use_container_width=True)
        valid_mape = acc_view.loc[acc_view["actual_bookings"] > 0, "mape"]
        if not valid_mape.empty:
            avg_mape = float(valid_mape.mean())
            st.caption(f"Average MAPE across shown quarters: **{(avg_mape * 100):.1f}%**")
        else:
            st.caption("MAPE: No historical data available")
    else:
        st.info("No historical accuracy samples available.")

with c8:
    st.markdown('<div class="section-header"><div class="section-title">Rep Leaderboard</div></div>',
                unsafe_allow_html=True)
    if not view.empty:
        rep_data = []
        for rep_name in view["rep_name"].unique():
            rep_view = view[view["rep_name"] == rep_name]
            rep_region = rep_view["region"].iloc[0]
            booked = rep_view[rep_view["stage"] == "Closed Won"]["acv"].sum()
            commit = rep_view[(rep_view["category"] == "Commit") &
                              (~rep_view["stage"].isin(["Closed Won", "Closed Lost"]))]["acv"].sum()
            weighted_r = rep_view["amount_weighted"].sum() * 0.2
            total_forecast = booked + commit + weighted_r
            rep_data.append({"rep_name": rep_name, "region": rep_region, "forecast": total_forecast})
        if rep_data:
            rep_df = pd.DataFrame(rep_data).sort_values("forecast", ascending=False).head(12)
            rfig = go.Figure(go.Bar(
                x=rep_df["forecast"],
                y=[f"{row['rep_name']} — {row['region']}" for _, row in rep_df.iterrows()],
                orientation="h",
                marker=dict(color="#1abc9c"),
                hovertemplate="<b>%{y}</b><br>$%{x:,.0f}<extra></extra>"
            ))
            rfig.update_layout(height=360, template="plotly_white", margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(rfig, use_container_width=True)
        else:
            st.info("No rep data available.")
    else:
        st.info("No rep data in this selection.")

# ---------- Customer Health, Retention & Expansion ----------
st.markdown('<div class="section-header"><div class="section-title">Customer Health, Retention & Expansion</div></div>',
            unsafe_allow_html=True)

pq = prev_quarter(sel_quarter)
rev_prev = (opps_df
            .loc[(opps_df["stage"].eq("Closed Won")) & (opps_df["close_quarter"].eq(pq))]
            .groupby("account_id", as_index=False)["acv"].sum()
            .rename(columns={"acv": "prev_rev"}))
rev_curr = (opps_df
            .loc[(opps_df["stage"].eq("Closed Won")) & (opps_df["close_quarter"].eq(sel_quarter))]
            .groupby("account_id", as_index=False)["acv"].sum()
            .rename(columns={"acv": "curr_rev"}))
rev_join = rev_prev.merge(rev_curr, on="account_id", how="left").fillna({"curr_rev": 0.0})

starting = float(rev_join["prev_rev"].sum())
retained = float(np.minimum(rev_join["prev_rev"], rev_join["curr_rev"]).sum())
expansion = float(np.maximum(rev_join["curr_rev"] - rev_join["prev_rev"], 0.0).sum())
contraction = float(np.maximum(rev_join["prev_rev"] - rev_join["curr_rev"], 0.0).sum())

grr = (retained / starting) if starting > 0 else np.nan
nrr = ((retained + expansion) / starting) if starting > 0 else np.nan
logo_churn = ((rev_join["curr_rev"] == 0).sum() / len(rev_join)) if len(rev_join) > 0 else np.nan

open_pipe_curr = (opps_df
                  .loc[(~opps_df["stage"].isin(["Closed Won", "Closed Lost"])) &
                       (opps_df["close_quarter"].eq(sel_quarter))]
                  .groupby("account_id", as_index=False)["acv"].sum()
                  .rename(columns={"acv": "open_pipe"}))
risk_df = (rev_join.merge(open_pipe_curr, on="account_id", how="left").fillna({"open_pipe": 0.0}))
at_risk_arr = float(risk_df.loc[(risk_df["curr_rev"] == 0) & (risk_df["open_pipe"] == 0), "prev_rev"].sum())

g1, g2, g3, g4 = st.columns(4)
g1.metric("GRR", f"{(grr * 100):.1f}%" if pd.notna(grr) else "—")
g2.metric("NRR", f"{(nrr * 100):.1f}%" if pd.notna(nrr) else "—")
g3.metric("Logo Churn", f"{(logo_churn * 100):.1f}%" if pd.notna(logo_churn) else "—")
g4.metric("At-Risk ARR (proxy)", f"${at_risk_arr:,.0f}")

ret_tbl = pd.DataFrame({
    "Metric": ["Starting Base (prev Q)", "Retained", "Expansion", "Contraction"],
    "Value": [starting, retained, expansion, contraction]
})
st.dataframe(ret_tbl.style.format({"Value": "$ {:,.0f}"}), use_container_width=True)

# Cohort Retention (first-won quarter → revenue over time)
st.markdown("**Cohort Revenue Heatmap**")

# Reddish monochrome gradient aligned to #e74c3c
reddish_cmap = LinearSegmentedColormap.from_list(
    "ExecReds",
    ["#FFECEC", "#F5A6A0", "#E74C3C", "#B83E31", "#7A261F"]
)

first_wins = (opps_df.loc[opps_df["stage"].eq("Closed Won")]
              .sort_values("close_date")
              .groupby("account_id", as_index=False).first()[["account_id", "close_quarter"]]
              .rename(columns={"close_quarter": "cohort"}))
won_rev = (opps_df.loc[opps_df["stage"].eq("Closed Won")]
           .groupby(["account_id", "close_quarter"], as_index=False)["acv"].sum()
           .rename(columns={"close_quarter": "quarter"}))
cohort_rev = (first_wins.merge(won_rev, on="account_id", how="left")
              .groupby(["cohort", "quarter"], as_index=False)["acv"].sum())
if not cohort_rev.empty:
    def sort_key(q):
        qn, y = q.split()
        return (int(y), int(qn[1]))
    cohort_pivot = (cohort_rev
                    .pivot(index="cohort", columns="quarter", values="acv")
                    .reindex(index=sorted(cohort_rev["cohort"].unique(), key=sort_key),
                             columns=timeline)
                    .fillna(0.0))
    st.dataframe(cohort_pivot.style.format("$ {:,.0f}").background_gradient(cmap=reddish_cmap),
                 use_container_width=True)
else:
    st.info("No cohort data available yet.")

# ---------- Sales Velocity & Win Rate ----------
st.markdown('<div class="section-header"><div class="section-title">Sales Velocity & Win Rate Analysis</div></div>',
            unsafe_allow_html=True)
c9, c10, c11 = st.columns(3)
with c9:
    st.metric("Average Deal Size", f"${avg_deal:,.0f}" if avg_deal > 0 else "—",
              help="Average ACV of won deals in current selection")
with c10:
    st.metric("Average Sales Cycle", f"{avg_cycle:.0f} days" if avg_cycle > 0 else "—",
              help="Average days from created to close for won deals")
with c11:
    st.metric("Win Rate", f"{win_rate:.1%}", help="Percentage of opportunities marked as Closed Won")

# ---------- Segment Performance ----------
st.markdown('<div class="section-header"><div class="section-title">Segment Performance Analysis</div></div>',
            unsafe_allow_html=True)
if not view.empty:
    segment_analysis = []
    for segment in sel_segments:
        seg_view = view[view["segment"] == segment]
        seg_quota = quota_view[quota_view["segment"] == segment]["quota"].sum()
        if not seg_view.empty:
            booked = seg_view[seg_view["stage"] == "Closed Won"]["acv"].sum()
            open_pipeline = seg_view[~seg_view["stage"].isin(["Closed Won", "Closed Lost"])]["acv"].sum()
            avg_deal_size = seg_view["acv"].mean()
            opportunity_count = len(seg_view)
            won_count = len(seg_view[seg_view["stage"] == "Closed Won"])
            win_rate_seg = won_count / opportunity_count if opportunity_count > 0 else 0
            segment_analysis.append({
                "Segment": segment,
                "Quota": seg_quota,
                "Booked": booked,
                "Open Pipeline": open_pipeline,
                "Avg Deal Size": avg_deal_size,
                "Opp Count": opportunity_count,
                "Win Rate": win_rate_seg
            })
    if segment_analysis:
        seg_df = pd.DataFrame(segment_analysis)
        st.dataframe(
            seg_df.style.format({
                "Quota": "$ {:,.0f}",
                "Booked": "$ {:,.0f}",
                "Open Pipeline": "$ {:,.0f}",
                "Avg Deal Size": "$ {:,.0f}",
                "Opp Count": "{:,}",
                "Win Rate": "{:.1%}"
            }),
            use_container_width=True
        )
        if show_downloads:
            st.download_button(
                "Download Segment Analysis (CSV)",
                seg_df.to_csv(index=False).encode("utf-8"),
                file_name=f"segment_analysis_{sel_quarter}.csv",
                mime="text/csv"
            )
else:
    st.info("No data available for segment analysis.")

# ---------- Raw Data Inspector ----------
with st.expander("🔎 Inspect Raw Data (current filters)"):
    if not view.empty:
        st.write(f"**Filtered Opportunities:** {len(view):,}")
        st.write(f"**Total ACV:** ${view['acv'].sum():,.0f}")
        st.write(f"**Weighted Pipeline:** ${view['amount_weighted'].sum():,.0f}")
        display_cols = ["opp_id", "account_name", "region", "segment", "product",
                        "rep_name", "stage", "acv", "amount_weighted", "close_date", "category"]
        st.dataframe(
            view[display_cols].sort_values("close_date").reset_index(drop=True),
            use_container_width=True
        )
        if show_downloads:
            st.download_button(
                "Download Filtered Opportunities (CSV)",
                view.to_csv(index=False).encode("utf-8"),
                file_name=f"opps_{sel_quarter}_filtered.csv",
                mime="text/csv"
            )
    else:
        st.info("No opportunities match the current filter criteria.")

# ---------- Footer ----------
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('<p class="footer-caption">📊 <strong>Data Quality</strong></p>', unsafe_allow_html=True)
    st.markdown(f'<p class="footer-caption">• {len(opps_df):,} total opportunities</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="footer-caption">• {len(accounts_df):,} accounts across {len(regions)} regions</p>', unsafe_allow_html=True)
with col2:
    st.markdown('<p class="footer-caption">⚡ <strong>Performance Metrics</strong></p>', unsafe_allow_html=True)
    if not view.empty:
        closed_rate = len(view[view["stage"].isin(["Closed Won", "Closed Lost"])]) / len(view)
        st.markdown(f'<p class="footer-caption">• {closed_rate:.1%} opportunities closed</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="footer-caption">• ${view["acv"].sum():,.0f} total pipeline value</p>', unsafe_allow_html=True)
with col3:
    st.markdown('<p class="footer-caption">🔄 <strong>Last Updated</strong></p>', unsafe_allow_html=True)
    st.markdown(f'<p class="footer-caption">• Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>', unsafe_allow_html=True)
    st.markdown('<p class="footer-caption">• Synthetic data for demo purposes</p>', unsafe_allow_html=True)

st.markdown("""
<p class="footer-caption">
<strong>Note:</strong> This dashboard uses synthetic data for demonstration. 
Adjust the seed and parameters in the sidebar to explore different scenarios.
The forecast method combines committed opportunities, probability-weighted best case, 
and pipeline coverage for comprehensive revenue visibility.
</p>
""", unsafe_allow_html=True)
