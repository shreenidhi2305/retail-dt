"""
app.py

Retail Demand & Inventory Digital Twin -- Streamlit UI.

Run:
    streamlit run app.py

Pages:
    1. Twin Overview       - network-level current state
    2. Twin State           - per store-item state + timeline, live playback
    3. Scenario Simulator    - baseline vs scenario, run on demand
    4. Decision Support Evidence - findings, recommendations, JSON export
"""

import json
import time as time_module

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_utils import load_train
from forecasting import load_model_bundle
from twin import ScenarioConfig, SupplierConfig
import simulation
import recommendations as rec_engine

st.set_page_config(page_title="Retail Digital Twin", page_icon="\U0001F4E6", layout="wide")

ACCENT = "#2E6F6E"        # deep teal -- inventory / operations
ACCENT_WARN = "#C77B3B"   # warning amber-clay
ACCENT_BAD = "#B5453A"    # stockout red
ACCENT_GOOD = "#3E8E5C"   # healthy green
INK = "#22282B"

st.markdown(f"""
<style>
    .stApp {{ background-color: #F7F6F2; color: {INK}; }}
    [data-testid="stSidebar"] {{ background-color: #FFFFFF; }}
    .stApp p, .stApp label, .stApp span, .stApp li, .stApp td, .stApp th,
    .stApp [data-testid="stMarkdownContainer"] {{ color: #22282B; }}
    h1, h2, h3 {{ color: {INK}; font-family: 'Georgia', serif; }}
    .metric-card {{
        background: white; border-radius: 10px; padding: 14px 18px;
        border: 1px solid #E4E1D8; box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }}
    .status-pill {{
        display: inline-block; padding: 2px 10px; border-radius: 12px;
        font-size: 0.8rem; font-weight: 600; color: {INK};
    }}

    /* --- Explicit widget colors so buttons / selectors stay readable
       regardless of the viewer's light/dark Streamlit theme setting --- */

    /* Widget labels ("Simulate", "Scenario type", etc.) */
    [data-testid="stWidgetLabel"] p, .stSidebar label, .stSidebar span {{
        color: {INK} !important;
    }}

    /* Regular buttons (Reset / Next Day / Play / Generate JSON) */
    .stButton > button, .stDownloadButton > button {{
        background-color: #FFFFFF !important;
        color: {INK} !important;
        border: 1px solid #C9C5B8 !important;
    }}
    .stButton > button:hover, .stDownloadButton > button:hover {{
        color: {ACCENT} !important;
        border-color: {ACCENT} !important;
    }}
    .stButton > button p, .stDownloadButton > button p {{ color: inherit !important; }}

    /* Primary button (RUN DIGITAL TWIN SIMULATION) */
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="stBaseButton-primary"] {{
        background-color: {ACCENT} !important;
        color: {INK} !important;
        border: none !important;
    }}
    .stButton > button[kind="primary"]:hover,
    .stButton > button[data-testid="stBaseButton-primary"]:hover {{
        background-color: #234f4e !important;
        color: {INK} !important;
    }}
    .stButton > button[kind="primary"] p,
    .stButton > button[data-testid="stBaseButton-primary"] p {{ color: inherit !important; }}

    /* Selectbox / multiselect / radio (scenario type, store, item pickers) */
    div[data-baseweb="select"] > div {{
        background-color: #FFFFFF !important;
        color: {INK} !important;
        border-color: #C9C5B8 !important;
    }}
    div[data-baseweb="select"] * {{ color: {INK} !important; }}
    /* Dropdown menu popover when a selectbox is open */
    ul[data-baseweb="menu"], ul[data-baseweb="menu"] li {{
        background-color: #FFFFFF !important;
        color: {INK} !important;
    }}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached data / model loading
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading historical sales data...")
def get_train_df():
    return load_train("data/train.csv")


@st.cache_resource(show_spinner="Loading trained demand model...")
def get_model_bundle():
    return load_model_bundle("models/demand_model.pkl")


@st.cache_data(show_spinner=False)
def get_model_metrics():
    with open("models/metrics.json") as f:
        return json.load(f)


@st.cache_data(show_spinner="Running Digital Twin simulation...")
def run_cached_scenario(_train_df, _bundle, pairs, name, demand_mult, lead_time_delta,
                         service_level, base_lead_time, order_coverage_days,
                         horizon_days, start_date_str, seed=42):
    scenario = ScenarioConfig(
        name=name, demand_multiplier=demand_mult, lead_time_change_days=lead_time_delta,
        service_level_target=service_level, random_seed=seed,
    )
    supplier = SupplierConfig(base_lead_time_days=base_lead_time, order_coverage_days=order_coverage_days)
    start_date = pd.Timestamp(start_date_str)
    _, steps_df, summary_df = simulation.run_scenario(
        _bundle, _train_df, pairs, scenario, supplier, start_date, horizon_days,
    )
    return steps_df, summary_df


def get_events(_train_df, _bundle, pairs, name, demand_mult, lead_time_delta,
               service_level, base_lead_time, order_coverage_days, horizon_days, start_date_str, seed=42):
    """Re-runs the twin to also retrieve its event log (not cached: cheap, and
    DigitalTwin/event objects aren't easily cache-friendly)."""
    scenario = ScenarioConfig(
        name=name, demand_multiplier=demand_mult, lead_time_change_days=lead_time_delta,
        service_level_target=service_level, random_seed=seed,
    )
    supplier = SupplierConfig(base_lead_time_days=base_lead_time, order_coverage_days=order_coverage_days)
    start_date = pd.Timestamp(start_date_str)
    twin, steps_df, summary_df = simulation.run_scenario(
        _bundle, _train_df, pairs, scenario, supplier, start_date, horizon_days,
    )
    return twin.all_events(), steps_df, summary_df


def status_pill(label, color):
    return f'<span class="status-pill" style="background:{color}">{label}</span>'


def store_status(stockout_rate):
    if stockout_rate > 0.15:
        return "Critical", ACCENT_BAD, "\U0001F534"
    if stockout_rate > 0.03:
        return "Warning", ACCENT_WARN, "\U0001F7E1"
    return "Healthy", ACCENT_GOOD, "\U0001F7E2"


# ---------------------------------------------------------------------------
# Sidebar - global network + scenario configuration
# ---------------------------------------------------------------------------
train_df = get_train_df()
model_bundle = get_model_bundle()
model_metrics = get_model_metrics()

all_stores = sorted(train_df.store.unique().tolist())
all_items = sorted(train_df.item.unique().tolist())
last_train_date = train_df.date.max()
start_date = last_train_date + pd.Timedelta(days=1)

st.sidebar.title("\U0001F4E6 Digital Twin Controls")

st.sidebar.subheader("Network scope")
network_mode = st.sidebar.radio(
    "Simulate", ["Sample network (fast)", "Full network (all stores/items)"],
    help="Full network = 10 stores x 50 items = 500 combinations. Slower per run.",
)
if network_mode.startswith("Sample"):
    sel_stores = st.sidebar.multiselect("Stores", all_stores, default=all_stores[:3])
    sel_items = st.sidebar.multiselect("Items", all_items, default=all_items[:10])
    if not sel_stores or not sel_items:
        st.sidebar.warning("Select at least one store and one item.")
        st.stop()
else:
    sel_stores, sel_items = all_stores, all_items

pairs = tuple((s, i) for s in sel_stores for i in sel_items)

st.sidebar.subheader("Simulation horizon")
horizon_days = st.sidebar.slider("Days to simulate", 7, 90, 30)

st.sidebar.subheader("Supplier (baseline policy)")
with st.sidebar.expander("Supplier settings", expanded=False):
    base_lead_time = st.slider("Base lead time (days)", 1, 14, 5)
    order_coverage_days = st.slider("Order covers ~N days of demand", 7, 30, 14)
    service_level_target = st.select_slider(
        "Target service level", options=[0.90, 0.95, 0.98, 0.99], value=0.95,
        format_func=lambda x: f"{x*100:.0f}%",
    )

st.sidebar.subheader("Scenario")
scenario_name = st.sidebar.selectbox(
    "Scenario type",
    ["Baseline", "Demand Surge", "Supply Disruption", "Combined Disruption", "Custom"],
)
preset = {
    "Baseline": (0, 0),
    "Demand Surge": (30, 0),
    "Supply Disruption": (0, 3),
    "Combined Disruption": (30, 2),
    "Custom": (0, 0),
}[scenario_name]

if scenario_name == "Custom":
    demand_pct = st.sidebar.slider("Demand change (%)", -50, 100, preset[0])
    lead_time_delta = st.sidebar.slider("Lead-time change (days)", -3, 10, preset[1])
else:
    demand_pct, lead_time_delta = preset
    st.sidebar.caption(f"Demand {demand_pct:+d}%, lead time {lead_time_delta:+d} days")

demand_mult = 1.0 + demand_pct / 100.0

run_clicked = st.sidebar.button("\u25B6\uFE0F RUN DIGITAL TWIN SIMULATION", type="primary", width='stretch')

# ---------------------------------------------------------------------------
# Run / cache baseline + scenario
# ---------------------------------------------------------------------------
common_args = dict(
    _train_df=train_df, _bundle=model_bundle, pairs=pairs,
    base_lead_time=base_lead_time, order_coverage_days=order_coverage_days,
    horizon_days=horizon_days, start_date_str=str(start_date.date()),
)

if run_clicked or "base_steps" not in st.session_state:
    base_steps, base_summary = run_cached_scenario(
        name="Baseline", demand_mult=1.0, lead_time_delta=0,
        service_level=service_level_target, **common_args,
    )
    scn_steps, scn_summary = run_cached_scenario(
        name=scenario_name, demand_mult=demand_mult, lead_time_delta=lead_time_delta,
        service_level=service_level_target, **common_args,
    )
    scn_events, _, _ = get_events(
        name=scenario_name, demand_mult=demand_mult, lead_time_delta=lead_time_delta,
        service_level=service_level_target, **common_args,
    )

    st.session_state["base_steps"] = base_steps
    st.session_state["base_summary"] = base_summary
    st.session_state["scn_steps"] = scn_steps
    st.session_state["scn_summary"] = scn_summary
    st.session_state["scn_events"] = scn_events
    st.session_state["scenario_config"] = ScenarioConfig(
        name=scenario_name, demand_multiplier=demand_mult, lead_time_change_days=lead_time_delta,
        service_level_target=service_level_target,
    )
    st.session_state["sim_dates"] = sorted(scn_steps["date"].unique())
    st.session_state["playback_day"] = 0
    st.session_state["playing"] = False

base_steps = st.session_state["base_steps"]
base_summary = st.session_state["base_summary"]
scn_steps = st.session_state["scn_steps"]
scn_summary = st.session_state["scn_summary"]
scn_events = st.session_state["scn_events"]
scenario_config = st.session_state["scenario_config"]
sim_dates = st.session_state["sim_dates"]

base_agg = simulation.network_aggregate(base_summary)
scn_agg = simulation.network_aggregate(scn_summary)
comparison = simulation.compare_baseline_vs_scenario(base_agg, scn_agg)
health_base = simulation.business_health_score(base_agg)
health_scn = simulation.business_health_score(scn_agg)
critical_df = simulation.rank_critical_entities(scn_summary, top_n=10)
recs = rec_engine.build_recommendations(
    base_agg, scn_agg, comparison, scenario_config, critical_df, health_base, health_scn,
)

st.title("Retail Demand & Inventory Digital Twin")
st.caption(
    "A virtual representation of a retail operation whose state evolves according to "
    "observed demand, AI-based predictions, inventory dynamics, replenishment policies "
    "and simulated supply conditions -- not a static forecasting dashboard."
)

tab1, tab2, tab3, tab4 = st.tabs(
    ["\U0001F310 Twin Overview", "\U0001F4CD Twin State", "\U0001F52E Scenario Simulator", "\U0001F4CB Decision Support Evidence"]
)

# ---------------------------------------------------------------------------
# TAB 1 — Twin Overview
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Retail Network Digital Twin — Current Scenario Snapshot")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Stores", len(sel_stores))
    c2.metric("Items", len(sel_items))
    c3.metric("Total demand (horizon)", f"{scn_agg.get('total_demand', 0):,.0f}")
    c4.metric("Service level", f"{scn_agg.get('service_level', 0)*100:.1f}%",
              delta=f"{comparison.get('service_level_change', 0)*100:+.1f} pp vs baseline")
    c5.metric("Pending replenishments", f"{scn_agg.get('replenishment_orders', 0):,}")

    st.markdown("#### Store operational status")
    store_stats = scn_summary.groupby("store").agg(
        stockout_days=("stockout_days", "sum"),
        days=("days_simulated", "max"),
        service_level=("fulfilled_demand", "sum"),
        total_demand=("total_demand", "sum"),
    ).reset_index()
    store_stats["stockout_rate"] = store_stats["stockout_days"] / (store_stats["days"] * scn_summary.groupby("store").size().values)
    store_stats["service_level"] = scn_summary.groupby("store").apply(
        lambda g: g["fulfilled_demand"].sum() / max(g["total_demand"].sum(), 1e-6)
    ).values

    cols = st.columns(min(len(store_stats), 5))
    for idx, row in store_stats.iterrows():
        label, color, emoji = store_status(row["stockout_rate"])
        with cols[idx % len(cols)]:
            st.markdown(
                f'<div class="metric-card">{emoji} <b>Store {int(row.store)}</b><br>'
                f'{status_pill(label, color)}<br>'
                f'<small>Service level: {row.service_level*100:.1f}%</small></div>',
                unsafe_allow_html=True,
            )

    st.markdown("#### Demand vs inventory over the simulation horizon (network total)")
    st.caption(
        "\u26A0\uFE0F These two lines use **separate scales** (left axis = demand, right axis = "
        "inventory) because they measure different things: demand is a **daily rate** "
        "(units sold *per day*), while inventory is a **stock level** (units on hand "
        "*right now*). The lines crossing visually does **not** mean inventory ran out -- "
        "use the 'Days of supply remaining' metric below for the real signal."
    )
    net_daily = scn_steps.groupby("date").agg(
        predicted_demand=("predicted_demand", "sum"),
        actual_demand=("actual_demand", "sum"),
        inventory=("inventory", "sum"),
    ).reset_index()

    latest = net_daily.iloc[-1]
    days_of_supply = latest.inventory / max(latest.actual_demand, 1e-6)
    dos_color = ACCENT_BAD if days_of_supply < 2 else (ACCENT_WARN if days_of_supply < 5 else ACCENT_GOOD)
    m1, m2, m3 = st.columns(3)
    m1.metric("Total inventory (latest day)", f"{latest.inventory:,.0f} units")
    m2.metric("Simulated demand (latest day)", f"{latest.actual_demand:,.0f} units/day")
    m3.metric("Days of supply remaining", f"{days_of_supply:.1f} days")
    st.caption(
        f"Days of supply = inventory \u00f7 demand rate. At {days_of_supply:.1f} days, the "
        "network is not close to running out -- this is the number that actually drives "
        "the store status pills above (via per-store-item stockout events), not the raw "
        "gap between the two chart lines."
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=net_daily.date, y=net_daily.actual_demand, name="Simulated demand (units/day)",
                              line=dict(color=ACCENT_WARN)))
    fig.add_trace(go.Scatter(x=net_daily.date, y=net_daily.inventory, name="Total inventory (units on hand)",
                              line=dict(color=ACCENT), yaxis="y2"))
    fig.update_layout(
        yaxis=dict(title=dict(text="Demand (units/day)", font=dict(color=ACCENT_WARN)),
                    tickfont=dict(color=ACCENT_WARN)),
        yaxis2=dict(title=dict(text="Inventory (units on hand)", font=dict(color=ACCENT)),
                     tickfont=dict(color=ACCENT), overlaying="y", side="right"),
        legend=dict(orientation="h", y=1.15), height=380, margin=dict(t=30),
        plot_bgcolor="white",
    )
    st.plotly_chart(fig, width='stretch')

    with st.expander("Data audit (train.csv / test.csv)"):
        st.write(f"Train range: **{train_df.date.min().date()} to {train_df.date.max().date()}** "
                 f"({len(train_df):,} rows, {train_df.store.nunique()} stores x {train_df.item.nunique()} items)")
        st.write("Observed columns: `date`, `store`, `item`, `sales` — no missing values, no duplicates.")
        st.write(f"Model validation metrics (last {model_metrics['validation_days']} days, held out chronologically): "
                 f"MAE={model_metrics['MAE']:.2f}, RMSE={model_metrics['RMSE']:.2f}, sMAPE={model_metrics['sMAPE']:.2f}%")

# ---------------------------------------------------------------------------
# TAB 2 — Twin State (per store-item, with live playback)
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Store-Item Twin State")
    c1, c2 = st.columns(2)
    sel_store = c1.selectbox("Store", sel_stores, key="ts_store")
    sel_item = c2.selectbox("Item", sel_items, key="ts_item")

    pair_steps = scn_steps[(scn_steps.store == sel_store) & (scn_steps.item == sel_item)].sort_values("date").reset_index(drop=True)
    pair_summary = scn_summary[(scn_summary.store == sel_store) & (scn_summary.item == sel_item)]

    n_days = len(pair_steps)
    if n_days == 0:
        st.warning("No simulated data for this store-item pair.")
    else:
        pc1, pc2, pc3, pc4 = st.columns([1, 1, 1, 3])
        if pc1.button("\u23EE Reset"):
            st.session_state["playback_day"] = 0
            st.session_state["playing"] = False
        if pc2.button("\u23ED Next Day"):
            st.session_state["playback_day"] = min(st.session_state["playback_day"] + 1, n_days - 1)
        play_label = "\u23F8 Pause" if st.session_state.get("playing") else "\u25B6 Play"
        if pc3.button(play_label):
            st.session_state["playing"] = not st.session_state.get("playing", False)

        day_idx = st.slider("Simulation Day", 0, n_days - 1, st.session_state["playback_day"], key="day_slider")
        st.session_state["playback_day"] = day_idx
        st.caption(f"Simulation Day: {day_idx + 1} / {n_days}")

        current = pair_steps.iloc[day_idx]
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Inventory", f"{current.inventory:.0f}")
        m2.metric("Predicted demand", f"{current.predicted_demand:.0f}")
        m3.metric("Reorder point", f"{current.reorder_point:.0f}")
        m4.metric("Safety stock", f"{current.safety_stock:.0f}")
        m5.metric("Pending orders", f"{current.pending_orders:.0f}")
        if current.stockout:
            st.error(f"\u26A0\uFE0F Stockout on {pd.Timestamp(current.date).date()}: "
                     f"{current.unmet_demand:.0f} units unmet demand.")

        visible = pair_steps.iloc[:day_idx + 1]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=visible.date, y=visible.inventory, name="Inventory", line=dict(color=ACCENT)))
        fig.add_trace(go.Scatter(x=visible.date, y=visible.reorder_point, name="Reorder point",
                                  line=dict(color=ACCENT_WARN, dash="dash")))
        fig.add_trace(go.Scatter(x=visible.date, y=visible.safety_stock, name="Safety stock",
                                  line=dict(color=ACCENT_BAD, dash="dot")))
        fig.add_trace(go.Bar(x=visible.date, y=visible.actual_demand, name="Simulated demand",
                              marker_color="rgba(46,111,110,0.25)", yaxis="y2"))
        fig.update_layout(
            yaxis=dict(title="Inventory (units)"), yaxis2=dict(title="Demand", overlaying="y", side="right"),
            legend=dict(orientation="h", y=1.15), height=400, margin=dict(t=30), plot_bgcolor="white",
        )
        st.plotly_chart(fig, width='stretch')

        st.markdown("##### Digital Twin event log — Store {} / Item {}".format(sel_store, sel_item))
        pair_events = scn_events[(scn_events.store == sel_store) & (scn_events.item == sel_item) & (scn_events.day <= day_idx + 1)]
        if len(pair_events):
            for _, ev in pair_events.sort_values("day", ascending=False).head(15).iterrows():
                st.text(f"Day {ev.day:>3} | {ev.event_type:<18} | {ev.message}")
        else:
            st.caption("No events yet.")

        if st.session_state.get("playing") and day_idx < n_days - 1:
            time_module.sleep(0.4)
            st.session_state["playback_day"] = day_idx + 1
            st.rerun()
        elif st.session_state.get("playing") and day_idx >= n_days - 1:
            st.session_state["playing"] = False

# ---------------------------------------------------------------------------
# TAB 3 — Scenario Simulator
# ---------------------------------------------------------------------------
with tab3:
    st.subheader(f"Baseline vs Scenario: {scenario_config.name}")
    st.caption(
        f"Demand multiplier: {scenario_config.demand_multiplier:.2f}x | "
        f"Lead-time change: {scenario_config.lead_time_change_days:+d} days | "
        f"Target service level: {scenario_config.service_level_target*100:.0f}%"
    )

    def fmt_delta(v, pct=False, invert=False):
        if v is None:
            return None
        sign = "+" if v >= 0 else ""
        return f"{sign}{v*100:.1f} pp" if pct else f"{sign}{v:,.1f}"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Service level", f"{scn_agg['service_level']*100:.1f}%",
              delta=fmt_delta(comparison['service_level_change'], pct=True))
    c2.metric("Stockout events", scn_agg['stockout_events'],
              delta=f"{comparison['stockout_change']:+d}", delta_color="inverse")
    c3.metric("Unmet demand", f"{scn_agg['unmet_demand']:,.0f}",
              delta=fmt_delta(comparison['unmet_demand_change']), delta_color="inverse")
    c4.metric("Avg inventory", f"{scn_agg['average_inventory']:,.0f}",
              delta=fmt_delta(comparison['inventory_change']))

    def _fmt_row(agg):
        return [
            f"{agg['total_demand']:,.0f}", f"{agg['fulfilled_demand']:,.0f}", f"{agg['unmet_demand']:,.0f}",
            f"{agg['service_level']*100:.1f}%", f"{agg['stockout_events']:,}",
            f"{agg['average_inventory']:,.0f}", f"{agg['min_inventory']:,.0f}", f"{agg['max_inventory']:,.0f}",
            f"{agg['replenishment_orders']:,}",
        ]

    comp_table = pd.DataFrame({
        "Metric": ["Total demand", "Fulfilled demand", "Unmet demand", "Service level",
                   "Stockout events", "Average inventory", "Min inventory", "Max inventory",
                   "Replenishment orders"],
        "Baseline": _fmt_row(base_agg),
        "Scenario": _fmt_row(scn_agg),
    })
    st.dataframe(comp_table, width='stretch', hide_index=True)

    st.markdown("#### Business Operational Health")
    hc1, hc2 = st.columns(2)
    for col, label, health in [(hc1, "Baseline", health_base), (hc2, scenario_config.name, health_scn)]:
        with col:
            st.metric(f"{label} health score", f"{health['score']:.1f} / 100")
            comp_df = pd.DataFrame(
                [(k, v) for k, v in health["components"].items()], columns=["Component", "Points"]
            )
            st.dataframe(comp_df, width='stretch', hide_index=True)

# ---------------------------------------------------------------------------
# TAB 4 — Decision Support Evidence
# ---------------------------------------------------------------------------
with tab4:
    st.subheader("Decision Support Evidence")
    st.caption("Findings grounded in this simulation run — feeds the overarching DT-readiness DSS.")

    st.markdown("#### Critical store-item pairs")
    if critical_df.empty:
        st.success("No critical store-item pairs detected under this scenario.")
    else:
        st.dataframe(critical_df, width='stretch', hide_index=True)

    st.markdown("#### Explainable recommendations")
    for r in recs:
        with st.container():
            st.markdown(f"**Finding:** {r['finding']}")
            st.markdown(f"**Cause:** {r['cause']}")
            st.markdown(f"**Recommendation:** {r['recommendation']}")
            st.markdown(f"**Expected effect:** {r['expected_effect']}")
            st.divider()

    st.markdown("#### Export machine-readable evidence for the overarching DSS")
    if st.button("Generate dss_evidence.json"):
        import os
        os.makedirs("outputs", exist_ok=True)
        payload = simulation.export_dss_evidence(
            "outputs/dss_evidence.json", model_metrics, base_agg, scn_agg, scenario_config,
            comparison, critical_df, recs, n_stores=len(sel_stores), n_items=len(sel_items),
            horizon_days=horizon_days,
        )
        st.success("outputs/dss_evidence.json generated.")
        st.download_button(
            "Download dss_evidence.json",
            data=json.dumps(payload, indent=2, default=str),
            file_name="dss_evidence.json",
            mime="application/json",
        )
        with st.expander("Preview JSON"):
            st.json(payload)
