"""
Delhivery Network Intelligence -- operations console (Task 5 deliverable).

A decision dashboard for the Head of Network Operations, built on the
artifacts produced by the pipeline scripts:

  build_graph.py            -> graph, node metrics, corridor audit
  task3_eta_model.py        -> outputs/model_comparison.csv (8 ETA models)
  task4_route_choice.py     -> route-choice bundle + scored dispatches
  task5_hub_impact.py       -> outputs/hub_impact.csv
  visualize_graph.py        -> outputs/visualizations/*

Run:  streamlit run app.py
"""

import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from task4_route_choice import (
    BUNDLE_PATH,
    DEFAULT_PARAMS,
    FEATURES_CSV,
    apply_decision,
    recommend,
)

ROOT = Path(__file__).parent
OUT = ROOT / "outputs"
VIZ = OUT / "visualizations"

# ---------------------------------------------------------------------------
# Page + design system
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Delhivery Network Intelligence",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="collapsed",
)

INK = "#e8ecf4"
MUTED = "#8b94a8"
RED = "#e63946"       # FTL / alerts
TEAL = "#2a9d8f"      # Carting / good
AMBER = "#e9c46a"
BLUE = "#4d7cfe"
CARD = "#121a2b"
EDGE = "#22304d"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

#MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; height: 0; }
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1380px; }

/* ---------- masthead ---------- */
.nic-mast {
  display: flex; align-items: baseline; gap: 14px; flex-wrap: wrap;
  border-bottom: 1px solid #22304d; padding-bottom: 14px; margin-bottom: 4px;
}
.nic-logo {
  font-family: 'Space Grotesk'; font-weight: 700; font-size: 1.55rem;
  letter-spacing: .02em; color: #e8ecf4;
}
.nic-logo b { color: #e63946; }
.nic-sub {
  font-size: .78rem; color: #8b94a8; letter-spacing: .14em;
  text-transform: uppercase; font-weight: 500;
}
.nic-chip {
  margin-left: auto; font-family: 'JetBrains Mono'; font-size: .7rem;
  color: #2a9d8f; border: 1px solid #1d3b38; border-radius: 3px;
  background: #0f1d22; padding: 3px 9px; letter-spacing: .08em;
}

/* ---------- tabs as product nav ---------- */
[data-testid="stTabs"] button[data-baseweb="tab"] {
  background: transparent; border-radius: 0; padding: 10px 4px; margin-right: 26px;
}
[data-testid="stTabs"] button[data-baseweb="tab"] p {
  font-family: 'Space Grotesk'; font-size: .86rem; font-weight: 600;
  letter-spacing: .12em; text-transform: uppercase; color: #8b94a8;
}
[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] p { color: #e8ecf4; }
[data-testid="stTabs"] [data-baseweb="tab-highlight"] { background: #e63946; height: 2px; }
[data-testid="stTabs"] [data-baseweb="tab-border"] { background: #22304d; }

/* ---------- cards / KPIs ---------- */
.kpi-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(168px, 1fr)); gap: 12px; }
.kpi {
  background: #121a2b; border: 1px solid #22304d; border-radius: 8px;
  padding: 14px 16px 12px; position: relative; overflow: hidden;
}
.kpi::before {
  content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 3px;
  background: var(--kpi-accent, #4d7cfe);
}
.kpi .lab {
  font-size: .66rem; letter-spacing: .14em; text-transform: uppercase;
  color: #8b94a8; font-weight: 600; margin-bottom: 6px;
}
.kpi .val { font-family: 'JetBrains Mono'; font-size: 1.45rem; font-weight: 600; color: #e8ecf4; line-height: 1.1; }
.kpi .sub { font-size: .72rem; color: #8b94a8; margin-top: 5px; }

.nic-card {
  background: #121a2b; border: 1px solid #22304d; border-radius: 8px;
  padding: 18px 20px; margin: 4px 0 14px;
}
.nic-h {
  font-family: 'Space Grotesk'; font-size: .8rem; font-weight: 600;
  letter-spacing: .16em; text-transform: uppercase; color: #8b94a8;
  margin: 18px 0 10px;
}
.nic-h b { color: #e8ecf4; }

.verdict {
  border-radius: 8px; padding: 18px 22px; margin-top: 6px;
  border: 1px solid; background: #121a2b;
}
.verdict .vtitle { font-family: 'Space Grotesk'; font-size: 1.3rem; font-weight: 700; }
.verdict .vline { font-size: .85rem; color: #aab3c5; margin-top: 6px; }
.verdict .vnum { font-family: 'JetBrains Mono'; color: #e8ecf4; }

.memo {
  background: #f4f1ea; color: #1d2430; border-radius: 6px;
  padding: 34px 40px; font-family: Georgia, 'Times New Roman', serif;
  line-height: 1.55; font-size: .94rem;
}
.memo h2 { font-family: 'Space Grotesk', sans-serif; letter-spacing: .04em; }
.memo table { width: 100%; border-collapse: collapse; font-size: .86rem; }
.memo th, .memo td { border-bottom: 1px solid #c9c2b2; text-align: left; padding: 6px 8px; }

.smallnote { font-size: .74rem; color: #67708a; }
hr { border-color: #22304d; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

PLOT_FONT = dict(family="Inter, sans-serif", color=MUTED, size=12)


def style_fig(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=PLOT_FONT,
        margin=dict(l=10, r=10, t=36, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hoverlabel=dict(font_family="JetBrains Mono"),
    )
    fig.update_xaxes(gridcolor=EDGE, zerolinecolor=EDGE)
    fig.update_yaxes(gridcolor=EDGE, zerolinecolor=EDGE)
    return fig


def kpis(items) -> None:
    """items: list of (label, value, sub, accent)."""
    cells = "".join(
        f'<div class="kpi" style="--kpi-accent:{a}"><div class="lab">{l}</div>'
        f'<div class="val">{v}</div><div class="sub">{s}</div></div>'
        for l, v, s, a in items
    )
    st.markdown(f'<div class="kpi-row">{cells}</div>', unsafe_allow_html=True)


def section(text: str) -> None:
    st.markdown(f'<div class="nic-h"><b>{text}</b></div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data access (all precomputed artifacts -- the 55 MB raw CSV is never read here)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_artifacts():
    data = {}
    for key, path in {
        "comparison": OUT / "model_comparison.csv",
        "node_metrics": OUT / "node_metrics.csv",
        "corridors": OUT / "corridor_audit.csv",
        "hub_impact": OUT / "hub_impact.csv",
        "scored": Path(FEATURES_CSV),
        "route_recs": OUT / "route_choice_recommendations.csv",
    }.items():
        data[key] = pd.read_csv(path) if path.exists() else None
    return data


@st.cache_resource(show_spinner=False)
def load_bundle():
    return joblib.load(BUNDLE_PATH) if os.path.exists(BUNDLE_PATH) else None


@st.cache_data(show_spinner=False)
def load_map_html() -> str | None:
    path = VIZ / "06_interactive_network.html"
    if not path.exists():
        return None
    html = path.read_text(encoding="utf-8")
    # pyvis references lib/bindings/utils.js relatively; iframes have no base
    # URL for it, so inline the file.
    utils = ROOT / "lib" / "bindings" / "utils.js"
    if utils.exists():
        html = re.sub(
            r'<script[^>]*src="lib/bindings/utils\.js"[^>]*>\s*</script>',
            "<script>" + utils.read_text(encoding="utf-8") + "</script>",
            html,
        )
    return html


def run_pipeline(cmd: list[str], label: str) -> None:
    with st.status(f"Running {label} ...", expanded=True) as status:
        st.markdown(
            f'<span class="smallnote">$ {" ".join(cmd)}</span>', unsafe_allow_html=True
        )
        proc = subprocess.run(
            cmd, cwd=ROOT, capture_output=True, text=True, timeout=1800
        )
        st.code((proc.stdout or "") + (proc.stderr or ""), language="text")
        if proc.returncode == 0:
            status.update(label=f"{label} finished", state="complete")
            load_artifacts.clear()
            load_bundle.clear()
        else:
            status.update(label=f"{label} failed (exit {proc.returncode})", state="error")


A = load_artifacts()
BUNDLE = load_bundle()

# ---------------------------------------------------------------------------
# Masthead
# ---------------------------------------------------------------------------

stamp = datetime.now().strftime("%d %b %Y · %H:%M")
st.markdown(
    f"""
    <div class="nic-mast">
      <span class="nic-logo">delhivery<b>·</b>nic</span>
      <span class="nic-sub">Network Intelligence Console</span>
      <span class="nic-chip">GRAPH LIVE · {stamp}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_cc, tab_map, tab_models, tab_route, tab_memo = st.tabs(
    ["Command Center", "Network Map", "Model Lab", "Route Advisor", "Strategy Memo"]
)

# ===========================================================================
# TAB 1 -- COMMAND CENTER
# ===========================================================================

with tab_cc:
    nm, corr, hub = A["node_metrics"], A["corridors"], A["hub_impact"]
    comp = A["comparison"]
    if nm is None or corr is None:
        st.warning("Run `python build_graph.py` first - graph artifacts are missing.")
    else:
        chronic = corr["is_chronically_delayed"].mean() * 100
        osrm_under = np.average(corr["median_factor"], weights=corr["total_trips"])
        breach_rate = (
            hub["breaches"].sum() / hub["legs_out"].sum() * 100 if hub is not None else np.nan
        )
        best_row = comp.loc[comp["mae"].idxmin()] if comp is not None else None
        base_row = comp[comp["model"] == "baseline"].iloc[0] if comp is not None else None
        mae_gain = (
            (base_row["mae"] - best_row["mae"]) / base_row["mae"] * 100
            if comp is not None else np.nan
        )

        kpis([
            ("Facilities", f"{len(nm):,}", "nodes in the directed graph", BLUE),
            ("Corridors", f"{len(corr):,}", "directed lane pairs", BLUE),
            ("Chronic corridors", f"{chronic:.0f}%", "actual > OSRM by 20%+", RED),
            ("OSRM drift", f"{osrm_under:.2f}×", "trip-weighted actual/OSRM", AMBER),
            ("SLA breach rate", f"{breach_rate:.1f}%", "vs calibrated promise +20%", RED),
            ("Graph model lift", f"−{mae_gain:.0f}% MAE", f"{best_row['model']} vs baseline" if best_row is not None else "", TEAL),
        ])

        c1, c2 = st.columns([3, 2], gap="medium")
        with c1:
            section("Where the network bleeds — SLA breaches by hub")
            if hub is not None:
                top = hub[hub["rankable"]].nlargest(12, "breaches").iloc[::-1]
                fig = go.Figure(go.Bar(
                    x=top["breaches"], y=top["name"], orientation="h",
                    marker_color=RED, marker_line_width=0, opacity=0.85,
                    customdata=np.stack([top["breach_rate"] * 100,
                                         top["network_breach_share_pct"]], axis=-1),
                    hovertemplate="%{y}<br>%{x} breaches · rate %{customdata[0]:.0f}%"
                                  "<br>%{customdata[1]:.1f}% of network<extra></extra>",
                ))
                st.plotly_chart(style_fig(fig, 420), width="stretch")
            else:
                st.info("Run `python task5_hub_impact.py` to populate hub impact.")
        with c2:
            section("Corridor delay-factor distribution")
            sample = corr[corr["total_trips"] >= 3]
            fig = px.histogram(sample, x="median_factor", nbins=60,
                               color_discrete_sequence=[BLUE])
            fig.add_vline(x=1.2, line_dash="dash", line_color=AMBER,
                          annotation_text="SLA threshold 1.2×",
                          annotation_font_color=AMBER)
            fig.update_layout(xaxis_title="median actual ÷ OSRM (per corridor)",
                              yaxis_title="corridors", showlegend=False)
            st.plotly_chart(style_fig(fig, 420), width="stretch")

        section("Structural chokepoints — betweenness vs incoming delay")
        nm_top = nm[nm["in_degree"] + nm["out_degree"] >= 5].copy()
        fig = px.scatter(
            nm_top, x="betweenness", y="avg_incoming_delay_factor",
            size="in_degree", color="bottleneck_score",
            color_continuous_scale=["#22304d", "#4d7cfe", "#e9c46a", "#e63946"],
            hover_name="name", size_max=26,
        )
        fig.update_layout(
            xaxis_title="betweenness centrality (how much traffic must pass through)",
            yaxis_title="avg incoming delay factor",
            coloraxis_colorbar=dict(title="bottleneck"),
        )
        st.plotly_chart(style_fig(fig, 430), width="stretch")
        st.markdown(
            '<p class="smallnote">Top-right = structurally critical AND chronically slow: '
            "the upgrade shortlist. Bubble size = inbound corridor count. "
            "All metrics computed on training trips only (no test leakage).</p>",
            unsafe_allow_html=True,
        )

# ===========================================================================
# TAB 2 -- NETWORK MAP
# ===========================================================================

with tab_map:
    section("Interactive logistics graph")
    st.markdown(
        '<p class="smallnote">Drag to pan · scroll to zoom · hover a node for hub stats. '
        "Node colour/size encode bottleneck severity; edges are corridors weighted by "
        "median delay factor.</p>",
        unsafe_allow_html=True,
    )
    map_html = load_map_html()
    if map_html is None:
        st.warning("Map not found - run `python visualize_graph.py` to generate "
                   "outputs/visualizations/06_interactive_network.html.")
    else:
        components.html(map_html, height=760, scrolling=False)

    section("Static atlas")
    gallery = {
        "01 · Full network overview": VIZ / "01_full_network.png",
        "02 · Bottleneck hubs": VIZ / "02_bottleneck_hubs.png",
        "03 · Chronically delayed corridors": VIZ / "03_delayed_corridors.png",
        "04 · Degree distributions": VIZ / "04_degree_distributions.png",
        "05 · Top hubs subgraph": VIZ / "05_top_hubs_subgraph.png",
        "07 · FTL vs Carting trade-off": VIZ / "07_route_choice_tradeoff.png",
        "08 · Route choice by distance": VIZ / "08_route_choice_by_distance.png",
    }
    avail = {k: v for k, v in gallery.items() if v.exists()}
    if avail:
        pick = st.selectbox("Select a view", list(avail.keys()), label_visibility="collapsed")
        st.image(str(avail[pick]), width="stretch")
    else:
        st.info("No static visualizations found - run `python visualize_graph.py`.")

# ===========================================================================
# TAB 3 -- MODEL LAB
# ===========================================================================

with tab_models:
    comp = A["comparison"]
    if comp is None:
        st.warning("Run `python task3_eta_model.py --retrain` first.")
    else:
        comp = comp.copy()
        base = comp[comp["model"] == "baseline"].iloc[0]

        section("Pick what matters, the lab picks the model")
        cw1, cw2 = st.columns([3, 2], gap="large")
        with cw1:
            w_acc = st.slider(
                "Decision priority — minute-error (MAE) vs SLA hit-rate (within 15%)",
                0, 100, 50,
                help="0 = only MAE matters · 100 = only the % of trips predicted "
                     "within 15% of actual (the business SLA metric) matters.",
            )
        with cw2:
            speed_gate = st.toggle(
                "Require sub-10 ms inference", value=False,
                help="Filters out models too slow to score every dispatch in real time.",
            )

        pool = comp.copy()
        if speed_gate:
            fast = pool["inference_time_sec"] < 0.010
            pool = pool[fast] if fast.any() else pool
        mae_score = (pool["mae"].max() - pool["mae"]) / (pool["mae"].max() - pool["mae"].min())
        w15_score = (pool["within_15_pct"] - pool["within_15_pct"].min()) / (
            pool["within_15_pct"].max() - pool["within_15_pct"].min())
        pool["score"] = (1 - w_acc / 100) * mae_score + (w_acc / 100) * w15_score
        champ = pool.loc[pool["score"].idxmax()]

        mae_gain = (base["mae"] - champ["mae"]) / base["mae"] * 100
        w15_gain = champ["within_15_pct"] - base["within_15_pct"]
        st.markdown(
            f"""
            <div class="verdict" style="border-color:{TEAL}">
              <div class="vtitle" style="color:{TEAL}">Recommended: {champ['model']}</div>
              <div class="vline">MAE <span class="vnum">{champ['mae']:.1f} min</span>
              (<span class="vnum">−{mae_gain:.1f}%</span> vs baseline) ·
              within-15% <span class="vnum">{champ['within_15_pct']:.1f}%</span>
              (<span class="vnum">+{w15_gain:.1f} pts</span>) ·
              {int(champ['n_features'])} features ·
              inference <span class="vnum">{champ['inference_time_sec']*1000:.1f} ms</span>
              for the full test set</div>
              <div class="vline">The graph advantage is measured, not claimed: every model
              is scored on the same held-out trip set with identical metrics
              (outputs/model_comparison.csv is the single source of truth).</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        c1, c2 = st.columns(2, gap="medium")
        order = comp.sort_values("mae", ascending=False)
        with c1:
            section("Mean absolute error (minutes, lower is better)")
            colors = [TEAL if m == champ["model"] else ("#54607a" if m == "baseline" else BLUE)
                      for m in order["model"]]
            fig = go.Figure(go.Bar(
                x=order["mae"], y=order["model"], orientation="h",
                marker_color=colors, marker_line_width=0,
                hovertemplate="%{y}: %{x:.1f} min<extra></extra>",
            ))
            fig.add_vline(x=float(base["mae"]), line_dash="dot", line_color=MUTED)
            st.plotly_chart(style_fig(fig, 380), width="stretch")
        with c2:
            section("Business metric — % of trips within 15% of actual")
            order2 = comp.sort_values("within_15_pct")
            colors2 = [TEAL if m == champ["model"] else ("#54607a" if m == "baseline" else BLUE)
                       for m in order2["model"]]
            fig = go.Figure(go.Bar(
                x=order2["within_15_pct"], y=order2["model"], orientation="h",
                marker_color=colors2, marker_line_width=0,
                hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
            ))
            fig.add_vline(x=float(base["within_15_pct"]), line_dash="dot", line_color=MUTED)
            st.plotly_chart(style_fig(fig, 380), width="stretch")

        section("Error vs accuracy frontier")
        fig = px.scatter(
            comp, x="mae", y="within_15_pct", text="model",
            size="n_features", size_max=30,
            color=comp["model"].eq(champ["model"]).map({True: "champion", False: "field"}),
            color_discrete_map={"champion": TEAL, "field": BLUE},
        )
        fig.update_traces(textposition="top center",
                          textfont=dict(size=11, color=INK, family="JetBrains Mono"))
        fig.update_layout(xaxis_title="MAE (min) — lower is better",
                          yaxis_title="% within 15% — higher is better",
                          showlegend=False)
        st.plotly_chart(style_fig(fig, 420), width="stretch")

        section("Full benchmark table")
        show = comp[["model", "n_features", "mae", "within_15_pct",
                     "training_time_sec", "inference_time_sec"]].copy()
        show.columns = ["model", "features", "MAE (min)", "within 15% (%)",
                        "train (s)", "infer (s)"]
        st.dataframe(
            show.style.format({"MAE (min)": "{:.2f}", "within 15% (%)": "{:.2f}",
                               "train (s)": "{:.3f}", "infer (s)": "{:.4f}"}),
            width="stretch", hide_index=True,
        )
        st.markdown(
            '<p class="smallnote">baseline = trip features only · graph_enhanced = '
            "+ hub metrics + GraphSAGE embeddings · graphsage_* = + GraphSAGE embeddings "
            "(XGB/LGBM/MLP/residual heads) · node2vec_* = + node2vec random-walk embeddings. "
            "All graph signal is computed from training trips only.</p>",
            unsafe_allow_html=True,
        )

# ===========================================================================
# TAB 4 -- ROUTE ADVISOR (Task 4 framework, user-tunable economics)
# ===========================================================================

with tab_route:
    scored_csv = A["scored"]
    if BUNDLE is None or scored_csv is None:
        st.warning("Run `python task4_route_choice.py --retrain` first - the route-choice "
                    "bundle / scored dispatches are missing.")
    else:
        section("Decision economics — tune what a minute is worth")
        st.markdown(
            '<p class="smallnote">Counterfactual times are model outputs and never change '
            "here; the sliders only re-price the decision rule, so every dispatch below "
            "re-decides instantly. Cost rates are explicit assumptions — plug in real "
            "contract rates when you have them.</p>",
            unsafe_allow_html=True,
        )

        pc1, pc2, pc3 = st.columns(3, gap="large")
        with pc1:
            vot = st.slider("Value of time (₹ per minute saved)", 0.0, 100.0,
                            float(DEFAULT_PARAMS["value_of_time"]), 1.0)
            risk_weight = st.slider("Tail-risk weight (p90 minutes)", 0.0, 2.0,
                                    float(DEFAULT_PARAMS["risk_weight"]), 0.05,
                                    help="How much a p90 (worst-case) minute counts "
                                         "relative to an expected minute.")
        with pc2:
            risk_gain = st.slider("Bottleneck-source uplift", 0.0, 1.5,
                                  float(DEFAULT_PARAMS["risk_gain"]), 0.05,
                                  help="Extra value placed on FTL reliability when the "
                                       "source hub is a structural bottleneck.")
            ftl_fixed = st.number_input("FTL fixed cost (₹/dispatch)", 0.0, 20000.0,
                                        float(DEFAULT_PARAMS["ftl_fixed"]), 100.0)
        with pc3:
            ftl_km = st.number_input("FTL rate (₹/km)", 0.0, 200.0,
                                     float(DEFAULT_PARAMS["ftl_per_km"]), 1.0)
            cart_km = st.number_input("Carting rate (₹/km)", 0.0, 200.0,
                                      float(DEFAULT_PARAMS["cart_per_km"]), 1.0)

        params = dict(value_of_time=vot, risk_weight=risk_weight, risk_gain=risk_gain,
                      ftl_fixed=ftl_fixed, ftl_per_km=ftl_km, cart_per_km=cart_km)
        live = apply_decision(scored_csv, BUNDLE, **params)

        n = len(live)
        rec_ftl = (live["recommended_route"] == "FTL").mean() * 100
        sw = live[live["switch"]]
        trusted = sw[~sw["extrapolated"]]
        to_ftl = trusted[trusted["recommended_route"] == "FTL"]
        to_cart = trusted[trusted["recommended_route"] == "Carting"]
        net_cost = trusted["switch_extra_cost"].sum()
        minutes = trusted["switch_minutes_saved"].sum()

        kpis([
            ("Dispatches scored", f"{n:,}", "held-out test legs", BLUE),
            ("FTL recommended", f"{rec_ftl:.0f}%", "of all dispatches", RED),
            ("Trusted switches", f"{len(trusted):,}",
             f"{len(sw):,} total · extrapolations excluded", AMBER),
            ("→ FTL upgrades", f"{len(to_ftl):,}",
             f"avg +{to_ftl['switch_minutes_saved'].mean():.0f} min saved" if len(to_ftl) else "—", RED),
            ("→ Carting downgrades", f"{len(to_cart):,}",
             f"avg ₹{-to_cart['switch_extra_cost'].mean():,.0f} saved" if len(to_cart) else "—", TEAL),
            ("Net effect", f"₹{-net_cost:,.0f}",
             f"cost saved · {minutes:,.0f} min saved", TEAL if net_cost <= 0 else RED),
        ])

        c1, c2 = st.columns([3, 2], gap="medium")
        with c1:
            section("Time-cost trade-off per dispatch")
            plot_df = live.sample(min(len(live), 4000), random_state=7)
            fig = px.scatter(
                plot_df, x="delta_time_min", y="delta_cost",
                color="recommended_route",
                color_discrete_map={"FTL": RED, "Carting": TEAL},
                opacity=0.45,
                hover_data={"delta_time_min": ":.0f", "delta_cost": ":.0f",
                            "break_even_vot": ":.1f", "recommended_route": False},
            )
            xs = np.linspace(plot_df["delta_time_min"].min(),
                             plot_df["delta_time_min"].max(), 10)
            fig.add_trace(go.Scatter(
                x=xs, y=vot * xs, mode="lines",
                line=dict(color=INK, dash="dash", width=1.5),
                name=f"break-even @ ₹{vot:.0f}/min",
            ))
            fig.update_traces(marker_size=5, selector=dict(mode="markers"))
            fig.update_layout(xaxis_title="minutes FTL saves vs Carting (predicted)",
                              yaxis_title="extra cost of FTL (₹)")
            st.plotly_chart(style_fig(fig, 430), width="stretch")
        with c2:
            section("Recommendation by distance band")
            bands = pd.cut(live["osrm_distance"],
                           bins=[0, 25, 50, 100, 250, 500, np.inf],
                           labels=["0–25", "25–50", "50–100", "100–250", "250–500", "500+"])
            share = (live.groupby(bands, observed=True)["recommended_route"]
                     .value_counts(normalize=True).unstack().fillna(0) * 100)
            fig = go.Figure()
            for rt, color in [("Carting", TEAL), ("FTL", RED)]:
                if rt in share:
                    fig.add_trace(go.Bar(name=rt, x=share.index.astype(str),
                                         y=share[rt], marker_color=color,
                                         marker_line_width=0))
            fig.update_layout(barmode="stack", xaxis_title="OSRM distance (km)",
                              yaxis_title="% of dispatches")
            st.plotly_chart(style_fig(fig, 430), width="stretch")

        section("Dispatch simulator — score a single shipment")
        nm = A["node_metrics"]
        with st.form("simulator"):
            f1, f2, f3 = st.columns([2, 2, 1.4], gap="medium")
            hubs = nm.sort_values("out_degree", ascending=False)
            label = (hubs["name"].fillna(hubs["center"])).tolist()
            with f1:
                src_i = st.selectbox("Source facility", range(len(hubs)),
                                     format_func=lambda i: label[i])
                dst_i = st.selectbox("Destination facility", range(len(hubs)), index=1,
                                     format_func=lambda i: label[i])
            with f2:
                dist = st.number_input("OSRM distance (km)", 1.0, 3000.0, 120.0, 5.0)
                osrm_min = st.number_input("OSRM time (minutes)", 1.0, 5000.0, 95.0, 5.0)
            with f3:
                hour = st.slider("Dispatch hour", 0, 23, 21)
                dow = st.selectbox("Day", ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
                current = st.radio("Current plan", ["Carting", "FTL"], horizontal=True)
            go_btn = st.form_submit_button("Score dispatch", type="primary")

        if go_btn:
            s_row, d_row = hubs.iloc[src_i], hubs.iloc[dst_i]
            one = pd.DataFrame([{
                "osrm_time": osrm_min, "osrm_distance": dist,
                "hour_of_day": hour,
                "day_of_week": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"].index(dow),
                "src_betweenness": s_row["betweenness"],
                "src_bottleneck_score": s_row["bottleneck_score"],
                "src_avg_delay": s_row["avg_incoming_delay_factor"],
                "src_in_degree": s_row["in_degree"],
                "src_out_degree": s_row["out_degree"],
                "dst_bottleneck_score": d_row["bottleneck_score"],
                "route_type": current,
            }])
            r = recommend(one, BUNDLE, **params).iloc[0]
            color = RED if r["recommended_route"] == "FTL" else TEAL
            warn = (" · <span style='color:#e9c46a'>outside observed distance support — "
                    "treat as extrapolation</span>" if r["extrapolated"] else "")
            st.markdown(
                f"""
                <div class="verdict" style="border-color:{color}">
                  <div class="vtitle" style="color:{color}">Dispatch as {r['recommended_route']}</div>
                  <div class="vline">Expected time — FTL <span class="vnum">{r['pred_time_ftl']:.0f} min</span>
                    (p90 <span class="vnum">{r['pred_p90_ftl']:.0f}</span>) ·
                    Carting <span class="vnum">{r['pred_time_cart']:.0f} min</span>
                    (p90 <span class="vnum">{r['pred_p90_cart']:.0f}</span>)</div>
                  <div class="vline">Cost — FTL <span class="vnum">₹{r['cost_ftl']:,.0f}</span> ·
                    Carting <span class="vnum">₹{r['cost_cart']:,.0f}</span> ·
                    break-even value-of-time
                    <span class="vnum">₹{min(r['break_even_vot'], 9999):,.1f}/min</span>{warn}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        section("Pipeline runner")
        with st.expander("Re-run the Task-4 framework on the full dataset"):
            st.markdown(
                '<p class="smallnote">Re-scores every test dispatch with the saved models '
                "(~1 min) or fully retrains the four counterfactual models (~2–3 min). "
                "Dashboards refresh automatically when it finishes.</p>",
                unsafe_allow_html=True,
            )
            rc1, rc2, rc3 = st.columns([1.2, 1.2, 3])
            retrain = rc1.toggle("Retrain models", value=False)
            if rc2.button("Run Task 4", type="primary"):
                cmd = [sys.executable, "task4_route_choice.py"] + (["--retrain"] if retrain else [])
                run_pipeline(cmd, "task4_route_choice")
            if rc3.button("Refresh hub impact (Task 5 data)"):
                run_pipeline([sys.executable, "task5_hub_impact.py"], "task5_hub_impact")

# ===========================================================================
# TAB 5 -- STRATEGY MEMO (Task 5)
# ===========================================================================

with tab_memo:
    hub, corr, comp = A["hub_impact"], A["corridors"], A["comparison"]
    scored_csv = A["scored"]
    if hub is None:
        st.warning("Run `python task5_hub_impact.py` first.")
    else:
        section("Memo assumptions — adjust, the memo rewrites itself")
        a1, a2, a3 = st.columns(3, gap="large")
        with a1:
            rev_per_breach = st.number_input(
                "Cost per SLA breach (₹)", 50.0, 10000.0, 500.0, 50.0,
                help="Penalty + refund + customer-experience cost of one late dispatch.")
        with a2:
            target = st.selectbox(
                "Upgrade benchmark", ["Network median hub", "Best-quartile hub"],
                help="After an upgrade, the hub is assumed to breach at this rate; "
                     "only the excess is counted as recoverable.")
        with a3:
            top_n = st.slider("Hubs in the table", 5, 15, 5)

        rk = hub[hub["rankable"]].copy()
        bench = rk["breach_rate"].median() if target == "Network median hub" \
            else rk["breach_rate"].quantile(0.25)
        rk["recoverable"] = ((rk["breach_rate"] - bench).clip(lower=0) * rk["legs_out"]).round(0)
        rk["rev_at_risk"] = rk["breaches"] * rev_per_breach
        rk["rev_recoverable"] = rk["recoverable"] * rev_per_breach

        total_breach = hub["breaches"].sum()
        rank_by = st.radio("Rank hubs by", ["SLA-breach contribution", "Recoverable if upgraded"],
                           horizontal=True, label_visibility="collapsed")
        key = "breaches" if rank_by == "SLA-breach contribution" else "recoverable"
        top = rk.nlargest(top_n, key)

        top3 = rk.nlargest(3, "recoverable")
        red3 = top3["recoverable"].sum()
        kpis([
            ("Network breaches", f"{total_breach:,.0f}", "late vs calibrated promise", RED),
            ("Revenue at risk", f"₹{total_breach * rev_per_breach / 1e5:,.1f} L",
             f"@ ₹{rev_per_breach:,.0f} per breach", RED),
            ("Top-3 upgrade win", f"−{red3 / total_breach * 100:.1f}%",
             f"{red3:,.0f} breaches avoided", TEAL),
            ("Recovered revenue", f"₹{red3 * rev_per_breach / 1e5:,.1f} L",
             "from the top-3 upgrades", TEAL),
        ])

        def intervention(r) -> str:
            if r["breach_rate"] > bench * 1.5 and r["betweenness"] > rk["betweenness"].quantile(0.9):
                return "Facility upgrade — add sortation & dock capacity at this chokepoint"
            if r["avg_incoming_delay_factor"] > 2.5:
                return "Parallel inbound routing — relieve congested feeder corridors"
            if r["breach_rate"] > bench * 1.2:
                return "Process audit + route-type shift on worst corridors"
            return "Volume-driven — protect with capacity planning, not rebuild"

        section(f"Top {top_n} hubs by {rank_by.lower()}")
        view = top[["name", "breaches", "network_breach_share_pct", "breach_rate",
                    "recoverable", "rev_recoverable", "betweenness"]].copy()
        view["breach_rate"] = view["breach_rate"] * 100
        view["intervention"] = top.apply(intervention, axis=1)
        view.columns = ["hub", "breaches", "% of network", "breach rate %",
                        "recoverable", "₹ recoverable", "betweenness", "recommended intervention"]
        st.dataframe(
            view.style.format({"% of network": "{:.1f}", "breach rate %": "{:.0f}",
                               "₹ recoverable": "₹{:,.0f}", "betweenness": "{:.3f}",
                               "breaches": "{:,.0f}", "recoverable": "{:,.0f}"}),
            width="stretch", hide_index=True,
        )

        # ------- corridor interventions -------
        section("Chronic corridors worth a dedicated fix")
        cc = corr[(corr["is_chronically_delayed"]) & (~corr["is_sparse"])].copy()
        cc["excess"] = cc["median_factor"] - 1.2
        cc["pain"] = cc["excess"] * cc["total_trips"]
        cct = cc.nlargest(8, "pain")[
            ["source_name", "destination_name", "median_factor", "total_trips", "pct_delayed"]
        ].copy()
        cct["pct_delayed"] = cct["pct_delayed"] * 100
        cct.columns = ["from", "to", "delay factor", "trips", "% trips delayed"]
        st.dataframe(
            cct.style.format({"delay factor": "{:.2f}", "% trips delayed": "{:.0f}"}),
            width="stretch", hide_index=True,
        )

        # ------- the memo itself -------
        best = comp.loc[comp["mae"].idxmin()] if comp is not None else None
        base = comp[comp["model"] == "baseline"].iloc[0] if comp is not None else None
        sw_line = ""
        if scored_csv is not None and BUNDLE is not None:
            dflt = apply_decision(scored_csv, BUNDLE)
            tsw = dflt[dflt["switch"] & ~dflt["extrapolated"]]
            ups = tsw[tsw["recommended_route"] == "FTL"]
            downs = tsw[tsw["recommended_route"] == "Carting"]
            sw_line = (
                f"Route-type review of held-out dispatches found "
                f"**{len(ups):,} Carting→FTL upgrades** (≈{ups['switch_minutes_saved'].mean():.0f} "
                f"min faster each for ≈₹{ups['switch_extra_cost'].mean():,.0f}) and "
                f"**{len(downs):,} FTL→Carting downgrades** that save "
                f"≈₹{-downs['switch_extra_cost'].mean():,.0f} each with no material time loss."
            )

        memo_rows = "\n".join(
            f"| {i + 1} | {r['name']} | {r['breaches']:,.0f} ({r['network_breach_share_pct']:.1f}%) "
            f"| {r['breach_rate'] * 100:.0f}% | {intervention(r)} |"
            for i, (_, r) in enumerate(rk.nlargest(5, key).iterrows())
        )
        memo_md = f"""## Network Operations Strategy Memo
**To:** Head of Network Operations · **From:** Data Science — Network Intelligence · **Date:** {datetime.now():%d %B %Y}

**The problem.** OSRM under-predicts door-to-door time on virtually every lane
(trip-weighted actual/OSRM ≈ {np.average(corr['median_factor'], weights=corr['total_trips']):.1f}×).
Against a calibrated promise (OSRM × route-type factor, +20% tolerance) the network still
breaches on **{total_breach / hub['legs_out'].sum() * 100:.0f}% of dispatches** — concentrated
in a small set of structural hubs, which makes this fixable.

**Better promises now.** The graph-based ETA model ({best['model'] if best is not None else 'n/a'})
cuts mean ETA error by **{(base['mae'] - best['mae']) / base['mae'] * 100:.0f}%** vs the current
trip-feature approach and lifts on-promise predictions (±15%) by
**{best['within_15_pct'] - base['within_15_pct']:.0f} points**. Deploying it re-prices risk on
every corridor without touching a single truck.

**Where to spend.** Top 5 bottleneck hubs by {rank_by.lower()}:

| # | Hub | Breaches (share) | Rate | Recommended intervention |
|---|-----|------------------|------|--------------------------|
{memo_rows}

**The payoff.** Upgrading the **top 3 hubs** to the {target.lower()} benchmark avoids
**{red3:,.0f} late dispatches ({red3 / total_breach * 100:.1f}% of all SLA breaches)** and
recovers **₹{red3 * rev_per_breach:,.0f}** per period at ₹{rev_per_breach:,.0f}/breach.
{sw_line}

**Caveats.** Costs per breach and freight rates are explicit assumptions — swap in contract
numbers in the console and every figure above re-derives. Counterfactual route-type claims
exclude out-of-support extrapolations.
"""
        section("The memo (live preview)")
        st.markdown(memo_md)
        st.download_button(
            "Download memo (.md)", memo_md.encode("utf-8"),
            file_name="network_ops_strategy_memo.md", mime="text/markdown",
            type="primary",
        )
