"""
Delhivery Network Intelligence -- LITE console.

Same five views as app.py, rebuilt for speed. app.py renders every tab on
every interaction; this build renders only the page you are on:

  st.navigation   -> one page function executes per rerun, not all five
  @st.fragment    -> slider moves rerun only their own section
  st.cache_data   -> the decision rule is computed once per parameter combo
  lazy imports    -> plotly loads on first chart; joblib/xgboost load only
                     when a page actually needs the route-choice bundle
  patched map     -> physics capped and switched off after stabilisation,
                     straight edges (halves the simulation bodies), opt-in

All numbers come from the same artifacts and the same formulas as app.py,
so both consoles agree to the digit.

Run:  streamlit run app_lite.py
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
OUT = ROOT / "outputs"
VIZ = OUT / "visualizations"
BUNDLE_PATH = ROOT / "models" / "route_choice.joblib"
SCORED_CSV = OUT / "route_choice_features.csv"

# ---------------------------------------------------------------------------
# Page + design system (system fonts: no webfont fetch on first paint)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Delhivery NIC · Lite",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

INK = "#e8ecf4"
MUTED = "#8b94a8"
RED = "#e63946"
TEAL = "#2a9d8f"
AMBER = "#e9c46a"
BLUE = "#4d7cfe"
EDGE = "#22304d"

st.markdown("""
<style>
html, body, [class*="css"] {
  font-family: -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
}
#MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; height: 0; }
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1380px; }

.nic-mast {
  display: flex; align-items: baseline; gap: 14px; flex-wrap: wrap;
  border-bottom: 1px solid #22304d; padding-bottom: 12px; margin-bottom: 6px;
}
.nic-logo { font-weight: 700; font-size: 1.45rem; letter-spacing: .02em; color: #e8ecf4; }
.nic-logo b { color: #e63946; }
.nic-sub {
  font-size: .76rem; color: #8b94a8; letter-spacing: .14em;
  text-transform: uppercase; font-weight: 500;
}
.nic-chip {
  margin-left: auto; font-family: Consolas, Menlo, monospace; font-size: .7rem;
  color: #2a9d8f; border: 1px solid #1d3b38; border-radius: 3px;
  background: #0f1d22; padding: 3px 9px; letter-spacing: .08em;
}

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
.kpi .val { font-family: Consolas, Menlo, monospace; font-size: 1.4rem; font-weight: 600; color: #e8ecf4; line-height: 1.1; }
.kpi .sub { font-size: .72rem; color: #8b94a8; margin-top: 5px; }

.nic-h {
  font-size: .8rem; font-weight: 600; letter-spacing: .16em;
  text-transform: uppercase; color: #8b94a8; margin: 18px 0 10px;
}
.nic-h b { color: #e8ecf4; }

.verdict { border-radius: 8px; padding: 18px 22px; margin-top: 6px; border: 1px solid; background: #121a2b; }
.verdict .vtitle { font-size: 1.25rem; font-weight: 700; }
.verdict .vline { font-size: .85rem; color: #aab3c5; margin-top: 6px; }
.verdict .vnum { font-family: Consolas, Menlo, monospace; color: #e8ecf4; }

.memo {
  background: #f4f1ea; color: #1d2430; border-radius: 6px;
  padding: 34px 40px; font-family: Georgia, 'Times New Roman', serif;
  line-height: 1.55; font-size: .94rem;
}
.smallnote { font-size: .74rem; color: #67708a; }
hr { border-color: #22304d; }
</style>
""", unsafe_allow_html=True)


def style_fig(fig, height: int = 360):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="-apple-system, Segoe UI, Roboto, sans-serif", color=MUTED, size=12),
        margin=dict(l=10, r=10, t=36, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hoverlabel=dict(font_family="Consolas, Menlo, monospace"),
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
# Data access -- per-file cached loaders, so a page only parses what it shows
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_csv(name: str) -> pd.DataFrame | None:
    path = OUT / name
    return pd.read_csv(path) if path.exists() else None


@st.cache_data(show_spinner=False)
def load_scored() -> pd.DataFrame | None:
    """Scored dispatches: only the 7 columns the decision rule + charts need
    (the full file has 33)."""
    if not SCORED_CSV.exists():
        return None
    return pd.read_csv(SCORED_CSV, usecols=[
        "osrm_distance", "route_type", "src_bottleneck_score",
        "pred_time_ftl", "pred_time_cart", "pred_p90_ftl", "pred_p90_cart",
    ])


@st.cache_resource(show_spinner="Loading route-choice models (first visit only) ...")
def get_bundle() -> dict | None:
    """joblib + xgboost are imported here, not at app start."""
    import joblib
    return joblib.load(BUNDLE_PATH) if BUNDLE_PATH.exists() else None


def apply_rule(scored: pd.DataFrame, bundle: dict, **overrides) -> pd.DataFrame:
    """Cost model + decision rule -- same math as task4_route_choice.apply_decision,
    inlined so the dashboard never imports the matplotlib/sklearn/xgboost chain."""
    p = {**bundle["default_params"], **overrides}
    out = scored.copy()

    dist = out["osrm_distance"].fillna(0).to_numpy(dtype=float)
    out["cost_cart"] = p["cart_per_km"] * dist
    out["cost_ftl"] = p["ftl_fixed"] + p["ftl_per_km"] * dist
    out["delta_cost"] = out["cost_ftl"] - out["cost_cart"]

    out["delta_time_min"] = out["pred_time_cart"] - out["pred_time_ftl"]
    out["delta_risk_min"] = out["pred_p90_cart"] - out["pred_p90_ftl"]

    bn = (out["src_bottleneck_score"].fillna(0) / bundle["bottleneck_p95"]).clip(0, 1)
    out["vot_effective"] = p["value_of_time"] * (1.0 + p["risk_gain"] * bn)

    weighted_saving = out["delta_time_min"] + p["risk_weight"] * out["delta_risk_min"]
    out["benefit_ftl"] = out["vot_effective"] * weighted_saving
    out["recommended_route"] = np.where(
        out["benefit_ftl"] > out["delta_cost"], "FTL", "Carting"
    )
    out["break_even_vot"] = np.where(
        weighted_saving > 1e-9, out["delta_cost"] / weighted_saving, np.inf
    )

    out["switch"] = out["recommended_route"] != out["route_type"]
    lo_hi = bundle["distance_support"]
    rec_lo = out["recommended_route"].map({rt: lo_hi[rt][0] for rt in lo_hi})
    rec_hi = out["recommended_route"].map({rt: lo_hi[rt][1] for rt in lo_hi})
    out["extrapolated"] = (out["osrm_distance"] < rec_lo) | (out["osrm_distance"] > rec_hi)

    rec_is_ftl = out["recommended_route"] == "FTL"
    out["switch_minutes_saved"] = np.where(
        rec_is_ftl, out["delta_time_min"], -out["delta_time_min"]
    )
    out["switch_extra_cost"] = np.where(
        rec_is_ftl, out["delta_cost"], -out["delta_cost"]
    )
    return out


@st.cache_data(show_spinner=False, max_entries=64)
def decide(vot: float, risk_weight: float, risk_gain: float,
           ftl_fixed: float, ftl_km: float, cart_km: float) -> pd.DataFrame | None:
    """Decision rule over every scored dispatch, cached per parameter combo --
    revisiting a slider position is free."""
    scored, bundle = load_scored(), get_bundle()
    if scored is None or bundle is None:
        return None
    return apply_rule(
        scored, bundle,
        value_of_time=vot, risk_weight=risk_weight, risk_gain=risk_gain,
        ftl_fixed=ftl_fixed, ftl_per_km=ftl_km, cart_per_km=cart_km,
    )


def decide_default() -> pd.DataFrame | None:
    bundle = get_bundle()
    if bundle is None:
        return None
    p = bundle["default_params"]
    return decide(p["value_of_time"], p["risk_weight"], p["risk_gain"],
                  p["ftl_fixed"], p["ftl_per_km"], p["cart_per_km"])


def score_one(row: pd.DataFrame, bundle: dict, **overrides) -> pd.Series:
    """Counterfactual times for a single dispatch (4 model calls) + decision."""
    fill, log_hi = bundle["fill_values"], bundle["log_hi"]
    X = row[bundle["features"]].copy().fillna(fill)

    def minutes(model) -> np.ndarray:
        pred = model.predict(X).astype(float) + getattr(model, "_p90_offset", 0.0)
        return np.expm1(np.clip(pred, 0.0, log_hi))

    out = row.copy()
    out["pred_time_ftl"] = minutes(bundle["models"]["time_FTL"])
    out["pred_time_cart"] = minutes(bundle["models"]["time_Carting"])
    out["pred_p90_ftl"] = minutes(bundle["models"]["p90_FTL"])
    out["pred_p90_cart"] = minutes(bundle["models"]["p90_Carting"])
    return apply_rule(out, bundle, **overrides).iloc[0]


def build_map_file() -> Path | None:
    """Pyvis map, patched for speed: stabilisation capped, physics switched off
    once settled (the browser stops burning CPU), straight edges (each 'dynamic'
    smooth edge adds an extra body to the simulation). Written next to the
    original so the browser fetches it over HTTP instead of the websocket;
    rebuilt only when the source map is newer."""
    src = VIZ / "06_interactive_network.html"
    if not src.exists():
        return None
    dst = VIZ / "06_interactive_network_lite.html"
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        return dst

    html = src.read_text(encoding="utf-8")
    html = html.replace(
        '"physics": {"barnesHut"',
        '"physics": {"stabilization": {"iterations": 350}, "barnesHut"', 1)
    html = html.replace('"smooth": {"type": "dynamic"}', '"smooth": false', 1)
    html = html.replace(
        'network.once("stabilizationIterationsDone", function() {',
        'network.once("stabilizationIterationsDone", function() {\n'
        '                          network.setOptions({physics: {enabled: false}});', 1)
    utils = ROOT / "lib" / "bindings" / "utils.js"
    if utils.exists():
        import re
        html = re.sub(
            r'<script[^>]*src="lib/bindings/utils\.js"[^>]*>\s*</script>',
            "<script>" + utils.read_text(encoding="utf-8") + "</script>",
            html,
        )
    dst.write_text(html, encoding="utf-8")
    return dst


def run_pipeline(cmd: list[str], label: str) -> None:
    import subprocess
    with st.status(f"Running {label} ...", expanded=True) as status:
        st.markdown(f'<span class="smallnote">$ {" ".join(cmd)}</span>',
                    unsafe_allow_html=True)
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=1800)
        st.code((proc.stdout or "") + (proc.stderr or ""), language="text")
        if proc.returncode == 0:
            status.update(label=f"{label} finished", state="complete")
            st.cache_data.clear()
            st.cache_resource.clear()
        else:
            status.update(label=f"{label} failed (exit {proc.returncode})", state="error")


# ===========================================================================
# PAGE 1 -- COMMAND CENTER
# ===========================================================================

def page_overview():
    from plotly import express as px, graph_objects as go

    nm, corr = load_csv("node_metrics.csv"), load_csv("corridor_audit.csv")
    hub, comp = load_csv("hub_impact.csv"), load_csv("model_comparison.csv")
    if nm is None or corr is None:
        st.warning("Run `python build_graph.py` first - graph artifacts are missing.")
        return

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
        ("Graph model lift", f"−{mae_gain:.0f}% MAE",
         f"{best_row['model']} vs baseline" if best_row is not None else "", TEAL),
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
    nm_top = nm[nm["in_degree"] + nm["out_degree"] >= 5]
    fig = px.scatter(
        nm_top, x="betweenness", y="avg_incoming_delay_factor",
        size="in_degree", color="bottleneck_score",
        color_continuous_scale=["#22304d", "#4d7cfe", "#e9c46a", "#e63946"],
        hover_name="name", size_max=26, render_mode="webgl",
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
# PAGE 2 -- NETWORK MAP
# ===========================================================================

def page_map():
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

    section("Interactive logistics graph")
    st.markdown(
        '<p class="smallnote">1,590 nodes · 2,481 corridors. The layout settles once, '
        "then physics switches off so the page stays responsive. Drag to pan · "
        "scroll to zoom · hover a node for hub stats.</p>",
        unsafe_allow_html=True,
    )
    if st.toggle("Load interactive map", value=False,
                 help="Loaded on demand so the rest of the console stays instant."):
        map_file = build_map_file()
        if map_file is None:
            st.warning("Map not found - run `python visualize_graph.py` to generate "
                       "outputs/visualizations/06_interactive_network.html.")
        else:
            st.iframe(map_file, height=760)


# ===========================================================================
# PAGE 3 -- MODEL LAB
# ===========================================================================

@st.fragment
def model_lab_body(comp: pd.DataFrame):
    from plotly import express as px, graph_objects as go

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
    mae_rng = max(pool["mae"].max() - pool["mae"].min(), 1e-9)
    w15_rng = max(pool["within_15_pct"].max() - pool["within_15_pct"].min(), 1e-9)
    mae_score = (pool["mae"].max() - pool["mae"]) / mae_rng
    w15_score = (pool["within_15_pct"] - pool["within_15_pct"].min()) / w15_rng
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
                      textfont=dict(size=11, color=INK, family="Consolas, monospace"))
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


def page_models():
    comp = load_csv("model_comparison.csv")
    if comp is None:
        st.warning("Run `python task3_eta_model.py --retrain` first.")
        return
    model_lab_body(comp)


# ===========================================================================
# PAGE 4 -- ROUTE ADVISOR
# ===========================================================================

@st.fragment
def route_advisor_body():
    from plotly import express as px, graph_objects as go

    bundle = get_bundle()
    defaults = bundle["default_params"]

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
                        float(defaults["value_of_time"]), 1.0)
        risk_weight = st.slider("Tail-risk weight (p90 minutes)", 0.0, 2.0,
                                float(defaults["risk_weight"]), 0.05,
                                help="How much a p90 (worst-case) minute counts "
                                     "relative to an expected minute.")
    with pc2:
        risk_gain = st.slider("Bottleneck-source uplift", 0.0, 1.5,
                              float(defaults["risk_gain"]), 0.05,
                              help="Extra value placed on FTL reliability when the "
                                   "source hub is a structural bottleneck.")
        ftl_fixed = st.number_input("FTL fixed cost (₹/dispatch)", 0.0, 20000.0,
                                    float(defaults["ftl_fixed"]), 100.0)
    with pc3:
        ftl_km = st.number_input("FTL rate (₹/km)", 0.0, 200.0,
                                 float(defaults["ftl_per_km"]), 1.0)
        cart_km = st.number_input("Carting rate (₹/km)", 0.0, 200.0,
                                  float(defaults["cart_per_km"]), 1.0)

    live = decide(vot, risk_weight, risk_gain, ftl_fixed, ftl_km, cart_km)
    params = dict(value_of_time=vot, risk_weight=risk_weight, risk_gain=risk_gain,
                  ftl_fixed=ftl_fixed, ftl_per_km=ftl_km, cart_per_km=cart_km)

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
        plot_df = live.sample(min(len(live), 2500), random_state=7)
        fig = px.scatter(
            plot_df, x="delta_time_min", y="delta_cost",
            color="recommended_route",
            color_discrete_map={"FTL": RED, "Carting": TEAL},
            opacity=0.45, render_mode="webgl",
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

    simulator(params)


def simulator(params: dict) -> None:
    section("Dispatch simulator — score a single shipment")
    nm = load_csv("node_metrics.csv")
    bundle = get_bundle()
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
        r = score_one(one, bundle, **params)
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


def page_route():
    if get_bundle() is None or load_scored() is None:
        st.warning("Run `python task4_route_choice.py --retrain` first - the route-choice "
                   "bundle / scored dispatches are missing.")
        return
    route_advisor_body()

    section("Pipeline runner")
    with st.expander("Re-run the Task-4 framework on the full dataset"):
        import sys
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
# PAGE 5 -- STRATEGY MEMO
# ===========================================================================

@st.fragment
def memo_body(hub: pd.DataFrame, corr: pd.DataFrame, comp: pd.DataFrame | None):
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

    best = comp.loc[comp["mae"].idxmin()] if comp is not None else None
    base = comp[comp["model"] == "baseline"].iloc[0] if comp is not None else None
    sw_line = ""
    dflt = decide_default()
    if dflt is not None:
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


def page_memo():
    hub, corr = load_csv("hub_impact.csv"), load_csv("corridor_audit.csv")
    comp = load_csv("model_comparison.csv")
    if hub is None:
        st.warning("Run `python task5_hub_impact.py` first.")
        return
    memo_body(hub, corr, comp)


# ===========================================================================
# Masthead + navigation (only the selected page function runs)
# ===========================================================================

def _in_streamlit() -> bool:
    """True under `streamlit run` / AppTest; False on a bare import (tests)."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


def render() -> None:
    stamp = datetime.now().strftime("%d %b %Y · %H:%M")
    st.markdown(
    f"""
    <div class="nic-mast">
      <span class="nic-logo">delhivery<b>·</b>nic</span>
      <span class="nic-sub">Network Intelligence Console · Lite</span>
      <span class="nic-chip">GRAPH LIVE · {stamp}</span>
    </div>
    """,
        unsafe_allow_html=True,
    )

    pg = st.navigation([
        st.Page(page_overview, title="Command Center", icon=":material/monitoring:", default=True),
        st.Page(page_map, title="Network Map", icon=":material/hub:"),
        st.Page(page_models, title="Model Lab", icon=":material/science:"),
        st.Page(page_route, title="Route Advisor", icon=":material/route:"),
        st.Page(page_memo, title="Strategy Memo", icon=":material/description:"),
    ])
    with st.sidebar:
        st.markdown(
            '<p class="smallnote">Lite build — pages render on demand, sliders re-run '
            'only their own section, models load on first use. Same artifacts and '
            'formulas as the full console (app.py).</p>',
            unsafe_allow_html=True,
        )
    pg.run()


if _in_streamlit():
    render()
