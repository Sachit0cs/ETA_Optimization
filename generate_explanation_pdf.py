"""
Generates a detailed PDF explanation of build_graph.py and visualize_graph.py.
Run: python generate_explanation_pdf.py
Output: ETA_Optimization_Code_Explanation.pdf
"""

from fpdf import FPDF
import os

OUTPUT_PATH = "ETA_Optimization_Code_Explanation.pdf"

# ─────────────────────────────────────────────────────────────
# PDF class with header/footer
# ─────────────────────────────────────────────────────────────
class PDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, "ETA Optimization -- Code Explanation", align="L")
        self.ln(2)
        self.set_draw_color(180, 180, 180)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-14)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def chapter_title(self, text):
        self.set_font("Helvetica", "B", 15)
        self.set_fill_color(30, 64, 120)
        self.set_text_color(255, 255, 255)
        self.cell(0, 10, text, fill=True); self.ln(10)
        self.set_text_color(0, 0, 0)
        self.ln(3)

    def section_title(self, text):
        self.set_font("Helvetica", "B", 12)
        self.set_fill_color(220, 230, 245)
        self.set_text_color(20, 40, 90)
        self.cell(0, 8, text, fill=True); self.ln(8)
        self.set_text_color(0, 0, 0)
        self.ln(2)

    def sub_title(self, text):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(60, 60, 140)
        self.multi_cell(0, 6, text)
        self.set_text_color(0, 0, 0)
        self.ln(1)

    def body(self, text):
        self.set_font("Helvetica", "", 9.5)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 5.5, text)
        self.ln(1)

    def code_block(self, text):
        self.set_font("Courier", "", 8.5)
        self.set_fill_color(240, 242, 246)
        self.set_text_color(30, 30, 30)
        self.set_draw_color(200, 200, 210)
        x = self.get_x()
        self.multi_cell(0, 5, text, border=1, fill=True)
        self.set_text_color(0, 0, 0)
        self.ln(2)

    def math_block(self, text):
        self.set_font("Courier", "B", 9)
        self.set_fill_color(255, 250, 230)
        self.set_draw_color(220, 180, 50)
        self.set_text_color(80, 40, 0)
        self.multi_cell(0, 5.5, text, border=1, fill=True)
        self.set_text_color(0, 0, 0)
        self.ln(2)

    def bullet(self, text):
        self.set_font("Helvetica", "", 9.5)
        self.set_text_color(30, 30, 30)
        self.cell(6)
        self.multi_cell(0, 5.5, f"*  {text}")
        self.ln(0.5)

    def note(self, text):
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(90, 90, 90)
        self.multi_cell(0, 5.5, f"Note: {text}")
        self.set_text_color(0, 0, 0)
        self.ln(1)

# ─────────────────────────────────────────────────────────────
# Build PDF
# ─────────────────────────────────────────────────────────────
pdf = PDF()
pdf.set_auto_page_break(auto=True, margin=18)
pdf.set_margins(14, 18, 14)
pdf.add_page()

# ═══════════════════════════════════════════════════════════════
# COVER / INTRO
# ═══════════════════════════════════════════════════════════════
pdf.set_font("Helvetica", "B", 22)
pdf.set_text_color(20, 40, 100)
pdf.ln(6)
pdf.cell(0, 12, "ETA Optimization -- Deep Code Explanation", align="C"); pdf.ln(12)
pdf.set_font("Helvetica", "", 11)
pdf.set_text_color(80, 80, 80)
pdf.cell(0, 8, "build_graph.py  &  visualize_graph.py", align="C"); pdf.ln(8)
pdf.set_text_color(0, 0, 0)
pdf.ln(5)

pdf.set_font("Helvetica", "", 10)
intro = (
    "This document provides a complete, line-by-line explanation of the two core Python files "
    "in the ETA Optimization project built on Delhivery logistics data. It covers the full "
    "mathematical foundations -- how nodes are defined, how edge weights are derived and stratified, "
    "how graph metrics are computed, and what every visualization is showing and why. "
    "The goal is to give you a thorough understanding of every design decision in the codebase."
)
pdf.multi_cell(0, 6, intro)
pdf.ln(4)

# ─── TABLE OF CONTENTS (simple) ───────────────────────────────
pdf.set_font("Helvetica", "B", 11)
pdf.set_fill_color(245, 245, 250)
pdf.cell(0, 8, "Table of Contents", fill=True); pdf.ln(8)
pdf.set_font("Helvetica", "", 10)
toc = [
    ("1", "build_graph.py -- Overview & Constants"),
    ("2", "Data Pipeline: load_and_preprocess()"),
    ("3", "Departure Record Extraction: extract_leg_records()"),
    ("4", "Edge Weight Mathematics: build_edge_weight_table()"),
    ("5", "Graph Construction: build_graph()  -- Node & Edge Structure"),
    ("6", "Graph Metrics Mathematics: compute_graph_metrics()"),
    ("7", "Corridor Audit: identify_delayed_corridors()"),
    ("8", "Orchestration: main()"),
    ("9", "visualize_graph.py -- Overview"),
    ("10", "Spring Layout Mathematics: compute_layout()"),
    ("11", "Plot 1: Full Network Visualization"),
    ("12", "Plot 2: Bottleneck Hubs"),
    ("13", "Plot 3: Delayed Corridors Bar Chart"),
    ("14", "Plot 4: Degree & Betweenness Distributions"),
    ("15", "Plot 5: Top-5 Hubs Subgraph"),
    ("16", "Plot 6: Interactive pyvis HTML"),
    ("17", "End-to-End Data Flow Summary"),
]
for num, title in toc:
    pdf.multi_cell(0, 6, f"{num}.  {title}")
pdf.ln(4)

# ═══════════════════════════════════════════════════════════════
# SECTION 1 -- build_graph.py OVERVIEW & CONSTANTS
# ═══════════════════════════════════════════════════════════════
pdf.add_page()
pdf.chapter_title("1.  build_graph.py -- Overview & Constants")

pdf.body(
    "build_graph.py is the core pipeline that transforms raw Delhivery delivery scan records "
    "into a directed weighted graph. The graph encodes the logistics network: every warehouse, "
    "sorting hub, or dispatch center is a node; every route between two facilities is an edge; "
    "and the edge weight captures how much slower (or faster) actual deliveries are compared to "
    "the theoretical OSRM road-routing estimate.\n\n"
    "The file is entirely self-contained -- one run produces six output files consumed by the "
    "ML prediction layer and the visualization script."
)

pdf.section_title("Module-Level Constants")
pdf.code_block(
    "DATA_PATH        = \"delivery_data.csv\"\n"
    "OUTPUT_DIR       = \"outputs\"\n"
    "SPARSE_THRESHOLD = 5       # corridors with fewer trips use route_type fallback\n"
    "DELAY_THRESHOLD  = 1.2     # factor > this = chronically delayed"
)
pdf.sub_title("DATA_PATH")
pdf.body(
    "Hardcoded path to the input CSV. This is the raw export from the Delhivery logistics "
    "platform containing one row per delivery scan event."
)
pdf.sub_title("OUTPUT_DIR")
pdf.body(
    "All six output artefacts (graph pickle, CSVs, summary text) are written into this "
    "directory. os.makedirs(..., exist_ok=True) in main() ensures it is created if absent."
)
pdf.sub_title("SPARSE_THRESHOLD = 5")
pdf.body(
    "A corridor's delay factor is estimated from its sample of trips. With fewer than 5 trips "
    "the sample median is statistically unreliable -- a single anomalous shipment can shift it "
    "dramatically. Any corridor below this threshold is flagged is_sparse=True so the ML "
    "pipeline knows to substitute a more reliable fallback weight (route-type median or global "
    "median) instead of trusting the raw edge weight."
)
pdf.sub_title("DELAY_THRESHOLD = 1.2")
pdf.math_block(
    "  delay_factor = actual_time / osrm_time\n\n"
    "  If delay_factor > 1.2  =>  actual transit took at least 20% longer than OSRM predicted.\n"
    "  This corridor is labeled 'chronically delayed'."
)
pdf.body(
    "OSRM (Open Source Routing Machine) computes the theoretically fastest road time given "
    "current map geometry and speed limits. It does not model traffic, loading times, "
    "or operational inefficiencies. A factor of 1.0 means perfectly on-time; 1.5 means "
    "50% slower than theoretical; 0.95 would mean faster (rare, suggests OSRM overestimates)."
)

# ═══════════════════════════════════════════════════════════════
# SECTION 2 -- load_and_preprocess
# ═══════════════════════════════════════════════════════════════
pdf.add_page()
pdf.chapter_title("2.  Data Pipeline: load_and_preprocess(path)")

pdf.code_block(
    "def load_and_preprocess(path: str) -> pd.DataFrame:\n"
    "    df = pd.read_csv(path)\n\n"
    "    for col in [\"od_start_time\", \"od_end_time\", \"trip_creation_time\"]:\n"
    "        if col in df.columns:\n"
    "            df[col] = pd.to_datetime(df[col], errors=\"coerce\")\n\n"
    "    df[\"hour\"] = df[\"od_start_time\"].dt.hour\n"
    "    df[\"tod_bucket\"] = pd.cut(\n"
    "        df[\"hour\"],\n"
    "        bins=[0, 6, 12, 18, 24],\n"
    "        labels=[\"night\", \"morning\", \"afternoon\", \"evening\"],\n"
    "        right=False,\n"
    "    )\n\n"
    "    if df[\"is_cutoff\"].dtype == object:\n"
    "        df[\"is_cutoff\"] = df[\"is_cutoff\"].str.strip().str.lower().map(\n"
    "            {\"true\": True, \"false\": False}\n"
    "        )\n"
    "    return df"
)

pdf.section_title("Line-by-line")
pdf.sub_title("df = pd.read_csv(path)")
pdf.body(
    "Reads the entire delivery CSV into a Pandas DataFrame. Each row is one scan event -- "
    "a record of a package being scanned at a facility at a specific moment."
)

pdf.sub_title("pd.to_datetime(df[col], errors='coerce')")
pdf.body(
    "Converts each of the three timestamp columns from raw strings (e.g. '2023-06-14 08:32:00') "
    "to Pandas Timestamp objects. errors='coerce' means any unparseable string silently becomes "
    "NaT (Not a Time) instead of raising an exception. This keeps the pipeline robust against "
    "malformed or missing timestamps without halting execution."
)

pdf.sub_title("df['hour'] = df['od_start_time'].dt.hour")
pdf.body(
    "Extracts the integer hour (0-23) from the od_start_time of each leg. This single integer "
    "is the basis for the time-of-day stratification. od_start_time is the moment the package "
    "departed from the source facility on this leg -- the departure hour is what determines "
    "which traffic/congestion regime the shipment encountered."
)

pdf.sub_title("pd.cut() -- Time-of-Day Buckets")
pdf.math_block(
    "  bins   = [0, 6, 12, 18, 24]   (right=False means left-inclusive, right-exclusive)\n\n"
    "  hour in [ 0,  6)  =>  tod_bucket = 'night'      (midnight to 5:59 AM)\n"
    "  hour in [ 6, 12)  =>  tod_bucket = 'morning'    (6:00 AM to 11:59 AM)\n"
    "  hour in [12, 18)  =>  tod_bucket = 'afternoon'  (noon to 5:59 PM)\n"
    "  hour in [18, 24)  =>  tod_bucket = 'evening'    (6:00 PM to 11:59 PM)"
)
pdf.body(
    "Why 6-hour windows? Finer bins (e.g. hourly) would produce very sparse groups for "
    "low-volume corridors, making median estimates unreliable. Coarser bins (e.g. day/night) "
    "would miss real intra-day variation. 6-hour windows capture the four major operational "
    "regimes -- overnight quiet, morning rush, afternoon congestion, evening wind-down -- "
    "while keeping enough trips per cell for stable statistics."
)

pdf.sub_title("is_cutoff normalisation")
pdf.body(
    "The is_cutoff column flags whether a scan is a 'cutoff' event (a deadline check) or a "
    "departure scan. When a CSV is exported and re-read, boolean True/False may be stored as "
    "the strings 'True' / 'False'. The .str.strip().str.lower().map(...) chain converts "
    "'True', ' True', 'TRUE' all to Python True, and similarly for False. Without this, "
    "the downstream boolean filter ~df['is_cutoff'] would fail or silently include wrong rows."
)

# ═══════════════════════════════════════════════════════════════
# SECTION 3 -- extract_leg_records
# ═══════════════════════════════════════════════════════════════
pdf.add_page()
pdf.chapter_title("3.  Departure Record Extraction: extract_leg_records(df)")

pdf.code_block(
    "def extract_leg_records(df: pd.DataFrame) -> pd.DataFrame:\n"
    "    \"\"\"\n"
    "    is_cutoff=False rows are the departure scan for each OD leg.\n"
    "    actual_time and osrm_time at departure = full remaining time for the leg.\n"
    "    \"\"\"\n"
    "    departures = df[~df[\"is_cutoff\"]].copy()\n"
    "    return departures"
)

pdf.section_title("Why only is_cutoff=False rows?")
pdf.body(
    "A single physical delivery leg generates multiple scan events in the database:\n"
)
pdf.bullet("A departure scan when the vehicle leaves the source facility (is_cutoff=False)")
pdf.bullet("One or more cutoff scans that check whether deadline targets are met (is_cutoff=True)")
pdf.bullet("An arrival scan at the destination")
pdf.ln(2)
pdf.body(
    "The delay factor (actual_time / osrm_time) is only meaningful when captured at the "
    "departure moment. At that point, actual_time represents the total time remaining for "
    "the entire leg from source to destination, and osrm_time is the corresponding OSRM "
    "theoretical estimate. Taking the ratio gives a clean per-leg delay measurement.\n\n"
    "If cutoff scan rows were included, the ratio would represent only a fragment of the leg "
    "(from the cutoff checkpoint to the destination), not the full corridor performance. "
    "Mixing full-leg and partial-leg ratios would contaminate the statistics."
)

pdf.math_block(
    "  factor (delay ratio) = actual_time / osrm_time\n\n"
    "  Where:\n"
    "    actual_time  = real elapsed transit time for this OD leg (from scan records)\n"
    "    osrm_time    = OSRM road-routing estimate for the same source -> destination\n\n"
    "  factor = 1.0  => exactly on OSRM estimate (rare in practice)\n"
    "  factor = 1.5  => took 50% longer than OSRM predicted\n"
    "  factor = 0.9  => slightly faster than OSRM estimate"
)

pdf.sub_title(".copy()")
pdf.body(
    "The .copy() ensures departures is a fully independent DataFrame, not a view of df. "
    "This prevents Pandas SettingWithCopyWarning when columns are later added or modified "
    "on the departures slice."
)

# ═══════════════════════════════════════════════════════════════
# SECTION 4 -- build_edge_weight_table  (MATH HEAVY)
# ═══════════════════════════════════════════════════════════════
pdf.add_page()
pdf.chapter_title("4.  Edge Weight Mathematics: build_edge_weight_table(departures)")

pdf.code_block(
    "def build_edge_weight_table(departures):\n"
    "    rt_medians    = departures.groupby('route_type')['factor'].median().to_dict()\n"
    "    global_median = float(departures['factor'].median())\n\n"
    "    edge_stats = (\n"
    "        departures\n"
    "        .groupby(['source_center','destination_center','route_type','tod_bucket'],\n"
    "                 observed=True)\n"
    "        .agg(\n"
    "            median_factor   = ('factor', 'median'),\n"
    "            trip_count      = ('trip_uuid', 'nunique'),\n"
    "            median_osrm_dist= ('osrm_distance', 'median'),\n"
    "            median_osrm_time= ('osrm_time', 'median'),\n"
    "            pct_delayed     = ('factor', lambda x: (x > DELAY_THRESHOLD).mean()),\n"
    "        ).reset_index()\n"
    "    )\n\n"
    "    edge_rt_agg = (\n"
    "        departures\n"
    "        .groupby(['source_center','destination_center','route_type'])\n"
    "        .agg(\n"
    "            overall_median_factor = ('factor', 'median'),\n"
    "            total_trips           = ('trip_uuid', 'nunique'),\n"
    "            median_osrm_dist      = ('osrm_distance', 'median'),\n"
    "            median_osrm_time      = ('osrm_time', 'median'),\n"
    "            pct_delayed           = ('factor', lambda x: (x > DELAY_THRESHOLD).mean()),\n"
    "        ).reset_index()\n"
    "    )\n"
    "    return edge_stats, edge_rt_agg, rt_medians, global_median"
)

pdf.section_title("Why Median, not Mean?")
pdf.body(
    "Delivery times follow a heavily right-skewed distribution. A single delayed truck "
    "(mechanical breakdown, accident, border inspection) can take 3-10x the expected time. "
    "The arithmetic mean is pulled upward by these outliers and would overestimate the "
    "typical delay. The median is the 50th percentile -- it represents what a randomly "
    "chosen trip actually experiences, unaffected by extreme values."
)
pdf.math_block(
    "  Given n trips on corridor (A->B), with factors f_1, f_2, ..., f_n:\n\n"
    "  mean   = (f_1 + f_2 + ... + f_n) / n          [sensitive to outliers]\n"
    "  median = middle value after sorting             [robust to outliers]\n\n"
    "  Example:  factors = [1.0, 1.1, 1.0, 1.2, 8.5]  (one breakdown)\n"
    "    mean   = 12.8 / 5 = 2.56   <-- severely inflated\n"
    "    median = 1.1                <-- representative of typical trip"
)

pdf.section_title("The Four-Level Fallback Hierarchy")
pdf.body("The function returns four objects that form a lookup hierarchy for edge weights:")

pdf.sub_title("Level 1 -- edge_stats: (src, dst, route_type, tod_bucket)")
pdf.body(
    "The most granular level. Groups trips by the exact origin-destination corridor, "
    "the transport mode (route_type), AND the time-of-day window. "
    "For example: Mumbai Warehouse -> Delhi Hub, surface route, afternoon trips. "
    "This captures the joint effect of route type and time of day."
)

pdf.sub_title("Level 2 -- edge_rt_agg: (src, dst, route_type)")
pdf.body(
    "Coarser: all trips on a corridor aggregated by route type only, ignoring time of day. "
    "Used as the primary edge attribute in the graph. Route type is preserved because "
    "a surface corridor and an air corridor between the same two cities have fundamentally "
    "different speed profiles and should never be averaged together."
)

pdf.sub_title("Level 3 -- rt_medians: route_type -> float")
pdf.body(
    "For sparse corridors (< 5 trips) where even the route-type-aggregated median may be "
    "unreliable, the ML pipeline falls back to the global median delay factor for that "
    "route type across the entire network."
)

pdf.sub_title("Level 4 -- global_median: float")
pdf.body(
    "The absolute last resort -- the median of all factor values across every trip, every "
    "corridor, every route type. Used only when even the route-type median is unavailable."
)

pdf.section_title("Aggregated Statistics Explained")

pdf.sub_title("trip_count = ('trip_uuid', 'nunique')")
pdf.body(
    "Counts distinct trip IDs (not rows). A single trip may have multiple scan rows; "
    "nunique prevents inflating the count by deduplicating on the logical trip identifier."
)

pdf.sub_title("pct_delayed = lambda x: (x > DELAY_THRESHOLD).mean()")
pdf.math_block(
    "  pct_delayed = (number of trips where factor > 1.2) / (total trips in group)\n\n"
    "  Equivalent to: mean of a boolean series [True if factor>1.2, else False]\n"
    "  Result: a value between 0.0 (no delayed trips) and 1.0 (all trips delayed)"
)
pdf.body(
    "This is different from the median factor. A corridor could have median_factor=1.1 "
    "(typically slight delay) but pct_delayed=0.4 (40% of trips exceed 1.2x) if there "
    "is a bimodal distribution. Both metrics together paint a fuller picture."
)

pdf.sub_title("observed=True in groupby")
pdf.body(
    "tod_bucket is a Pandas Categorical column (created by pd.cut). Without observed=True, "
    "groupby would generate rows for every (src, dst, route_type, tod_bucket) combination "
    "including those with zero actual trips -- producing many NaN rows. observed=True "
    "restricts output to groups that actually appear in the data."
)

# ═══════════════════════════════════════════════════════════════
# SECTION 5 -- build_graph  (NODE & EDGE STRUCTURE)
# ═══════════════════════════════════════════════════════════════
pdf.add_page()
pdf.chapter_title("5.  Graph Construction: build_graph() -- Node & Edge Structure")

pdf.section_title("What is a NetworkX DiGraph?")
pdf.body(
    "A DiGraph (Directed Graph) is a mathematical structure G = (V, E) where:\n"
    "  V = set of vertices (nodes)\n"
    "  E = set of ordered pairs (u, v) called directed edges (arcs)\n\n"
    "Unlike an undirected graph, (u, v) and (v, u) are different edges. "
    "This is appropriate for logistics because a route from Mumbai to Delhi may have "
    "very different delay characteristics than Delhi to Mumbai (different traffic, "
    "different load factors, different routes)."
)

pdf.section_title("Node Structure")
pdf.code_block(
    "# Build facility ID -> name lookup\n"
    "node_meta: dict = {}\n"
    "for _, row in df[['source_center','source_name']].drop_duplicates().iterrows():\n"
    "    node_meta[row['source_center']] = (\n"
    "        str(row['source_name']) if pd.notna(row['source_name'])\n"
    "        else str(row['source_center'])\n"
    "    )\n\n"
    "all_centers = set(df['source_center'].unique()) | set(df['destination_center'].unique())\n"
    "for center in all_centers:\n"
    "    G.add_node(center, name=node_meta.get(center, str(center)))"
)
pdf.body(
    "Every logistics facility (warehouse, hub, dispatch center, pickup point) becomes a "
    "node. The node's identifier is its numeric or alphanumeric center_id. Each node "
    "carries one attribute: name -- the human-readable facility name.\n\n"
    "The name is populated from the source_name and destination_name columns. The "
    "pd.notna() guard handles cases where a facility appears only as a destination and "
    "has a missing source_name -- it falls back to the string form of the center ID.\n\n"
    "The union of source_center and destination_center IDs ensures every facility "
    "mentioned anywhere in the data gets a node, even if it only appears as a "
    "destination (and thus wouldn't be in source_center alone)."
)

pdf.math_block(
    "  Node v in V:\n"
    "    v.id   = center_id      (unique facility identifier)\n"
    "    v.name = facility_name  (human-readable label, e.g. 'Delhi NCR Hub')"
)

pdf.section_title("Edge Structure -- First Pass (route-type aggregated)")
pdf.code_block(
    "for (src, dst), grp in edge_rt_agg.groupby(['source_center','destination_center']):\n"
    "    total_trips    = int(grp['total_trips'].sum())\n"
    "    weighted_factor = float(\n"
    "        np.average(grp['overall_median_factor'], weights=grp['total_trips'])\n"
    "    )\n"
    "    G.add_edge(src, dst,\n"
    "        weight          = weighted_factor,\n"
    "        total_trips     = total_trips,\n"
    "        median_osrm_dist= median_dist,\n"
    "        pct_delayed     = pct_delayed,\n"
    "        is_sparse       = (total_trips < SPARSE_THRESHOLD),\n"
    "        route_types     = rt_breakdown,\n"
    "        tod_lookup      = {},\n"
    "    )"
)

pdf.section_title("How the Scalar Edge Weight is Computed")
pdf.body(
    "A corridor between two facilities may be served by multiple route types simultaneously "
    "(e.g. both a surface truck route and an air freight route). Each route type has its "
    "own median delay factor. To produce a single scalar weight for use in graph algorithms "
    "(shortest path, betweenness), a trip-count weighted average is used:"
)
pdf.math_block(
    "  Let RT = {route types serving corridor (src, dst)}\n"
    "  Let m_r = overall_median_factor for route type r\n"
    "  Let n_r = total_trips for route type r\n\n"
    "  weighted_factor = SUM(m_r * n_r for r in RT) / SUM(n_r for r in RT)\n\n"
    "  = np.average(median_factors, weights=trip_counts)\n\n"
    "  Intuition: a route type that carries more trips should have proportionally\n"
    "  more influence on the combined edge weight."
)

pdf.body(
    "The result is stored as the 'weight' attribute on the edge. NetworkX algorithms "
    "that accept a weight parameter (betweenness_centrality, shortest_path) will use "
    "this value. Higher weight = slower corridor = effectively 'longer' path in delay "
    "terms."
)

pdf.section_title("Route-Type Breakdown (rt_breakdown)")
pdf.body(
    "In addition to the scalar weight, each edge stores a nested dictionary keyed by "
    "route_type, containing the per-type statistics. This allows the ML prediction layer "
    "to retrieve the weight for a specific transport mode on a corridor at runtime:"
)
pdf.code_block(
    "edge['route_types'] = {\n"
    "  'surface': {'overall_median_factor': 1.18, 'total_trips': 340, ...},\n"
    "  'air':     {'overall_median_factor': 1.05, 'total_trips': 42,  ...},\n"
    "}"
)

pdf.section_title("Edge Structure -- Second Pass (time-of-day lookup)")
pdf.code_block(
    "for _, row in edge_stats.iterrows():\n"
    "    src, dst = row['source_center'], row['destination_center']\n"
    "    if G.has_edge(src, dst):\n"
    "        key = (row['route_type'], str(row['tod_bucket']))\n"
    "        G[src][dst]['tod_lookup'][key] = {\n"
    "            'median_factor': row['median_factor'],\n"
    "            'trip_count':    int(row['trip_count']),\n"
    "            'pct_delayed':   row['pct_delayed'],\n"
    "        }"
)
pdf.body(
    "After all edges exist, the fine-grained time-of-day statistics are attached. "
    "tod_lookup is keyed by (route_type, tod_bucket) tuples. At prediction time, the "
    "ML model queries:\n"
)
pdf.code_block(
    "  tod_lookup[('surface', 'afternoon')]\n"
    "  # => {'median_factor': 1.31, 'trip_count': 87, 'pct_delayed': 0.52}"
)
pdf.body(
    "The two-pass approach is intentional: the first pass works at route-type granularity "
    "(coarser), the second at route_type x tod_bucket granularity (finer). Doing them "
    "separately avoids a complex multi-level merge and keeps the logic easy to follow."
)

pdf.section_title("Complete Edge Attribute Summary")
pdf.math_block(
    "  Edge (src -> dst) attributes:\n\n"
    "  weight           : float   Trip-count weighted median delay factor across route types\n"
    "  total_trips      : int     Total unique trips ever observed on this corridor\n"
    "  median_osrm_dist : float   Median OSRM distance in km\n"
    "  pct_delayed      : float   Fraction of trips with factor > 1.2 (0.0 to 1.0)\n"
    "  is_sparse        : bool    True if total_trips < 5 (weight is unreliable)\n"
    "  route_types      : dict    Per route-type breakdown dict\n"
    "  tod_lookup       : dict    Keyed by (route_type, tod_bucket), value = delay stats"
)

# ═══════════════════════════════════════════════════════════════
# SECTION 6 -- compute_graph_metrics
# ═══════════════════════════════════════════════════════════════
pdf.add_page()
pdf.chapter_title("6.  Graph Metrics Mathematics: compute_graph_metrics(G)")

pdf.code_block(
    "def compute_graph_metrics(G):\n"
    "    betweenness = nx.betweenness_centrality(G, weight='weight', normalized=True)\n"
    "    in_degree   = dict(G.in_degree())\n"
    "    out_degree  = dict(G.out_degree())\n"
    "    in_deg_w    = dict(G.in_degree(weight='weight'))\n"
    "    out_deg_w   = dict(G.out_degree(weight='weight'))\n"
    "    clustering  = nx.clustering(G.to_undirected())\n\n"
    "    avg_in_delay = {}\n"
    "    for node in G.nodes():\n"
    "        preds = list(G.predecessors(node))\n"
    "        avg_in_delay[node] = (\n"
    "            float(np.mean([G[p][node]['weight'] for p in preds])) if preds else 0.0\n"
    "        )\n\n"
    "    metrics['bottleneck_score'] = (\n"
    "        metrics['betweenness'] * metrics['avg_incoming_delay_factor']\n"
    "    )"
)

pdf.section_title("Betweenness Centrality -- Mathematical Definition")
pdf.body(
    "Betweenness centrality measures how central a node is to the flow of information "
    "(or packages) through the network. A node with high betweenness lies on many "
    "shortest paths between other nodes -- if it fails or slows down, many routes are affected."
)
pdf.math_block(
    "  For node v:\n\n"
    "  BC(v) = SUM over all pairs (s,t) where s != v != t:\n"
    "              sigma(s, t | v) / sigma(s, t)\n\n"
    "  Where:\n"
    "    sigma(s, t)     = total number of shortest paths from s to t\n"
    "    sigma(s, t | v) = number of those paths that pass through v\n\n"
    "  normalized=True divides by (n-1)(n-2) so the result is in [0, 1].\n"
    "  weight='weight' means 'shortest' = lowest total delay factor (Dijkstra)."
)
pdf.note(
    "Using weight='weight' means Dijkstra finds minimum-delay paths, not minimum-hop paths. "
    "A node lying on many low-delay routes scores higher than one on many high-delay routes."
)

pdf.section_title("Degree Metrics")
pdf.math_block(
    "  in_degree(v)          = number of edges pointing INTO v\n"
    "                          (how many corridors feed packages into this facility)\n\n"
    "  out_degree(v)         = number of edges pointing OUT of v\n"
    "                          (how many onward routes this facility dispatches to)\n\n"
    "  in_degree_weighted(v) = SUM of weights on all incoming edges\n"
    "                          (total accumulated delay factor from all feeder corridors)\n\n"
    "  out_degree_weighted(v)= SUM of weights on all outgoing edges\n"
    "                          (total accumulated delay factor for all dispatch corridors)"
)

pdf.section_title("Clustering Coefficient")
pdf.body(
    "Clustering coefficient measures how tightly connected a node's neighbours are to "
    "each other. G.to_undirected() is used because directed clustering has a more "
    "complex definition and is less interpretable for operational purposes."
)
pdf.math_block(
    "  C(v) = (actual edges between v's neighbours) /\n"
    "         (maximum possible edges between v's neighbours)\n\n"
    "  C(v) = 0   => no two neighbours of v are connected to each other\n"
    "  C(v) = 1   => every pair of v's neighbours is directly connected\n\n"
    "  High clustering = v sits inside a dense cluster of mutually-connected facilities\n"
    "  Low clustering  = v is a bridge between otherwise disconnected parts of the network"
)

pdf.section_title("Average Incoming Delay Factor")
pdf.math_block(
    "  avg_in_delay(v) = MEAN of weight(p -> v) for all predecessors p of v\n\n"
    "                  = (1 / in_degree(v)) * SUM of edge weights of incoming edges\n\n"
    "  This is a proxy for how congested the inbound routes to facility v are.\n"
    "  If packages arriving at v are consistently delayed, this value will be > 1.2."
)

pdf.section_title("Bottleneck Score -- The Key Composite Metric")
pdf.math_block(
    "  bottleneck_score(v) = betweenness(v) * avg_in_delay(v)\n\n"
    "  This is a custom metric. Why multiply?\n\n"
    "  High betweenness alone => the node is structurally critical but may run on time\n"
    "  High avg_in_delay alone => the node is congested but may not be on critical paths\n\n"
    "  Only when BOTH are high does the node become a true operational bottleneck:\n"
    "  critical to the network AND suffering from actual delays.\n\n"
    "  Nodes are ranked by bottleneck_score descending => top rows = highest priority\n"
    "  targets for operational improvement."
)
pdf.note(
    "This is an ad-hoc operational metric, not a standard graph theory quantity. "
    "It is designed to answer 'which hubs should ops teams focus on first?'"
)

# ═══════════════════════════════════════════════════════════════
# SECTION 7 -- identify_delayed_corridors
# ═══════════════════════════════════════════════════════════════
pdf.add_page()
pdf.chapter_title("7.  Corridor Audit: identify_delayed_corridors(G)")

pdf.code_block(
    "def identify_delayed_corridors(G):\n"
    "    rows = []\n"
    "    for src, dst, data in G.edges(data=True):\n"
    "        rows.append({\n"
    "            'source_center':        src,\n"
    "            'destination_center':   dst,\n"
    "            'median_factor':        data['weight'],\n"
    "            'total_trips':          data['total_trips'],\n"
    "            'is_chronically_delayed': data['weight'] > DELAY_THRESHOLD,\n"
    "            ...\n"
    "        })\n"
    "    return pd.DataFrame(rows).sort_values('median_factor', ascending=False)"
)

pdf.body(
    "G.edges(data=True) iterates over all edges in the graph, yielding a triple "
    "(src_node, dst_node, attribute_dict) for each. The function reads attributes "
    "directly from the graph's edge dictionaries (built in build_graph()) rather "
    "than re-reading any CSV, ensuring consistency with the actual graph state.\n\n"
    "is_chronically_delayed = weight > 1.2 is a boolean flag based on the same "
    "DELAY_THRESHOLD constant used throughout. Sorting by median_factor descending "
    "means the worst corridors appear at the top of corridor_audit.csv, making "
    "triage straightforward for operations teams."
)

# ═══════════════════════════════════════════════════════════════
# SECTION 8 -- main()
# ═══════════════════════════════════════════════════════════════
pdf.add_page()
pdf.chapter_title("8.  Orchestration: main()")

pdf.body("main() calls all previous functions in dependency order and saves outputs.")

steps = [
    ("os.makedirs(OUTPUT_DIR, exist_ok=True)",
     "Creates the outputs/ directory if it doesn't exist. exist_ok=True means no error if it already exists."),
    ("df = load_and_preprocess(DATA_PATH)",
     "Loads and cleans the raw CSV. All 6 downstream outputs derive from this DataFrame."),
    ("departures = extract_leg_records(df)",
     "Filters to departure-only rows for clean delay factor computation."),
    ("edge_stats, edge_rt_agg, rt_medians, global_median = build_edge_weight_table(departures)",
     "Computes all delay statistics. edge_stats and edge_rt_agg are immediately saved to CSV as independent outputs before graph construction."),
    ("G = build_graph(edge_rt_agg, edge_stats, df)",
     "Constructs the DiGraph. Takes both aggregation levels plus the full df (for node name lookup)."),
    ("metrics = compute_graph_metrics(G)",
     "Computes betweenness, degree, and bottleneck scores. This is the most compute-intensive step -- betweenness is O(VE) complexity."),
    ("corridors = identify_delayed_corridors(G)",
     "Flattens graph edges to DataFrame for audit. Saved as corridor_audit.csv."),
    ("pickle.dump(G, f)",
     "Serialises the entire NetworkX DiGraph to a binary .pkl file. This preserves all edge attributes (including tod_lookup dicts) so the ML pipeline can reload the full graph without re-running the pipeline."),
    ("graph_summary.txt",
     "Human-readable text file summarising node/edge counts, sparse corridor counts, delayed corridor fraction, global median factor, route-type medians, and top-5 bottleneck hubs. Useful for quick reporting without opening a CSV."),
]
for code, explanation in steps:
    pdf.sub_title(code)
    pdf.body(explanation)
    pdf.ln(1)

pdf.sub_title("Sparse corridor count")
pdf.code_block(
    "sparse_count = sum(1 for _, _, d in G.edges(data=True) if d.get('is_sparse'))"
)
pdf.body(
    "Iterates over all graph edges and counts those with is_sparse=True (total_trips < 5). "
    "This is a diagnostic -- a high sparse_count suggests the training dataset is too small "
    "or the network has many rarely-used corridors that need special handling in the ML model."
)

# ═══════════════════════════════════════════════════════════════
# SECTION 9 -- visualize_graph.py OVERVIEW
# ═══════════════════════════════════════════════════════════════
pdf.add_page()
pdf.chapter_title("9.  visualize_graph.py -- Overview")

pdf.body(
    "visualize_graph.py is the companion visualization script. It reads the artifacts "
    "produced by build_graph.py (the graph pickle and the two metric CSVs) and produces "
    "five static PNG plots and one interactive HTML file.\n\n"
    "Key design choices:\n"
)
pdf.bullet("matplotlib.use('Agg') -- sets a non-interactive backend. This allows plots to be generated in headless server environments (no display required). 'Agg' renders to PNG in memory.")
pdf.bullet("The spring layout (force-directed) is computed once and reused for all network plots (plots 1, 2, 5) so node positions are consistent across views.")
pdf.bullet("plt.close() after each plot frees memory immediately -- important when the graph has thousands of nodes and matplotlib figure objects are large.")

pdf.section_title("load_artifacts()")
pdf.code_block(
    "def load_artifacts():\n"
    "    with open(f'{OUTPUT_DIR}/logistics_graph.pkl', 'rb') as f:\n"
    "        G = pickle.load(f)\n"
    "    metrics   = pd.read_csv(f'{OUTPUT_DIR}/node_metrics.csv')\n"
    "    corridors = pd.read_csv(f'{OUTPUT_DIR}/corridor_audit.csv')\n"
    "    return G, metrics, corridors"
)
pdf.body(
    "Deserialises the pickled DiGraph -- all edge attributes (weight, tod_lookup, etc.) "
    "are fully restored. metrics and corridors are loaded from CSV rather than the pickle "
    "because DataFrames are easier to filter and sort for plotting purposes."
)

# ═══════════════════════════════════════════════════════════════
# SECTION 10 -- compute_layout
# ═══════════════════════════════════════════════════════════════
pdf.add_page()
pdf.chapter_title("10.  Spring Layout Mathematics: compute_layout(G, seed=42)")

pdf.code_block(
    "def compute_layout(G, seed=42):\n"
    "    k = 2.5 / np.sqrt(max(G.number_of_nodes(), 1))\n"
    "    return nx.spring_layout(G, k=k, seed=seed, iterations=60)"
)

pdf.section_title("What is a Spring (Force-Directed) Layout?")
pdf.body(
    "A spring layout treats edges as physical springs and nodes as charged particles. "
    "The algorithm iteratively minimises an energy function:"
)
pdf.math_block(
    "  Forces on each node:\n\n"
    "  1. Spring (attractive) force  -- pulls connected nodes toward each other\n"
    "     F_spring(u,v) = k * distance(u,v)   for every edge (u,v)\n\n"
    "  2. Repulsive force -- pushes all nodes apart to avoid overlap\n"
    "     F_repel(u,v) = k^2 / distance(u,v)  for every pair (u,v)\n\n"
    "  At equilibrium: nodes that are densely connected cluster together;\n"
    "  loosely connected nodes spread apart."
)

pdf.section_title("The k Parameter")
pdf.math_block(
    "  k = 2.5 / sqrt(N)    where N = number of nodes\n\n"
    "  k controls the optimal distance between nodes.\n"
    "  - Large N => small k => nodes packed tighter (avoid infinite spread)\n"
    "  - Small N => large k => nodes spread further apart (avoid overlap)\n\n"
    "  The factor 2.5 is a tuning constant -- the standard NetworkX default uses 1/sqrt(N);\n"
    "  2.5 provides more spacing between nodes for readability."
)

pdf.sub_title("seed=42")
pdf.body(
    "The layout algorithm starts from random initial positions. seed=42 fixes the random "
    "number generator so the layout is reproducible -- running the script twice produces "
    "identical plots. This matters for comparing plots across runs."
)

pdf.sub_title("iterations=60")
pdf.body(
    "The algorithm runs 60 simulation steps to converge. More iterations = better layout "
    "quality but slower execution. 60 is a pragmatic balance for large logistics graphs "
    "(hundreds to thousands of nodes)."
)

# ═══════════════════════════════════════════════════════════════
# SECTION 11 -- Plot 1: Full Network
# ═══════════════════════════════════════════════════════════════
pdf.add_page()
pdf.chapter_title("11.  Plot 1: Full Network Visualization")

pdf.section_title("Node Colour and Size")
pdf.code_block(
    "norm_bw    = Normalize(vmin=0, vmax=max_bw)\n"
    "cmap_node  = cm.YlOrRd\n"
    "node_colors = [cmap_node(norm_bw(b)) for b in node_bw]\n"
    "node_sizes  = [80 + 2500 * norm_bw(b) for b in node_bw]"
)
pdf.body("Node colour maps betweenness centrality to the YlOrRd (Yellow-Orange-Red) colormap:")
pdf.math_block(
    "  norm_bw(b) = (b - 0) / (max_bw - 0) = b / max_bw   => value in [0, 1]\n\n"
    "  colour = YlOrRd(norm_bw(b)):\n"
    "    0.0 = pale yellow   (low betweenness, peripheral node)\n"
    "    1.0 = deep red      (high betweenness, critical relay hub)\n\n"
    "  node_size = 80 + 2500 * norm_bw(b):\n"
    "    min size = 80  (peripheral nodes: small dots)\n"
    "    max size = 2580 (top hub: large circle)\n"
    "    Linear scaling makes size visually encode the same variable as colour."
)

pdf.section_title("Edge Colour and Width")
pdf.code_block(
    "norm_ew     = Normalize(vmin=w_min, vmax=w_max)\n"
    "cmap_edge   = cm.RdYlGn_r\n"
    "edge_colors = [cmap_edge(norm_ew(w)) for w in edge_weights]\n"
    "edge_widths = [0.3 + 1.5 * norm_ew(w) for w in edge_weights]"
)
pdf.math_block(
    "  norm_ew(w) = (w - w_min) / (w_max - w_min)  => value in [0, 1]\n\n"
    "  colour = RdYlGn_r (reversed Red-Yellow-Green):\n"
    "    0.0 = green  (lowest delay, fastest corridor)\n"
    "    0.5 = yellow (moderate delay)\n"
    "    1.0 = red    (highest delay, slowest corridor)\n\n"
    "  edge_width = 0.3 + 1.5 * norm_ew(w):\n"
    "    min width = 0.3 (fast corridors: thin)\n"
    "    max width = 1.8 (slow corridors: thick)"
)
pdf.body(
    "Using RdYlGn_r (reversed) is counterintuitive but deliberate: in logistics, RED = BAD "
    "(high delay). The reversed colormap makes red correspond to high delay factor values."
)

pdf.sub_title("Label only top-15 by betweenness")
pdf.body(
    "Drawing text labels for every node would make the plot illegible with hundreds of "
    "facilities. Only the 15 highest-betweenness nodes get labels, truncated to 14 "
    "characters to prevent text collision."
)

pdf.sub_title("connectionstyle='arc3,rad=0.08'")
pdf.body(
    "Draws edges as slightly curved arcs instead of straight lines. This makes both "
    "directions of a bidirectional pair (e.g. A->B and B->A) visible as separate curves "
    "rather than overlapping on the same straight line."
)

# ═══════════════════════════════════════════════════════════════
# SECTION 12 -- Plot 2
# ═══════════════════════════════════════════════════════════════
pdf.add_page()
pdf.chapter_title("12.  Plot 2: Bottleneck Hubs")

pdf.body(
    "This plot isolates the operational story: which specific facilities matter most, "
    "and which corridors are chronically failing?"
)
pdf.section_title("Visual Encoding")
pdf.bullet("Top-10 bottleneck hubs (by bottleneck_score): large red circles (#d73027), size scaled by betweenness")
pdf.bullet("All other facilities: small grey circles (#cccccc), alpha=0.4 (faded background)")
pdf.bullet("Chronically delayed edges (weight > 1.2): thick red lines, width=1.8")
pdf.bullet("Normal edges: thin grey lines, width=0.3")

pdf.math_block(
    "  hub_size(v) = 800 + 4000 * (betweenness(v) / max_betweenness)\n\n"
    "  min hub size = 800  (bottom of top-10)\n"
    "  max hub size = 4800 (highest betweenness hub)\n\n"
    "  Edge colour decision:\n"
    "    if weight(u,v) > 1.2  =>  '#d73027' (red)\n"
    "    else                  =>  '#999999' (grey)"
)

# ═══════════════════════════════════════════════════════════════
# SECTION 13 -- Plot 3
# ═══════════════════════════════════════════════════════════════
pdf.add_page()
pdf.chapter_title("13.  Plot 3: Delayed Corridors Bar Chart")

pdf.code_block(
    "reliable = corridors[corridors['total_trips'] >= 5]\n"
    "top      = reliable.nlargest(top_n, 'median_factor')"
)
pdf.body(
    "Before plotting, sparse corridors (< 5 trips) are excluded. Their median_factor "
    "is statistically unreliable and would give a misleading picture. The chart shows "
    "only the top 30 most-delayed corridors among those with at least 5 observed trips."
)
pdf.section_title("Colour Thresholds")
pdf.math_block(
    "  factor > 1.5  =>  '#d73027'  (deep red)    SEVERE delay (>50% over estimate)\n"
    "  factor > 1.2  =>  '#fc8d59'  (orange)       MODERATE delay (20-50% over)\n"
    "  factor <= 1.2 =>  '#fee08b'  (pale yellow)  MILD (within 20% of estimate)"
)
pdf.sub_title("Annotation")
pdf.body(
    "ax.text() places the numeric factor value just to the right of each bar end. "
    "This avoids the reader having to map bar length to the x-axis for the exact value."
)
pdf.sub_title("Reference lines")
pdf.body(
    "A dashed vertical line at x=1.2 marks the delay threshold. A dotted line at x=1.0 "
    "marks the OSRM baseline (perfect on-time performance). Together they give the reader "
    "instant visual context."
)

# ═══════════════════════════════════════════════════════════════
# SECTION 14 -- Plot 4
# ═══════════════════════════════════════════════════════════════
pdf.add_page()
pdf.chapter_title("14.  Plot 4: Degree & Betweenness Distributions")

pdf.body(
    "Three histograms in a 1x3 grid revealing the structural properties of the network:"
)
pdf.sub_title("In-Degree Distribution (blue)")
pdf.body(
    "Shows how many incoming corridors each facility has. A highly right-skewed distribution "
    "indicates a hub-and-spoke topology where a few major hubs receive from many sources."
)
pdf.sub_title("Out-Degree Distribution (red)")
pdf.body(
    "Shows how many outgoing corridors each facility has. Asymmetry between in-degree and "
    "out-degree distributions can reveal facilities that are pure collection points (high in, "
    "low out) or pure dispatch centers (low in, high out)."
)
pdf.sub_title("Betweenness Distribution (purple)")
pdf.math_block(
    "  95th percentile line: p95 = metrics['betweenness'].quantile(0.95)\n\n"
    "  Nodes to the right of p95 are the top 5% most critical relay hubs.\n"
    "  The distribution is almost always heavily right-skewed (most nodes have near-zero\n"
    "  betweenness; a handful of hubs dominate the routing)."
)

# ═══════════════════════════════════════════════════════════════
# SECTION 15 -- Plot 5
# ═══════════════════════════════════════════════════════════════
pdf.add_page()
pdf.chapter_title("15.  Plot 5: Top-5 Hubs Subgraph")

pdf.code_block(
    "subgraph_nodes = set(top_hubs)\n"
    "for hub in top_hubs:\n"
    "    subgraph_nodes.update(G.predecessors(hub))\n"
    "    subgraph_nodes.update(G.successors(hub))\n"
    "SG = G.subgraph(subgraph_nodes).copy()"
)
pdf.body(
    "The full network plot with hundreds of nodes makes it hard to read individual "
    "connections. This plot zooms into the 1-hop neighbourhood of the top-5 bottleneck hubs:\n"
)
pdf.bullet("G.predecessors(hub) = all nodes with a direct edge INTO the hub (suppliers)")
pdf.bullet("G.successors(hub) = all nodes with a direct edge OUT of the hub (consumers)")
pdf.bullet("The resulting subgraph contains only the hub and its immediate neighbours")
pdf.ln(2)
pdf.body(
    "A fresh spring layout is computed for this smaller subgraph with k=3.0/sqrt(len(SG)) "
    "and more iterations=80, giving a clearer, less crowded layout than the full network."
)
pdf.section_title("Node Encoding")
pdf.bullet("Top-5 bottleneck hubs: large red circles (#d73027), size=1200")
pdf.bullet("Immediate neighbours: medium blue circles (#aec7e8), size=350")
pdf.bullet("All nodes get name labels (the smaller node count makes this readable)")
pdf.bullet("Delayed edges (weight > 1.2): red, width=2.0; normal edges: light blue, width=0.8")

# ═══════════════════════════════════════════════════════════════
# SECTION 16 -- Plot 6 -- Interactive HTML
# ═══════════════════════════════════════════════════════════════
pdf.add_page()
pdf.chapter_title("16.  Plot 6: Interactive pyvis HTML")

pdf.code_block(
    "net = Network(height='820px', width='100%', directed=True,\n"
    "              bgcolor='#1a1a2e', font_color='white', notebook=False)\n"
    "net.set_options('{ \"physics\": { \"barnesHut\": { ... } } }')"
)
pdf.body(
    "pyvis wraps vis.js, a JavaScript graph visualization library, into Python. "
    "The output is a standalone HTML file that renders an interactive, physics-animated "
    "graph in any browser. Users can zoom, pan, drag nodes, and hover for tooltips."
)

pdf.section_title("Barnes-Hut Physics Parameters")
pdf.math_block(
    "  gravitationalConstant: -3500\n"
    "    Repulsion strength between nodes. More negative = nodes spread further apart.\n\n"
    "  centralGravity: 0.25\n"
    "    Pulls all nodes toward the center to prevent the graph from flying apart.\n\n"
    "  springLength: 180  (pixels)\n"
    "    Rest length of edge springs. Longer = edges appear less crowded.\n\n"
    "  springConstant: 0.04\n"
    "    Spring stiffness. Lower = more elastic, nodes find gentler equilibrium.\n\n"
    "  minVelocity: 0.75\n"
    "    Simulation stops when all node velocities drop below this value."
)

pdf.section_title("Node Tooltip (title attribute)")
pdf.code_block(
    "title = (\n"
    "    f'<b>{name}</b><br>'\n"
    "    f'Betweenness: {b:.4f}<br>'\n"
    "    f'Bottleneck Score: {bs:.4f}<br>'\n"
    "    f'In-degree: {G.in_degree(node)} | Out-degree: {G.out_degree(node)}'\n"
    ")"
)
pdf.body(
    "The title string is rendered as an HTML tooltip on hover. It surfaces the four most "
    "operationally relevant metrics for each node without cluttering the visual."
)

pdf.section_title("Edge Colour Logic")
pdf.math_block(
    "  weight > 1.5   =>  '#ff4444'  (bright red)    severe delay\n"
    "  weight > 1.2   =>  '#ff9933'  (orange)         moderate delay\n"
    "  otherwise      =>  '#33cc55'  (green)           acceptable\n\n"
    "  edge width = 0.5 + 3.0 * min(max(weight - 0.5, 0), 2.0)\n\n"
    "  Clamps weight to [0.5, 2.5] before scaling, so extreme outliers don't\n"
    "  produce absurdly thick edges. Width range: [0.5, 3.5] pixels."
)

# ═══════════════════════════════════════════════════════════════
# SECTION 17 -- END-TO-END FLOW SUMMARY
# ═══════════════════════════════════════════════════════════════
pdf.add_page()
pdf.chapter_title("17.  End-to-End Data Flow Summary")

pdf.body(
    "Here is the complete data flow from raw CSV to visualizations:\n"
)
pdf.code_block(
    "delivery_data.csv\n"
    "    |\n"
    "    v  load_and_preprocess()\n"
    "    |  - Parse timestamps\n"
    "    |  - Engineer: hour, tod_bucket\n"
    "    |  - Normalise: is_cutoff\n"
    "    |\n"
    "    v  extract_leg_records()\n"
    "    |  - Filter to is_cutoff=False (departure scans only)\n"
    "    |  - One row per OD leg departure\n"
    "    |\n"
    "    v  build_edge_weight_table()\n"
    "    |  - Compute median delay factor at two granularities\n"
    "    |  - edge_stats  (src, dst, route_type, tod_bucket)\n"
    "    |  - edge_rt_agg (src, dst, route_type)\n"
    "    |  - rt_medians, global_median  (fallbacks)\n"
    "    |  => Save: edge_weight_table.csv, edge_rt_aggregated.csv\n"
    "    |\n"
    "    v  build_graph()\n"
    "    |  - Nodes: all facility IDs + name attribute\n"
    "    |  - Edges: weighted, stratified, with tod_lookup\n"
    "    |  => G (NetworkX DiGraph)\n"
    "    |\n"
    "    v  compute_graph_metrics()\n"
    "    |  - betweenness, degree, clustering\n"
    "    |  - avg_incoming_delay_factor\n"
    "    |  - bottleneck_score\n"
    "    |  => Save: node_metrics.csv\n"
    "    |\n"
    "    v  identify_delayed_corridors()\n"
    "    |  - Flatten edges to DataFrame\n"
    "    |  - Flag is_chronically_delayed\n"
    "    |  => Save: corridor_audit.csv\n"
    "    |\n"
    "    v  pickle.dump(G)\n"
    "    |  => Save: logistics_graph.pkl\n"
    "    |\n"
    "    v  [visualize_graph.py]\n"
    "    |  - Load: logistics_graph.pkl, node_metrics.csv, corridor_audit.csv\n"
    "    |  - compute_layout(): spring layout positions\n"
    "    |  => 01_full_network.png\n"
    "    |  => 02_bottleneck_hubs.png\n"
    "    |  => 03_delayed_corridors.png\n"
    "    |  => 04_degree_distributions.png\n"
    "    |  => 05_top_hubs_subgraph.png\n"
    "    |  => 06_interactive_network.html"
)

pdf.ln(4)
pdf.section_title("Key Formulas at a Glance")
pdf.math_block(
    "  delay_factor      = actual_time / osrm_time\n\n"
    "  edge weight       = weighted_average(median_factors, weights=trip_counts)\n"
    "                    = np.average(median_factors, weights=trip_counts)\n\n"
    "  pct_delayed       = count(factor > 1.2) / total_trips\n\n"
    "  betweenness(v)    = SUM_st [ sigma(s,t|v) / sigma(s,t) ]  / (n-1)(n-2)\n\n"
    "  avg_in_delay(v)   = MEAN of weight(p->v)  for all predecessors p\n\n"
    "  bottleneck_score  = betweenness(v) * avg_in_delay(v)\n\n"
    "  spring layout k   = 2.5 / sqrt(N)\n\n"
    "  node_size_plot1   = 80 + 2500 * (betweenness / max_betweenness)\n"
    "  hub_size_plot2    = 800 + 4000 * (betweenness / max_betweenness)\n"
    "  edge_width_plot1  = 0.3 + 1.5 * ((weight - w_min) / (w_max - w_min))"
)

# ─── Save ────────────────────────────────────────────────────
pdf.output(OUTPUT_PATH)
print(f"PDF saved: {OUTPUT_PATH}")
