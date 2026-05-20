import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import os
from scipy.interpolate import CloughTocher2DInterpolator, LinearNDInterpolator

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── Constants ───
BASE_RC = 0.34
BASE_COVERAGE = 1.25
BASE_PREMIUM = 0.0125
BASE_DISCOUNT = 0.06
BASE_ANNUITY = sum(1 / (1 + 0.06) ** y for y in range(1, 41))
ANGLES = [52, 60, 70, 77]
ANGLE_COLORS = {
    52: [59, 130, 246], 60: [239, 68, 68],
    70: [234, 179, 8],  77: [34, 197, 94],
}
ANGLE_LABELS = {52: '52°', 60: '60°', 70: '70°', 77: '77°'}

# Demand data files
ORENNIA_FILE = 'orennia_market_demand_05.18.26_v2.csv'
WOODMAC_FILE = 'woodmac_demand_05.18.26.csv'


def annuity_factor(r, n=40):
    if r <= 0:
        return float(n)
    return sum(1 / (1 + r) ** y for y in range(1, n + 1))


# ─── Page Config ───
st.set_page_config(page_title="Hail Risk Sensitivity Tool v7", page_icon="🌨️",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=JetBrains+Mono:wght@400;500&display=swap');
    .stApp { font-family: 'DM Sans', sans-serif; }
    .main-title { font-size: 2rem; font-weight: 700; color: #1a1a2e; margin-bottom: 0; letter-spacing: -0.5px; }
    .subtitle { font-size: 1rem; color: #6b7280; margin-top: 0; margin-bottom: 1.5rem; }
    section[data-testid="stSidebar"] { background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%); color: #f1f5f9; }
    section[data-testid="stSidebar"] .stMarkdown h3 { color: #f8fafc; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 1.5px; margin-top: 1.5rem; font-weight: 700; }
    section[data-testid="stSidebar"] label { color: #f1f5f9 !important; font-weight: 500; }
    section[data-testid="stSidebar"] .stMarkdown p { color: #e2e8f0 !important; }
    section[data-testid="stSidebar"] .stRadio label, section[data-testid="stSidebar"] .stCheckbox label { color: #f8fafc !important; }
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color: #cbd5e1 !important; }
    section[data-testid="stSidebar"] .stNumberInput label, section[data-testid="stSidebar"] .stSlider label { color: #f1f5f9 !important; }
    .metric-card { background: linear-gradient(135deg, #f8fafc, #f1f5f9); border-radius: 12px; padding: 1.2rem; border: 1px solid #e2e8f0; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }
    .metric-card h4 { margin: 0; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 1px; color: #64748b; }
    .metric-card .value { font-family: 'JetBrains Mono', monospace; font-size: 1.5rem; font-weight: 700; margin: 0.3rem 0; }
    .metric-card .unit { font-size: 0.75rem; color: #94a3b8; }
    .winner-banner { background: linear-gradient(135deg, #059669, #10b981); color: white; padding: 1rem 1.5rem; border-radius: 12px; text-align: center; font-size: 1.1rem; font-weight: 600; margin: 1rem 0; }
    .lookup-result { background: linear-gradient(135deg, #1e293b, #334155); color: #e2e8f0; padding: 1.5rem; border-radius: 12px; margin: 1rem 0; }
    .lookup-result h3 { color: #38bdf8; margin-top: 0; }
</style>
""", unsafe_allow_html=True)


# ─── Data Loading ───
@st.cache_data
def load_data(suffix):
    return pd.read_csv(os.path.join(SCRIPT_DIR, f'hail_data_{suffix}.csv'))

@st.cache_data
def load_demand_data(source):
    """Load demand CSV for the selected source."""
    filename = ORENNIA_FILE if source == 'Orennia' else WOODMAC_FILE
    path = os.path.join(SCRIPT_DIR, filename)
    if os.path.exists(path):
        df = pd.read_csv(path)
        required = ['hail_lat', 'hail_lon', 'Year', 'DC Capacity (MW)']
        if all(col in df.columns for col in required):
            return df
        st.sidebar.warning(f"⚠️ {filename} missing required columns. Using uniform distribution.")
    return None

@st.cache_data
def derive_market_defaults(demand_df):
    """Derive default market sizes (GWdc) per year from the demand data."""
    if demand_df is None:
        return {2026: 36, 2027: 44, 2028: 50, 2029: 55, 2030: 60, 2031: 65, 2032: 70}
    yearly = demand_df.groupby('Year')['DC Capacity (MW)'].sum() / 1000
    counts = demand_df.groupby('Year').size()
    valid_years = counts[counts >= 20].index
    return {int(yr): round(yearly.get(yr, 0), 1) for yr in sorted(valid_years)}


def compute_costs(df, replacement_cost, coverage_ratio, annual_premium,
                  interest_rate, risk_pct, capex_dict, ins_on, risk_on, capex_on,
                  active_angles=None):
    if active_angles is None:
        active_angles = ANGLES
    new_annuity = annuity_factor(interest_rate / 100.0)
    ins_scale = ((replacement_cost / BASE_RC) * (coverage_ratio / 125.0)
                 * (annual_premium / 1.25) * (new_annuity / BASE_ANNUITY))
    risk_scale = ((replacement_cost / BASE_RC) * (risk_pct / 100.0)
                  * (new_annuity / BASE_ANNUITY))

    computed = df[['lat', 'lon']].copy()
    for angle in ANGLES:
        ins_val = df[f'ins_{angle}'] * ins_scale if ins_on else 0.0
        risk_val = df[f'risk_{angle}'] * risk_scale if risk_on else 0.0
        capex_val = capex_dict[angle] if capex_on else 0.0
        computed[f'total_{angle}'] = ins_val + risk_val + capex_val
        computed[f'ins_{angle}'] = ins_val if ins_on else 0.0
        computed[f'risk_{angle}'] = risk_val if risk_on else 0.0
        computed[f'capex_{angle}'] = capex_val if capex_on else 0.0

    # Best angle selection only among active angles
    active_cols = [f'total_{a}' for a in active_angles]
    computed['best_angle'] = computed[active_cols].idxmin(axis=1).str.replace('total_', '').astype(int)
    computed['best_cost'] = computed[active_cols].min(axis=1)
    return computed


def compute_blended(df20, df32, pct_20, replacement_cost, coverage_ratio, annual_premium,
                    interest_rate, risk_pct, capex_dict, ins_on, risk_on, capex_on,
                    active_angles=None):
    """Pro-rata blend of 2.0mm and 3.2mm glass results.

    For each angle: blended_total = (pct_20/100) * total_20 + (1-pct_20/100) * total_32
    Best angle is then selected from blended totals.
    """
    if active_angles is None:
        active_angles = ANGLES
    c20 = compute_costs(df20, replacement_cost, coverage_ratio, annual_premium,
                        interest_rate, risk_pct, capex_dict, ins_on, risk_on, capex_on, active_angles)
    c32 = compute_costs(df32, replacement_cost, coverage_ratio, annual_premium,
                        interest_rate, risk_pct, capex_dict, ins_on, risk_on, capex_on, active_angles)

    # Merge on lat/lon
    blended = c20[['lat', 'lon']].copy()
    w20 = pct_20 / 100.0
    w32 = 1 - w20
    for col_prefix in ['total_', 'ins_', 'risk_', 'capex_']:
        for angle in ANGLES:
            col = f'{col_prefix}{angle}'
            blended[col] = w20 * c20[col].values + w32 * c32[col].values

    active_cols = [f'total_{a}' for a in active_angles]
    blended['best_angle'] = blended[active_cols].idxmin(axis=1).str.replace('total_', '').astype(int)
    blended['best_cost'] = blended[active_cols].min(axis=1)
    return blended


def compute_luce_probabilities(computed, sigma, active_angles):
    """Compute Luce/Logit probability of each tilt angle winning at each location.

    Uses per-location cost normalization: subtract min cost across active angles so
    the cheapest angle has utility 0. Then P_a = exp(-sigma * C_a_norm) / sum_j exp(-sigma * C_j_norm).
    """
    if len(active_angles) == 0:
        return computed.copy()

    cost_cols = [f'total_{a}' for a in active_angles]
    costs = computed[cost_cols].values  # shape (n_locations, n_active)

    # Per-location normalization: subtract min so cheapest has cost 0
    min_costs = costs.min(axis=1, keepdims=True)
    costs_norm = costs - min_costs  # shape (n_locations, n_active)

    # Softmax with negative utility: lower cost -> higher probability
    exponents = -sigma * costs_norm  # negative because lower cost = higher utility
    # Numerical stability: subtract max (but since min cost is 0, max exponent is 0 -> already stable)
    exp_vals = np.exp(exponents)
    probs = exp_vals / exp_vals.sum(axis=1, keepdims=True)

    result = computed.copy()
    for i, angle in enumerate(active_angles):
        result[f'prob_{angle}'] = probs[:, i]
    # For angles not in active set, probability = 0
    for angle in ANGLES:
        if angle not in active_angles:
            result[f'prob_{angle}'] = 0.0

    return result


def interpolate_best_product(lats, lons, best_angles):
    """Create a dense grid of best-product assignments, clipped to data footprint."""
    from scipy.interpolate import NearestNDInterpolator
    from scipy.spatial import Delaunay, cKDTree

    points = np.column_stack([lats, lons])
    interp = NearestNDInterpolator(points, best_angles)
    hull = Delaunay(points)
    tree = cKDTree(points)

    lat_range = np.arange(lats.min() - 0.25, lats.max() + 0.25, 0.25)
    lon_range = np.arange(lons.min() - 0.25, lons.max() + 0.25, 0.25)
    grid_lat, grid_lon = np.meshgrid(lat_range, lon_range)
    grid_pts = np.column_stack([grid_lat.ravel(), grid_lon.ravel()])

    # Clip: only points inside convex hull AND within 1.5° of a data point
    inside = hull.find_simplex(grid_pts) >= 0
    dists, _ = tree.query(grid_pts)
    mask = inside & (dists < 1.5)

    clipped_pts = grid_pts[mask]
    grid_vals = interp(clipped_pts[:, 0], clipped_pts[:, 1])

    grid_df = pd.DataFrame({
        'lat': clipped_pts[:, 0], 'lon': clipped_pts[:, 1],
        'best_angle': grid_vals.astype(int),
    })
    grid_df['color'] = grid_df['best_angle'].map(ANGLE_COLORS)
    grid_df['best_angle_display'] = grid_df['best_angle'].astype(str) + '°'
    return grid_df

# US States GeoJSON URL (loaded at runtime by pydeck from public CDN)
US_STATES_URL = "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json"


# ─── Sidebar ───
st.sidebar.markdown("## 🌨️ Hail Sensitivity Tool")

st.sidebar.markdown("### Tilt Angles in Analysis")
st.sidebar.caption("Toggle which stow angles are considered.")
ang_cols = st.sidebar.columns(4)
angle_enabled = {}
defaults = {52: True, 60: True, 70: True, 77: True}
for i, a in enumerate(ANGLES):
    with ang_cols[i]:
        angle_enabled[a] = st.checkbox(f"{a}°", value=defaults[a], key=f"ang_{a}")
ACTIVE_ANGLES = [a for a in ANGLES if angle_enabled[a]]
if not ACTIVE_ANGLES:
    st.sidebar.error("Select at least one tilt angle.")
    ACTIVE_ANGLES = [52]

st.sidebar.markdown("### Glass Type")
glass_choice = st.sidebar.radio("Module glass thickness",
                                 ["3.2 mm", "2.0 mm", "Blended", "Compare Both"],
                                 index=0, horizontal=True)
if glass_choice == "Blended":
    glass_pct_20 = st.sidebar.slider("2.0 mm Share (%)", 0, 100, 50, 5,
                                      help="Pro-rata mix of 2.0mm and 3.2mm glass. Default 50/50.")
else:
    glass_pct_20 = 50  # unused

st.sidebar.markdown("### Economics")
replacement_cost = st.sidebar.slider("Module Replacement Cost ($/W)", 0.20, 0.60, 0.34, 0.01,
                                      help="Base: $0.34/W")
interest_rate = st.sidebar.slider("Discount Rate (%)", 0.0, 15.0, 6.0, 0.25,
                                   help="Base: 6.0% → 15.05× annuity over 40 years")

st.sidebar.markdown("### Insurance")
coverage_ratio = st.sidebar.slider("Coverage Ratio (%)", 100, 150, 125, 5,
                                    help="Multiplier on PML for insured value. Base: 125%")
annual_premium = st.sidebar.slider("Annual Premium Rate (%)", 0.25, 5.0, 1.25, 0.05,
                                    help="Annual insurance premium as % of insured value. Base: 1.25%")

st.sidebar.markdown("### Developer Risk")
risk_pct = st.sidebar.slider("Risk Considered (%)", 0, 100, 100, 5,
                              help="% of below-deductible risk to price in. Base: 100%")

st.sidebar.markdown("### CapEx Premium (¢/W)")
capex_52 = st.sidebar.slider("52° CapEx", 0.0, 10.0, 0.0, 0.1)
capex_60 = st.sidebar.slider("60° CapEx", 0.0, 10.0, 0.0, 0.1)
capex_70 = st.sidebar.slider("70° CapEx", 0.0, 10.0, 0.4, 0.1)
capex_77 = st.sidebar.slider("77° CapEx", 0.0, 10.0, 1.2, 0.1)
capex_dict = {52: capex_52, 60: capex_60, 70: capex_70, 77: capex_77}

st.sidebar.markdown("### Cost Layers")
ins_on = st.sidebar.checkbox("Insurance", value=True)
risk_on = st.sidebar.checkbox("Developer Risk", value=True)
capex_on = st.sidebar.checkbox("CapEx Premium", value=True)

st.sidebar.markdown("### Choice Model (Luce / Logit)")
sigma = st.sidebar.slider(
    "Sigma (cost sensitivity)", 0.0, 5.0, 2.5, 0.1,
    help="0 = no sensitivity (all equally likely). Higher = more decisive. Default 2.5."
)

st.sidebar.markdown("### Demand Shape (geographic distribution)")
demand_shape = st.sidebar.radio(
    "Where is demand located",
    ["Orennia", "Wood Mackenzie"],
    index=0, horizontal=True, key="shape_src",
    help="Orennia = project-level pipeline locations. WoodMac = state-level evenly distributed."
)

st.sidebar.markdown("### Demand Magnitude (total GW)")
magnitude_source = st.sidebar.radio(
    "Total GW per year from",
    ["Orennia", "Wood Mackenzie", "Manual"],
    index=0, horizontal=True, key="mag_src",
    help="Defaults from the chosen forecast. 'Manual' lets you set everything yourself."
)

# Load both demand sources
_shape_df = load_demand_data(demand_shape)
_orennia_df_full = load_demand_data("Orennia")
_woodmac_df_full = load_demand_data("Wood Mackenzie")

# Derive defaults from the magnitude source (or use Orennia's if Manual)
if magnitude_source == "Orennia":
    _market_defaults = derive_market_defaults(_orennia_df_full)
elif magnitude_source == "Wood Mackenzie":
    _market_defaults = derive_market_defaults(_woodmac_df_full)
else:  # Manual
    _market_defaults = {2026: 36, 2027: 44, 2028: 50, 2029: 55, 2030: 60, 2031: 65, 2032: 70}

st.sidebar.markdown("### Market Size by Year (GWdc)")
st.sidebar.caption(f"Shape: **{demand_shape}**  |  Magnitude: **{magnitude_source}**")

market_sizes = {}
for yr, default_gw in _market_defaults.items():
    market_sizes[yr] = st.sidebar.number_input(
        f"{yr} (data: {default_gw:.1f})", min_value=0.0, max_value=500.0,
        value=float(default_gw), step=1.0, key=f"mkt_{yr}_{magnitude_source}")

layers_active = []
if ins_on: layers_active.append("Ins")
if risk_on: layers_active.append("Risk")
if capex_on: layers_active.append("CapEx")
layers_str = " + ".join(layers_active) if layers_active else "None"


# ═══════════════════════════════════════════
# RENDERING FUNCTIONS
# ═══════════════════════════════════════════

def render_single(df, label="", precomputed=None):
    if precomputed is not None:
        computed = precomputed
    else:
        computed = compute_costs(df, replacement_cost, coverage_ratio, annual_premium,
                                 interest_rate, risk_pct, capex_dict, ins_on, risk_on, capex_on,
                                 active_angles=ACTIVE_ANGLES)

    if label:
        st.markdown(f"#### {label}")

    # ─── Metric Cards (only active angles) ───
    cols = st.columns(max(1, len(ACTIVE_ANGLES)))
    for i, angle in enumerate(ACTIVE_ANGLES):
        avg = computed[f'total_{angle}'].mean()
        c = ANGLE_COLORS[angle]
        with cols[i]:
            st.markdown(f'<div class="metric-card"><h4>{angle}° Avg Total</h4>'
                        f'<div class="value" style="color:rgb({c[0]},{c[1]},{c[2]})">{avg:.3f}</div>'
                        f'<div class="unit">¢/W</div></div>', unsafe_allow_html=True)

    win_counts = computed['best_angle'].value_counts()
    overall_winner = win_counts.idxmax()
    win_pct = win_counts.max() / len(computed) * 100
    angles_str = ", ".join([f"{a}°" for a in ACTIVE_ANGLES])
    st.markdown(f'<div class="winner-banner">🏆 {overall_winner}° wins at {win_counts.max()} of '
                f'{len(computed)} locations ({win_pct:.1f}%) — Layers: {layers_str} — Angles: {angles_str}</div>',
                unsafe_allow_html=True)

    # ─── 3D Column Map (only active angles) ───
    st.markdown("#### 🗺️ Total Cost by Angle (3D Columns)")
    map_rows = []
    for _, row in computed.iterrows():
        for angle in ACTIVE_ANGLES:
            val = row[f'total_{angle}']
            if val > 0:
                map_rows.append({
                    'lat': row['lat'],
                    'lon': row['lon'] + (ACTIVE_ANGLES.index(angle) - (len(ACTIVE_ANGLES)-1)/2) * 0.15,
                    'elevation': val * 50000,
                    'color': ANGLE_COLORS[angle],
                    'angle': angle,
                    'cost_display': f"{val:.3f}",
                })
    map_df = pd.DataFrame(map_rows)
    if len(map_df) > 0:
        col_layer = pdk.Layer("ColumnLayer", data=map_df, get_position='[lon, lat]',
                              get_elevation='elevation', elevation_scale=1, radius=12000,
                              get_fill_color='color', pickable=True, auto_highlight=True)
        states_3d = pdk.Layer("GeoJsonLayer", data=US_STATES_URL,
                              stroked=True, filled=False, pickable=False,
                              get_line_color=[100, 100, 100, 140], line_width_min_pixels=1)
        st.pydeck_chart(pdk.Deck(
            layers=[states_3d, col_layer],
            initial_view_state=pdk.ViewState(latitude=39.0, longitude=-98.0, zoom=3.8, pitch=45),
            map_style="light",
            tooltip={"html": "<b>{angle}°</b>: {cost_display} ¢/W",
                     "style": {"backgroundColor": "#1e293b", "color": "#e2e8f0",
                                "fontSize": "13px", "padding": "8px 12px", "borderRadius": "8px"}},
        ), use_container_width=True, height=500)

    # ─── Interpolated Optimal Angle Map ───
    st.markdown("#### 🎯 Optimal Angle by Location")
    computed['color'] = computed['best_angle'].map(ANGLE_COLORS)
    computed['best_cost_display'] = computed['best_cost'].round(3).astype(str)

    grid_df = interpolate_best_product(computed['lat'].values, computed['lon'].values,
                                        computed['best_angle'].values)

    # Background: interpolated fill (not pickable — tooltip comes from foreground only)
    bg_layer = pdk.Layer("ScatterplotLayer", data=grid_df, get_position='[lon, lat]',
                         get_fill_color='color', get_radius=15000, pickable=False,
                         opacity=0.65)

    # Foreground: actual data points (pickable for tooltip)
    fg_layer = pdk.Layer("ScatterplotLayer", data=computed, get_position='[lon, lat]',
                         get_fill_color='color', get_radius=8000, pickable=True,
                         auto_highlight=True, opacity=1.0)

    # State borders
    states_layer = pdk.Layer(
        "GeoJsonLayer", data=US_STATES_URL,
        stroked=True, filled=False, pickable=False,
        get_line_color=[100, 100, 100, 180], line_width_min_pixels=1,
    )

    st.pydeck_chart(pdk.Deck(
        layers=[bg_layer, states_layer, fg_layer],
        initial_view_state=pdk.ViewState(latitude=39.0, longitude=-98.0, zoom=3.8, pitch=0),
        map_style="light",
        tooltip={"html": "<b>Best:</b> {best_angle}° — {best_cost_display} ¢/W",
                 "style": {"backgroundColor": "#1e293b", "color": "#e2e8f0",
                            "fontSize": "13px", "padding": "8px 12px", "borderRadius": "8px"}},
    ), use_container_width=True, height=500)

    # ─── Win Count Table (active angles only) ───
    st.markdown("#### 📊 Win Counts")
    wc = pd.DataFrame({
        'Angle': [f'{a}°' for a in ACTIVE_ANGLES],
        'Wins': [win_counts.get(a, 0) for a in ACTIVE_ANGLES],
        'Win %': [f"{win_counts.get(a, 0)/len(computed)*100:.1f}%" for a in ACTIVE_ANGLES],
        'Avg Total (¢/W)': [f"{computed[f'total_{a}'].mean():.4f}" for a in ACTIVE_ANGLES],
    })
    st.dataframe(wc, hide_index=True, use_container_width=True)

    return computed


# ─── Market Share & Demand Analysis ───
def render_market_share(computed, shape_df, label=""):
    """Market share by tilt angle with multi-year select, shape/magnitude split, and export."""
    st.markdown("---")
    st.markdown("### 📈 Market Share by Tilt Angle"
                + (f" — {label}" if label else "")
                + f"  [shape: {demand_shape}, magnitude: {magnitude_source}]")

    has_shape = shape_df is not None
    years = sorted(market_sizes.keys())

    # ─── Multi-year selector ───
    st.markdown("**Years to include** (select one or more):")
    yr_cols = st.columns(min(len(years), 8))
    year_selected = {}
    for i, yr in enumerate(years):
        with yr_cols[i % len(yr_cols)]:
            year_selected[yr] = st.checkbox(str(yr), value=True, key=f"yrsel_{yr}_{label}")
    selected_years = [yr for yr in years if year_selected[yr]]

    if not selected_years:
        st.warning("Select at least one year.")
        return

    total_gw = sum(market_sizes[yr] for yr in selected_years)
    total_mw = total_gw * 1000
    yr_label = (f"{min(selected_years)}–{max(selected_years)}"
                if len(selected_years) > 1 else str(selected_years[0]))

    # ─── Build location-level scaled demand from shape source ───
    merged = None
    use_shape = False
    if has_shape:
        shape_sub = shape_df[shape_df['Year'].isin(selected_years)].copy()
        if len(shape_sub) > 0:
            demand_by_loc = shape_sub.groupby(['hail_lat', 'hail_lon']).agg(
                total_mw=('DC Capacity (MW)', 'sum')).reset_index()
            merged = demand_by_loc.merge(
                computed[['lat', 'lon', 'best_angle']],
                left_on=['hail_lat', 'hail_lon'], right_on=['lat', 'lon'], how='inner')
            if len(merged) > 0:
                shape_total = merged['total_mw'].sum()
                scale = total_mw / shape_total if shape_total > 0 else 1.0
                merged['scaled_mw'] = merged['total_mw'] * scale
                angle_summary = merged.groupby('best_angle')['scaled_mw'].sum().reset_index()
                angle_summary.columns = ['Best Product', 'MWdc']
                data_source = f"Shape: {demand_shape} (scaled to {total_gw:.1f} GW)"
                use_shape = True

    if not use_shape:
        merged = None
        n_locations = len(computed)
        mw_per_loc = total_mw / n_locations if n_locations > 0 else 0
        angle_summary = computed.groupby('best_angle').size().reset_index(name='count')
        angle_summary['MWdc'] = angle_summary['count'] * mw_per_loc
        angle_summary = angle_summary[['best_angle', 'MWdc']]
        angle_summary.columns = ['Best Product', 'MWdc']
        data_source = "Uniform distribution"

    # Ensure all active angles appear
    for a in ACTIVE_ANGLES:
        if a not in angle_summary['Best Product'].values:
            angle_summary = pd.concat([angle_summary,
                                       pd.DataFrame({'Best Product': [a], 'MWdc': [0]})],
                                      ignore_index=True)
    angle_summary = angle_summary[angle_summary['Best Product'].isin(ACTIVE_ANGLES)].copy()
    angle_summary = angle_summary.sort_values('Best Product')
    angle_summary['GWdc'] = (angle_summary['MWdc'] / 1000).round(2)
    total_for_share = angle_summary['MWdc'].sum()
    angle_summary['Share (%)'] = (angle_summary['MWdc'] / total_for_share * 100).round(1) if total_for_share > 0 else 0
    angle_summary['Best Product'] = angle_summary['Best Product'].astype(str) + '°'

    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown(f"**{yr_label}**  —  Total Market: **{total_gw:.1f} GWdc**")
        st.caption(f"Source: {data_source}")
        display_df = angle_summary[['Best Product', 'GWdc', 'MWdc', 'Share (%)']].copy()
        display_df['MWdc'] = display_df['MWdc'].round(0).astype(int)
        st.dataframe(display_df, hide_index=True, use_container_width=True)

    with c2:
        chart_df = angle_summary.set_index('Best Product')[['GWdc']]
        st.bar_chart(chart_df, use_container_width=True, height=300)

    # ─── Export button ───
    export_df = build_export(display_df, selected_years, total_gw, label)
    csv_bytes = export_df.to_csv(index=False).encode()
    st.download_button(
        "📥 Export Summary CSV", csv_bytes,
        file_name=f"hail_summary_{label.replace(' ', '_')}_{yr_label}.csv",
        mime="text/csv", key=f"export_{label}",
    )

    # ─── 3D Demand Map ───
    st.markdown("#### 🏗️ Demand by Location (MWdc)")
    if merged is not None and len(merged) > 0:
        map_demand = merged[['lat', 'lon', 'best_angle', 'scaled_mw']].copy()
        map_demand.rename(columns={'scaled_mw': 'mw'}, inplace=True)
    else:
        map_demand = computed[['lat', 'lon', 'best_angle']].copy()
        n_locs = len(map_demand)
        map_demand['mw'] = total_mw / n_locs if n_locs > 0 else 0

    map_demand = map_demand[(map_demand['mw'] > 0) &
                             (map_demand['best_angle'].isin(ACTIVE_ANGLES))].copy()

    if len(map_demand) > 0:
        map_demand['color'] = map_demand['best_angle'].map(ANGLE_COLORS)
        map_demand['elevation'] = map_demand['mw'] * 200
        map_demand['mw_display'] = map_demand['mw'].round(0).astype(int).astype(str)
        map_demand['angle_display'] = map_demand['best_angle'].astype(str) + '°'

        demand_col_layer = pdk.Layer(
            "ColumnLayer", data=map_demand, get_position='[lon, lat]',
            get_elevation='elevation', elevation_scale=1, radius=18000,
            get_fill_color='color', pickable=True, auto_highlight=True,
        )
        states_demand = pdk.Layer("GeoJsonLayer", data=US_STATES_URL,
                                  stroked=True, filled=False, pickable=False,
                                  get_line_color=[100, 100, 100, 140], line_width_min_pixels=1)
        st.pydeck_chart(pdk.Deck(
            layers=[states_demand, demand_col_layer],
            initial_view_state=pdk.ViewState(latitude=39.0, longitude=-98.0, zoom=3.8, pitch=45),
            map_style="light",
            tooltip={"html": "<b>{angle_display}</b> — {mw_display} MWdc",
                     "style": {"backgroundColor": "#1e293b", "color": "#e2e8f0",
                                "fontSize": "13px", "padding": "8px 12px", "borderRadius": "8px"}},
        ), use_container_width=True, height=500)
    else:
        st.info("No demand data to display for the selected years.")

    # ─── Year-by-Year Breakdown ───
    if len(selected_years) > 1:
        st.markdown("#### Year-by-Year Breakdown")
        yoy_rows = []
        for yr in selected_years:
            yr_mw = market_sizes[yr] * 1000
            if has_shape:
                yr_shape = shape_df[shape_df['Year'] == yr]
                if len(yr_shape) > 0:
                    dbl = yr_shape.groupby(['hail_lat', 'hail_lon']).agg(
                        total_mw=('DC Capacity (MW)', 'sum')).reset_index()
                    mrg = dbl.merge(computed[['lat', 'lon', 'best_angle']],
                                    left_on=['hail_lat', 'hail_lon'],
                                    right_on=['lat', 'lon'], how='inner')
                    if len(mrg) > 0:
                        sc = yr_mw / mrg['total_mw'].sum() if mrg['total_mw'].sum() > 0 else 1
                        mrg['scaled_mw'] = mrg['total_mw'] * sc
                        for a in ACTIVE_ANGLES:
                            sub = mrg[mrg['best_angle'] == a]
                            yoy_rows.append({'Year': yr, 'Angle': f'{a}°',
                                             'GWdc': round(sub['scaled_mw'].sum() / 1000, 2)})
                        continue
            # Uniform fallback for this year
            counts = computed['best_angle'].value_counts()
            n = len(computed)
            mw_per = yr_mw / n if n > 0 else 0
            for a in ACTIVE_ANGLES:
                cnt = counts.get(a, 0)
                yoy_rows.append({'Year': yr, 'Angle': f'{a}°',
                                 'GWdc': round(cnt * mw_per / 1000, 2)})
        if yoy_rows:
            yoy_df = pd.DataFrame(yoy_rows)
            pivot = yoy_df.pivot_table(index='Angle', columns='Year', values='GWdc',
                                       aggfunc='sum').fillna(0)
            st.dataframe(pivot, use_container_width=True)


def build_export(market_table, selected_years, total_gw, label):
    """Build a single CSV-ready export of inputs and outputs."""
    rows = []
    rows.append({'Section': 'Inputs', 'Key': 'Glass Type', 'Value': glass_choice})
    if glass_choice == "Blended":
        rows.append({'Section': 'Inputs', 'Key': '2.0 mm Share (%)', 'Value': glass_pct_20})
    rows.append({'Section': 'Inputs', 'Key': 'Tilt Angles Active',
                 'Value': ', '.join([f'{a}°' for a in ACTIVE_ANGLES])})
    rows.append({'Section': 'Inputs', 'Key': 'Replacement Cost ($/W)', 'Value': replacement_cost})
    rows.append({'Section': 'Inputs', 'Key': 'Discount Rate (%)', 'Value': interest_rate})
    rows.append({'Section': 'Inputs', 'Key': 'Coverage Ratio (%)', 'Value': coverage_ratio})
    rows.append({'Section': 'Inputs', 'Key': 'Annual Premium (%)', 'Value': annual_premium})
    rows.append({'Section': 'Inputs', 'Key': 'Dev Risk Considered (%)', 'Value': risk_pct})
    rows.append({'Section': 'Inputs', 'Key': '52° CapEx (¢/W)', 'Value': capex_52})
    rows.append({'Section': 'Inputs', 'Key': '60° CapEx (¢/W)', 'Value': capex_60})
    rows.append({'Section': 'Inputs', 'Key': '70° CapEx (¢/W)', 'Value': capex_70})
    rows.append({'Section': 'Inputs', 'Key': '77° CapEx (¢/W)', 'Value': capex_77})
    rows.append({'Section': 'Inputs', 'Key': 'Insurance Layer',
                 'Value': 'On' if ins_on else 'Off'})
    rows.append({'Section': 'Inputs', 'Key': 'Dev Risk Layer',
                 'Value': 'On' if risk_on else 'Off'})
    rows.append({'Section': 'Inputs', 'Key': 'CapEx Layer',
                 'Value': 'On' if capex_on else 'Off'})
    rows.append({'Section': 'Inputs', 'Key': 'Demand Shape', 'Value': demand_shape})
    rows.append({'Section': 'Inputs', 'Key': 'Demand Magnitude Source', 'Value': magnitude_source})
    rows.append({'Section': 'Inputs', 'Key': 'Sigma (Luce sensitivity)', 'Value': sigma})
    rows.append({'Section': 'Inputs', 'Key': 'Years Selected',
                 'Value': ', '.join(str(y) for y in selected_years)})
    rows.append({'Section': 'Inputs', 'Key': 'Total Market (GWdc)', 'Value': round(total_gw, 2)})
    for yr in selected_years:
        rows.append({'Section': 'Market Size', 'Key': f'{yr} (GWdc)',
                     'Value': market_sizes[yr]})

    for _, mr in market_table.iterrows():
        rows.append({'Section': 'Market Share',
                     'Key': f"{mr['Best Product']} GWdc", 'Value': mr['GWdc']})
        rows.append({'Section': 'Market Share',
                     'Key': f"{mr['Best Product']} MWdc", 'Value': mr['MWdc']})
        rows.append({'Section': 'Market Share',
                     'Key': f"{mr['Best Product']} Share (%)", 'Value': mr['Share (%)']})

    return pd.DataFrame(rows)


# ─── Value Gap Analysis (portfolio baseline) ───
def render_value_gap(computed, label=""):
    st.markdown("---")
    st.markdown("### 💰 Value Gap Analysis" + (f" — {label}" if label else ""))
    st.markdown("The **base case** is the full set of active tilt angles (always picks cheapest). "
                "Selecting fewer products shows the added cost of limiting your portfolio.")

    gap_cols = st.columns(max(1, len(ACTIVE_ANGLES)))
    selected = []
    for i, angle in enumerate(ACTIVE_ANGLES):
        with gap_cols[i]:
            if st.checkbox(f"{angle}°", value=True, key=f"vg_{angle}_{label}"):
                selected.append(angle)

    if len(selected) == 0:
        st.warning("Select at least one product angle.")
        return

    # Base case: full active portfolio
    computed['base_cost'] = computed[[f'total_{a}' for a in ACTIVE_ANGLES]].min(axis=1)
    # Subset
    computed['subset_cost'] = computed[[f'total_{a}' for a in selected]].min(axis=1)
    computed['gap'] = computed['subset_cost'] - computed['base_cost']

    max_gap = computed['gap'].quantile(0.95) if computed['gap'].max() > 0 else 1.0
    computed['gap_norm'] = (computed['gap'] / max_gap).clip(0, 1) if max_gap > 0 else 0.0
    computed['gap_color'] = computed['gap_norm'].apply(
        lambda g: [int(239 * (1 - g) + 34 * g), int(68 * (1 - g) + 197 * g),
                    int(68 * (1 - g) + 94 * g)])
    computed['gap_display'] = computed['gap'].round(3).astype(str)
    computed['base_display'] = computed['base_cost'].round(3).astype(str)
    computed['subset_display'] = computed['subset_cost'].round(3).astype(str)

    states_layer = pdk.Layer("GeoJsonLayer", data=US_STATES_URL,
                             stroked=True, filled=False, pickable=False,
                             get_line_color=[100, 100, 100, 180], line_width_min_pixels=1)

    st.pydeck_chart(pdk.Deck(
        layers=[pdk.Layer("ScatterplotLayer", data=computed, get_position='[lon, lat]',
                          get_fill_color='gap_color', get_radius=30000,
                          pickable=True, auto_highlight=True),
                states_layer],
        initial_view_state=pdk.ViewState(latitude=39.0, longitude=-98.0, zoom=3.8, pitch=0),
        map_style="light",
        tooltip={"html": ("<b>Full portfolio:</b> {base_display} ¢/W<br>"
                           "<b>Subset:</b> {subset_display} ¢/W<br>"
                           "<b>Added cost:</b> {gap_display} ¢/W"),
                 "style": {"backgroundColor": "#1e293b", "color": "#e2e8f0",
                            "fontSize": "13px", "padding": "8px 12px", "borderRadius": "8px"}},
    ), use_container_width=True, height=500)

    st.markdown(f"**Avg added cost:** {computed['gap'].mean():.4f} ¢/W  |  "
                f"**Median:** {computed['gap'].median():.4f} ¢/W  |  "
                f"**Max:** {computed['gap'].max():.4f} ¢/W")
    st.markdown("🟢 Green = large gap (clear full-portfolio advantage) → 🔴 Red = small gap (subset nearly as good)")


# ─── Site Lookup ───
def render_lookup(df, computed, label=""):
    st.markdown("---")
    st.markdown("### 📍 Site Lookup" + (f" — {label}" if label else ""))

    lk1, lk2 = st.columns(2)
    with lk1:
        input_lat = st.number_input("Latitude", 24.0, 50.0, 33.0, 0.1, key=f"lat_{label}")
    with lk2:
        input_lon = st.number_input("Longitude", -125.0, -66.0, -97.0, 0.1, key=f"lon_{label}")

    if st.button("🔍 Lookup", key=f"lookup_{label}"):
        points = df[['lat', 'lon']].values
        results, breakdowns = {}, {}
        for angle in ANGLES:
            try:
                interp = CloughTocher2DInterpolator(points, computed[f'total_{angle}'].values)
                val = interp(input_lat, input_lon)
                if np.isnan(val):
                    val = LinearNDInterpolator(points, computed[f'total_{angle}'].values)(input_lat, input_lon)
                results[angle] = float(val) if not np.isnan(val) else None
            except Exception:
                results[angle] = None

            bd = {}
            for layer_name, prefix in [('Insurance', 'ins'), ('Dev Risk', 'risk')]:
                try:
                    interp = CloughTocher2DInterpolator(points, computed[f'{prefix}_{angle}'].values)
                    v = interp(input_lat, input_lon)
                    if np.isnan(v):
                        v = LinearNDInterpolator(points, computed[f'{prefix}_{angle}'].values)(input_lat, input_lon)
                    bd[layer_name] = float(v) if not np.isnan(v) else 0.0
                except Exception:
                    bd[layer_name] = 0.0
            bd['CapEx'] = capex_dict[angle] if capex_on else 0.0
            breakdowns[angle] = bd

        valid = {a: v for a, v in results.items() if v is not None}
        if not valid:
            st.warning("⚠️ Outside data coverage. Try coordinates within the contiguous US.")
        else:
            best_angle = min(valid, key=valid.get)
            best_cost = valid[best_angle]
            sorted_angles = sorted(valid.items(), key=lambda x: x[1])

            html = f'<div class="lookup-result"><h3>📍 ({input_lat:.1f}, {input_lon:.1f})</h3>'
            html += f'<p style="color:#34d399;font-size:1.2rem;font-weight:700;">🏆 Best: {best_angle}° at {best_cost:.4f} ¢/W</p>'
            html += '<table style="width:100%;border-collapse:collapse;color:#e2e8f0;">'
            html += '<tr style="border-bottom:2px solid #475569;"><th style="text-align:left;padding:8px;">Angle</th>'
            html += '<th style="text-align:right;padding:8px;">Insurance</th><th style="text-align:right;padding:8px;">Dev Risk</th>'
            html += '<th style="text-align:right;padding:8px;">CapEx</th><th style="text-align:right;padding:8px;">Total</th>'
            html += '<th style="text-align:right;padding:8px;">vs Best</th></tr>'
            for angle, cost in sorted_angles:
                is_best = angle == best_angle
                s = 'color:#34d399;font-weight:700;' if is_best else ''
                bd = breakdowns[angle]
                delta = "—" if is_best else f"+{cost - best_cost:.4f}"
                mark = " ✓" if is_best else ""
                html += f'<tr style="border-bottom:1px solid #334155;{s}"><td style="padding:8px;">{angle}°{mark}</td>'
                html += f'<td style="text-align:right;padding:8px;">{bd["Insurance"]:.4f}</td>'
                html += f'<td style="text-align:right;padding:8px;">{bd["Dev Risk"]:.4f}</td>'
                html += f'<td style="text-align:right;padding:8px;">{bd["CapEx"]:.1f}</td>'
                html += f'<td style="text-align:right;padding:8px;">{cost:.4f}</td>'
                html += f'<td style="text-align:right;padding:8px;">{delta}</td></tr>'
            html += '</table></div>'
            st.markdown(html, unsafe_allow_html=True)


# ─── Luce/Logit Probabilistic Market Share ───
def render_luce_market_share(computed, shape_df, label=""):
    """Probability-weighted market share using the Luce/Logit choice model.

    Shares the same multi-year selection and shape/magnitude controls as the
    deterministic market share section (read from module-level state).
    """
    st.markdown("---")
    st.markdown("### 🎲 Probabilistic Market Share (Luce Model)"
                + (f" — {label}" if label else "")
                + f"  [σ = {sigma:.2f}, shape: {demand_shape}, magnitude: {magnitude_source}]")
    st.caption("Each location splits its demand probabilistically across active tilt angles using "
               "P(a) = exp(−σ·ΔCost_a) / Σ exp(−σ·ΔCost_j), where ΔCost is normalized to the "
               "per-location minimum. Higher σ = more decisive; σ=0 = uniform.")

    # Reuse the same year selection state as deterministic market share
    has_shape = shape_df is not None
    years = sorted(market_sizes.keys())
    selected_years = [yr for yr in years if st.session_state.get(f"yrsel_{yr}_{label}", True)]

    if not selected_years:
        st.warning("Select at least one year in the deterministic market share section above.")
        return None

    total_gw = sum(market_sizes[yr] for yr in selected_years)
    total_mw = total_gw * 1000
    yr_label = (f"{min(selected_years)}–{max(selected_years)}"
                if len(selected_years) > 1 else str(selected_years[0]))

    # Compute Luce probabilities
    luce = compute_luce_probabilities(computed, sigma, ACTIVE_ANGLES)

    # Build location-level scaled demand using the shape source
    merged_demand = None
    if has_shape:
        shape_sub = shape_df[shape_df['Year'].isin(selected_years)]
        if len(shape_sub) > 0:
            demand_by_loc = shape_sub.groupby(['hail_lat', 'hail_lon']).agg(
                total_mw=('DC Capacity (MW)', 'sum')).reset_index()
            merge_cols = ['lat', 'lon'] + [f'prob_{a}' for a in ANGLES]
            merged_demand = demand_by_loc.merge(
                luce[merge_cols],
                left_on=['hail_lat', 'hail_lon'], right_on=['lat', 'lon'], how='inner')
            if len(merged_demand) > 0:
                shape_total = merged_demand['total_mw'].sum()
                scale = total_mw / shape_total if shape_total > 0 else 1.0
                merged_demand['scaled_mw'] = merged_demand['total_mw'] * scale

    if merged_demand is None or len(merged_demand) == 0:
        # Uniform fallback
        mw_per_loc = total_mw / len(luce) if len(luce) > 0 else 0
        merged_demand = luce[['lat', 'lon'] + [f'prob_{a}' for a in ANGLES]].copy()
        merged_demand['scaled_mw'] = mw_per_loc
        data_source = "Uniform distribution"
    else:
        data_source = f"Shape: {demand_shape} (scaled to {total_gw:.1f} GW)"

    # Probability-adjusted market share by angle: sum across locations of (scaled_mw * P_a)
    angle_rows = []
    for a in ACTIVE_ANGLES:
        mw_a = (merged_demand['scaled_mw'] * merged_demand[f'prob_{a}']).sum()
        angle_rows.append({'Best Product': f'{a}°', 'MWdc': mw_a})
    angle_summary = pd.DataFrame(angle_rows)
    angle_summary['GWdc'] = (angle_summary['MWdc'] / 1000).round(2)
    tot = angle_summary['MWdc'].sum()
    angle_summary['Share (%)'] = (angle_summary['MWdc'] / tot * 100).round(1) if tot > 0 else 0
    angle_summary['MWdc'] = angle_summary['MWdc'].round(0).astype(int)

    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown(f"**{yr_label}**  —  Total Market: **{total_gw:.1f} GWdc**")
        st.caption(f"Source: {data_source}")
        st.dataframe(angle_summary[['Best Product', 'GWdc', 'MWdc', 'Share (%)']],
                     hide_index=True, use_container_width=True)
    with c2:
        chart_df = angle_summary.set_index('Best Product')[['GWdc']]
        st.bar_chart(chart_df, use_container_width=True, height=300)

    # Year-by-year probabilistic breakdown
    if len(selected_years) > 1:
        st.markdown("#### Year-by-Year Probabilistic Breakdown")
        yoy_rows = []
        for yr in selected_years:
            yr_mw = market_sizes[yr] * 1000
            if has_shape:
                yr_shape = shape_df[shape_df['Year'] == yr]
                if len(yr_shape) > 0:
                    dbl = yr_shape.groupby(['hail_lat', 'hail_lon']).agg(
                        total_mw=('DC Capacity (MW)', 'sum')).reset_index()
                    mrg = dbl.merge(luce[['lat', 'lon'] + [f'prob_{a}' for a in ANGLES]],
                                    left_on=['hail_lat', 'hail_lon'],
                                    right_on=['lat', 'lon'], how='inner')
                    if len(mrg) > 0:
                        sc = yr_mw / mrg['total_mw'].sum() if mrg['total_mw'].sum() > 0 else 1
                        mrg['scaled_mw'] = mrg['total_mw'] * sc
                        for a in ACTIVE_ANGLES:
                            mw_a = (mrg['scaled_mw'] * mrg[f'prob_{a}']).sum()
                            yoy_rows.append({'Year': yr, 'Angle': f'{a}°',
                                             'GWdc': round(mw_a / 1000, 2)})
                        continue
            # Uniform fallback
            mw_per = yr_mw / len(luce)
            for a in ACTIVE_ANGLES:
                mw_a = (mw_per * luce[f'prob_{a}']).sum()
                yoy_rows.append({'Year': yr, 'Angle': f'{a}°', 'GWdc': round(mw_a / 1000, 2)})
        if yoy_rows:
            yoy_df = pd.DataFrame(yoy_rows)
            pivot = yoy_df.pivot_table(index='Angle', columns='Year', values='GWdc',
                                       aggfunc='sum').fillna(0)
            st.dataframe(pivot, use_container_width=True)

    return {'luce': luce, 'merged_demand': merged_demand,
            'angle_summary': angle_summary, 'selected_years': selected_years,
            'total_gw': total_gw}


def render_luce_demand_map(luce_result, label=""):
    """Probability-adjusted demand map with red/yellow/green coloring vs random chance."""
    if luce_result is None:
        return
    st.markdown("---")
    st.markdown("### 🎯 Probability-Adjusted Demand Map" + (f" — {label}" if label else ""))
    st.markdown("Select which tilt angles to highlight. **Bar height** = probability-adjusted demand "
                "from the selected angles. **Bar color** = how the selected angles' combined probability "
                "compares to random chance (k/n where k = selected, n = active).")

    # Checkboxes — second layer of selection, only among active angles
    cb_cols = st.columns(max(1, len(ACTIVE_ANGLES)))
    selected_angles = []
    for i, angle in enumerate(ACTIVE_ANGLES):
        with cb_cols[i]:
            if st.checkbox(f"{angle}°", value=True, key=f"luce_{angle}_{label}"):
                selected_angles.append(angle)

    if len(selected_angles) == 0:
        st.info("Select at least one angle to display the probability-adjusted map.")
        return

    n_active = len(ACTIVE_ANGLES)
    k_selected = len(selected_angles)
    p_random = k_selected / n_active  # baseline probability if costs were equal

    merged_demand = luce_result['merged_demand']
    map_df = merged_demand[['lat', 'lon', 'scaled_mw'] +
                             [f'prob_{a}' for a in ACTIVE_ANGLES]].copy()

    # Sum the probabilities of selected angles
    map_df['p_selected'] = sum(map_df[f'prob_{a}'] for a in selected_angles)
    # Probability-adjusted MW from the selected subset
    map_df['adj_mw'] = map_df['scaled_mw'] * map_df['p_selected']

    # Color: log ratio of P_selected to p_random, clamped to [-1.5, 1.5]
    # Edge case: if all angles selected, p_selected = 1.0 and p_random = 1.0 -> ratio = 0 -> yellow
    if p_random > 0:
        # Avoid log(0); floor p_selected at a tiny value
        p_safe = map_df['p_selected'].clip(lower=1e-6)
        map_df['log_ratio'] = np.log(p_safe / p_random).clip(-1.5, 1.5)
    else:
        map_df['log_ratio'] = 0.0
    map_df['color_t'] = (map_df['log_ratio'] + 1.5) / 3.0  # normalize to [0, 1]

    # Build RGB scaled palette: red (low) -> yellow (mid) -> green (high)
    def color_for(t):
        # t in [0, 1]; 0 = dark red, 0.5 = yellow, 1 = dark green
        if t <= 0.5:
            # Red -> Yellow
            f = t / 0.5  # 0..1
            r = int(180 + (240 - 180) * f)  # 180 -> 240
            g = int(20 + (200 - 20) * f)    # 20 -> 200
            b = int(20 + (40 - 20) * f)     # 20 -> 40
        else:
            # Yellow -> Green
            f = (t - 0.5) / 0.5  # 0..1
            r = int(240 - (240 - 20) * f)   # 240 -> 20
            g = int(200 + (140 - 200) * f)  # 200 -> 140
            b = int(40 + (60 - 40) * f)     # 40 -> 60
        return [r, g, b]

    map_df['color'] = map_df['color_t'].apply(color_for)

    # Map values
    map_df = map_df[map_df['adj_mw'] > 0].copy()
    if len(map_df) == 0:
        st.info("No demand to display.")
        return

    map_df['elevation'] = map_df['adj_mw'] * 200
    map_df['adj_mw_display'] = map_df['adj_mw'].round(0).astype(int).astype(str)
    map_df['p_sel_display'] = (map_df['p_selected'] * 100).round(1).astype(str) + '%'
    map_df['p_rand_display'] = f"{p_random*100:.1f}%"
    map_df['ratio_display'] = (map_df['p_selected'] / p_random).round(2).astype(str) + '×'

    col_layer = pdk.Layer(
        "ColumnLayer", data=map_df, get_position='[lon, lat]',
        get_elevation='elevation', elevation_scale=1, radius=18000,
        get_fill_color='color', pickable=True, auto_highlight=True,
    )
    states_layer = pdk.Layer("GeoJsonLayer", data=US_STATES_URL,
                             stroked=True, filled=False, pickable=False,
                             get_line_color=[100, 100, 100, 140], line_width_min_pixels=1)
    st.pydeck_chart(pdk.Deck(
        layers=[states_layer, col_layer],
        initial_view_state=pdk.ViewState(latitude=39.0, longitude=-98.0, zoom=3.8, pitch=45),
        map_style="light",
        tooltip={"html": ("<b>Selected P:</b> {p_sel_display} (random: {p_rand_display}, "
                           "{ratio_display})<br><b>Adjusted Demand:</b> {adj_mw_display} MWdc"),
                 "style": {"backgroundColor": "#1e293b", "color": "#e2e8f0",
                            "fontSize": "13px", "padding": "8px 12px", "borderRadius": "8px"}},
    ), use_container_width=True, height=500)

    sel_str = ", ".join(f"{a}°" for a in selected_angles)
    st.markdown(f"**Selected:** {sel_str}  |  **k/n random chance:** {p_random*100:.1f}%  |  "
                f"**Avg P_selected:** {map_df['p_selected'].mean()*100:.1f}%  |  "
                f"**Total Adj Demand:** {map_df['adj_mw'].sum()/1000:.2f} GWdc")
    st.markdown("🔴 Red = less likely than random   🟡 Yellow = matches random chance   🟢 Green = more likely than random")


# ═══════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════
st.markdown('<p class="main-title">🌨️ Hail Risk Total Cost of Ownership</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Interactive sensitivity analysis across tracker stow angles (52°, 60°, 70°, 77°)</p>',
            unsafe_allow_html=True)

new_annuity = annuity_factor(interest_rate / 100.0)
st.caption(f"Discount rate: {interest_rate:.2f}% → {new_annuity:.2f}× annuity "
           f"(base: 6.00% → {BASE_ANNUITY:.2f}×)  |  "
           f"Premium: {annual_premium:.2f}%  |  Coverage: {coverage_ratio}%  |  "
           f"Demand shape: **{demand_shape}**  |  Magnitude: **{magnitude_source}**  |  "
           f"σ = **{sigma:.2f}**")

shape_df = _shape_df  # demand shape data (Orennia or WoodMac)

if glass_choice == "Compare Both":
    df20, df32 = load_data('20'), load_data('32')
    tab1, tab2 = st.tabs(["2.0 mm Glass", "3.2 mm Glass"])
    with tab1:
        comp20 = render_single(df20, "2.0 mm Glass")
        render_lookup(df20, comp20, "2.0mm")
        render_market_share(comp20, shape_df, "2.0mm")
        render_value_gap(comp20, "2.0mm")
        luce20 = render_luce_market_share(comp20, shape_df, "2.0mm")
        render_luce_demand_map(luce20, "2.0mm")
    with tab2:
        comp32 = render_single(df32, "3.2 mm Glass")
        render_lookup(df32, comp32, "3.2mm")
        render_market_share(comp32, shape_df, "3.2mm")
        render_value_gap(comp32, "3.2mm")
        luce32 = render_luce_market_share(comp32, shape_df, "3.2mm")
        render_luce_demand_map(luce32, "3.2mm")

    # Glass comparison
    st.markdown("---")
    st.markdown("### 🔀 Where Does Glass Thickness Change the Optimal Angle?")
    merged = comp20[['lat', 'lon', 'best_angle', 'best_cost']].merge(
        comp32[['lat', 'lon', 'best_angle', 'best_cost']], on=['lat', 'lon'], suffixes=('_20', '_32'))
    changed = merged[merged['best_angle_20'] != merged['best_angle_32']]
    st.markdown(f"At **{len(changed)}** of {len(merged)} locations ({len(changed)/len(merged)*100:.1f}%), "
                f"the optimal angle differs between glass types.")
    if len(changed) > 0:
        cdf = changed[['lat', 'lon', 'best_angle_20', 'best_cost_20', 'best_angle_32', 'best_cost_32']].copy()
        cdf.columns = ['Lat', 'Lon', '2.0mm Best', '2.0mm Cost', '3.2mm Best', '3.2mm Cost']
        cdf['2.0mm Cost'] = cdf['2.0mm Cost'].round(4)
        cdf['3.2mm Cost'] = cdf['3.2mm Cost'].round(4)
        st.dataframe(cdf.sort_values('Lat'), hide_index=True, use_container_width=True, height=300)

elif glass_choice == "Blended":
    df20, df32 = load_data('20'), load_data('32')
    blended = compute_blended(df20, df32, glass_pct_20, replacement_cost, coverage_ratio,
                              annual_premium, interest_rate, risk_pct, capex_dict,
                              ins_on, risk_on, capex_on, active_angles=ACTIVE_ANGLES)
    label = f"Blended ({glass_pct_20}% 2.0mm / {100-glass_pct_20}% 3.2mm)"
    render_single(df32, label, precomputed=blended)
    render_lookup(df32, blended, "Blended")
    render_market_share(blended, shape_df, "Blended")
    render_value_gap(blended, "Blended")
    luce_b = render_luce_market_share(blended, shape_df, "Blended")
    render_luce_demand_map(luce_b, "Blended")

else:
    suffix = '32' if glass_choice == "3.2 mm" else '20'
    df = load_data(suffix)
    computed = render_single(df, glass_choice)
    render_lookup(df, computed, glass_choice)
    render_market_share(computed, shape_df, glass_choice)
    render_value_gap(computed, glass_choice)
    luce_r = render_luce_market_share(computed, shape_df, glass_choice)
    render_luce_demand_map(luce_r, glass_choice)

# ─── Parameter Summary ───
st.markdown("---")
st.markdown("### 📊 Current Parameter Summary")
p1, p2, p3 = st.columns(3)
with p1:
    glass_label = glass_choice
    if glass_choice == "Blended":
        glass_label = f"Blended ({glass_pct_20}/{100-glass_pct_20})"
    st.markdown(f"| Parameter | Value |\n|---|---|\n| Glass Type | **{glass_label}** |"
                f"\n| Replacement Cost | **${replacement_cost:.2f}/W** |"
                f"\n| Discount Rate | **{interest_rate:.2f}%** |\n| Annuity Factor | **{new_annuity:.2f}×** |"
                f"\n| Active Angles | **{', '.join(str(a)+'°' for a in ACTIVE_ANGLES)}** |")
with p2:
    st.markdown(f"| Parameter | Value |\n|---|---|\n| Annual Premium | **{annual_premium:.2f}%** |"
                f"\n| Coverage Ratio | **{coverage_ratio}%** |\n| Dev Risk Factor | **{risk_pct}%** |"
                f"\n| Active Layers | **{layers_str}** |"
                f"\n| Demand Shape | **{demand_shape}** |"
                f"\n| Magnitude Source | **{magnitude_source}** |"
                f"\n| Sigma (Luce) | **{sigma:.2f}** |")
with p3:
    mkt_str = "| Year | GWdc |\n|---|---|"
    for yr in sorted(market_sizes.keys()):
        mkt_str += f"\n| {yr} | **{market_sizes[yr]:.1f}** |"
    st.markdown(mkt_str)
