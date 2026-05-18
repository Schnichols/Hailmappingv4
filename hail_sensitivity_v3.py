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

# Default market size by year (GWdc)
DEFAULT_MARKET = {2025: 30, 2026: 36, 2027: 44, 2028: 50, 2029: 55, 2030: 60}


def annuity_factor(r, n=40):
    if r <= 0:
        return float(n)
    return sum(1 / (1 + r) ** y for y in range(1, n + 1))


# ─── Page Config ───
st.set_page_config(page_title="Hail Risk Sensitivity Tool", page_icon="🌨️",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=JetBrains+Mono:wght@400;500&display=swap');
    .stApp { font-family: 'DM Sans', sans-serif; }
    .main-title { font-size: 2rem; font-weight: 700; color: #1a1a2e; margin-bottom: 0; letter-spacing: -0.5px; }
    .subtitle { font-size: 1rem; color: #6b7280; margin-top: 0; margin-bottom: 1.5rem; }
    section[data-testid="stSidebar"] { background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%); color: #e2e8f0; }
    section[data-testid="stSidebar"] .stMarkdown h3 { color: #94a3b8; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1.5px; margin-top: 1.5rem; }
    section[data-testid="stSidebar"] label { color: #cbd5e1 !important; }
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
def load_orennia():
    path = os.path.join(SCRIPT_DIR, 'orennia_matched.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def compute_costs(df, replacement_cost, coverage_ratio, annual_premium,
                  interest_rate, risk_pct, capex_dict, ins_on, risk_on, capex_on):
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

    total_cols = [f'total_{a}' for a in ANGLES]
    computed['best_angle'] = computed[total_cols].idxmin(axis=1).str.replace('total_', '').astype(int)
    computed['best_cost'] = computed[total_cols].min(axis=1)
    return computed


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

st.sidebar.markdown("### Glass Type")
glass_choice = st.sidebar.radio("Module glass thickness", ["3.2 mm", "2.0 mm", "Compare Both"],
                                 index=0, horizontal=True)

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

st.sidebar.markdown("### Market Size by Year (GWdc)")
market_sizes = {}
for yr, default_gw in DEFAULT_MARKET.items():
    market_sizes[yr] = st.sidebar.number_input(
        f"{yr}", min_value=0.0, max_value=200.0,
        value=float(default_gw), step=1.0, key=f"mkt_{yr}")

layers_active = []
if ins_on: layers_active.append("Ins")
if risk_on: layers_active.append("Risk")
if capex_on: layers_active.append("CapEx")
layers_str = " + ".join(layers_active) if layers_active else "None"


# ═══════════════════════════════════════════
# RENDERING FUNCTIONS
# ═══════════════════════════════════════════

def render_single(df, label=""):
    computed = compute_costs(df, replacement_cost, coverage_ratio, annual_premium,
                             interest_rate, risk_pct, capex_dict, ins_on, risk_on, capex_on)

    if label:
        st.markdown(f"#### {label}")

    # ─── Metric Cards ───
    cols = st.columns(4)
    for i, angle in enumerate(ANGLES):
        avg = computed[f'total_{angle}'].mean()
        c = ANGLE_COLORS[angle]
        with cols[i]:
            st.markdown(f'<div class="metric-card"><h4>{angle}° Avg Total</h4>'
                        f'<div class="value" style="color:rgb({c[0]},{c[1]},{c[2]})">{avg:.3f}</div>'
                        f'<div class="unit">¢/W</div></div>', unsafe_allow_html=True)

    win_counts = computed['best_angle'].value_counts()
    overall_winner = win_counts.idxmax()
    win_pct = win_counts.max() / len(computed) * 100
    st.markdown(f'<div class="winner-banner">🏆 {overall_winner}° wins at {win_counts.max()} of '
                f'{len(computed)} locations ({win_pct:.1f}%) — Layers: {layers_str}</div>',
                unsafe_allow_html=True)

    # ─── 3D Column Map ───
    st.markdown("#### 🗺️ Total Cost by Angle (3D Columns)")
    map_rows = []
    for _, row in computed.iterrows():
        for angle in ANGLES:
            val = row[f'total_{angle}']
            if val > 0:
                map_rows.append({
                    'lat': row['lat'],
                    'lon': row['lon'] + (ANGLES.index(angle) - 1.5) * 0.15,
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

    # ─── Win Count Table ───
    st.markdown("#### 📊 Win Counts")
    wc = pd.DataFrame({
        'Angle': [f'{a}°' for a in ANGLES],
        'Wins': [win_counts.get(a, 0) for a in ANGLES],
        'Win %': [f"{win_counts.get(a, 0)/len(computed)*100:.1f}%" for a in ANGLES],
        'Avg Total (¢/W)': [f"{computed[f'total_{a}'].mean():.4f}" for a in ANGLES],
    })
    st.dataframe(wc, hide_index=True, use_container_width=True)

    return computed


# ─── Market Share & Demand Analysis ───
def render_market_share(computed, orennia_df, label=""):
    """Market share by tilt angle with year toggle and scalable market size."""
    st.markdown("---")
    st.markdown("### 📈 Market Share by Tilt Angle" + (f" — {label}" if label else ""))

    has_orennia = orennia_df is not None
    years = sorted(market_sizes.keys())
    year_options = ['All Years'] + [str(y) for y in years]
    selected_year = st.selectbox("Select Year", year_options, index=0, key=f"mkt_year_{label}")

    if selected_year == 'All Years':
        total_gw = sum(market_sizes.values())
        yr_label = f"All Years ({min(years)}–{max(years)})"
    else:
        yr = int(selected_year)
        total_gw = market_sizes.get(yr, 0)
        yr_label = selected_year

    total_mw = total_gw * 1000

    # Method 1: If Orennia data exists, use project-level demand weighted by location
    if has_orennia:
        if selected_year == 'All Years':
            oren_sub = orennia_df.copy()
        else:
            oren_sub = orennia_df[orennia_df['Year'] == int(selected_year)].copy()

        if len(oren_sub) > 0:
            demand_by_loc = oren_sub.groupby(['hail_lat', 'hail_lon']).agg(
                total_mw=('DC Capacity (MW)', 'sum'),
                project_count=('DC Capacity (MW)', 'count'),
            ).reset_index()

            merged = demand_by_loc.merge(
                computed[['lat', 'lon', 'best_angle']],
                left_on=['hail_lat', 'hail_lon'], right_on=['lat', 'lon'], how='inner')

            if len(merged) > 0:
                # Scale Orennia MW to match user-specified market size
                orennia_total = merged['total_mw'].sum()
                scale = total_mw / orennia_total if orennia_total > 0 else 1.0
                merged['scaled_mw'] = merged['total_mw'] * scale

                angle_summary = merged.groupby('best_angle')['scaled_mw'].sum().reset_index()
                angle_summary.columns = ['Best Product', 'MWdc']
                data_source = "Orennia pipeline (scaled)"
            else:
                has_orennia = False  # fallback

    # Method 2: No Orennia — distribute market evenly across grid locations
    if not has_orennia or (has_orennia and selected_year != 'All Years' and len(oren_sub) == 0):
        n_locations = len(computed)
        mw_per_loc = total_mw / n_locations if n_locations > 0 else 0
        angle_summary = computed.groupby('best_angle').size().reset_index(name='count')
        angle_summary['MWdc'] = angle_summary['count'] * mw_per_loc
        angle_summary = angle_summary[['best_angle', 'MWdc']].copy()
        angle_summary.columns = ['Best Product', 'MWdc']
        data_source = "Uniform distribution"

    # Ensure all angles appear
    for a in ANGLES:
        if a not in angle_summary['Best Product'].values:
            angle_summary = pd.concat([angle_summary, pd.DataFrame({'Best Product': [a], 'MWdc': [0]})],
                                       ignore_index=True)

    angle_summary = angle_summary.sort_values('Best Product')
    angle_summary['GWdc'] = (angle_summary['MWdc'] / 1000).round(2)
    angle_summary['Share (%)'] = (angle_summary['MWdc'] / angle_summary['MWdc'].sum() * 100).round(1)
    angle_summary['Best Product'] = angle_summary['Best Product'].astype(str) + '°'

    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown(f"**{yr_label}**  —  Total Market: **{total_gw:.0f} GWdc**")
        st.caption(f"Source: {data_source}")
        display_df = angle_summary[['Best Product', 'GWdc', 'MWdc', 'Share (%)']].copy()
        display_df['MWdc'] = display_df['MWdc'].round(0).astype(int)
        st.dataframe(display_df, hide_index=True, use_container_width=True)

    with c2:
        chart_df = angle_summary.set_index('Best Product')[['GWdc']]
        st.bar_chart(chart_df, use_container_width=True, height=300)

    # Year-over-year comparison table
    if selected_year == 'All Years':
        st.markdown("#### Year-by-Year Breakdown")
        yoy_rows = []
        for yr in years:
            yr_mw = market_sizes[yr] * 1000
            n_locs = len(computed)
            mw_per = yr_mw / n_locs if n_locs > 0 else 0

            if has_orennia and orennia_df is not None:
                yr_oren = orennia_df[orennia_df['Year'] == yr]
                if len(yr_oren) > 0:
                    dbl = yr_oren.groupby(['hail_lat', 'hail_lon']).agg(
                        total_mw=('DC Capacity (MW)', 'sum')).reset_index()
                    mrg = dbl.merge(computed[['lat', 'lon', 'best_angle']],
                                    left_on=['hail_lat', 'hail_lon'],
                                    right_on=['lat', 'lon'], how='inner')
                    if len(mrg) > 0:
                        sc = yr_mw / mrg['total_mw'].sum() if mrg['total_mw'].sum() > 0 else 1
                        mrg['scaled_mw'] = mrg['total_mw'] * sc
                        for a in ANGLES:
                            sub = mrg[mrg['best_angle'] == a]
                            yoy_rows.append({'Year': yr, 'Angle': f'{a}°',
                                             'GWdc': round(sub['scaled_mw'].sum() / 1000, 2),
                                             'MWdc': int(sub['scaled_mw'].sum())})
                        continue

            # Fallback: uniform
            counts = computed['best_angle'].value_counts()
            for a in ANGLES:
                cnt = counts.get(a, 0)
                yoy_rows.append({'Year': yr, 'Angle': f'{a}°',
                                 'GWdc': round(cnt * mw_per / 1000, 2),
                                 'MWdc': int(cnt * mw_per)})

        if yoy_rows:
            yoy_df = pd.DataFrame(yoy_rows)
            pivot = yoy_df.pivot_table(index='Angle', columns='Year', values='GWdc',
                                        aggfunc='sum').fillna(0)
            st.dataframe(pivot, use_container_width=True)


# ─── Value Gap Analysis (portfolio baseline) ───
def render_value_gap(computed, label=""):
    st.markdown("---")
    st.markdown("### 💰 Value Gap Analysis" + (f" — {label}" if label else ""))
    st.markdown("The **base case** is the full 4-product portfolio (always picks cheapest). "
                "Selecting fewer products shows the added cost of limiting your portfolio.")

    gap_cols = st.columns(4)
    selected = []
    for i, angle in enumerate(ANGLES):
        with gap_cols[i]:
            if st.checkbox(f"{angle}°", value=True, key=f"vg_{angle}_{label}"):
                selected.append(angle)

    if len(selected) == 0:
        st.warning("Select at least one product angle.")
        return

    # Base case: full portfolio (min across all 4)
    computed['base_cost'] = computed[[f'total_{a}' for a in ANGLES]].min(axis=1)

    # Subset: min across selected angles only
    if len(selected) == len(ANGLES):
        computed['subset_cost'] = computed['base_cost']
    else:
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


# ═══════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════
st.markdown('<p class="main-title">🌨️ Hail Risk Total Cost of Ownership</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Interactive sensitivity analysis across tracker stow angles (52°, 60°, 70°, 77°)</p>',
            unsafe_allow_html=True)

new_annuity = annuity_factor(interest_rate / 100.0)
st.caption(f"Discount rate: {interest_rate:.2f}% → {new_annuity:.2f}× annuity "
           f"(base: 6.00% → {BASE_ANNUITY:.2f}×)  |  "
           f"Premium: {annual_premium:.2f}%  |  Coverage: {coverage_ratio}%")

orennia_df = load_orennia()

if glass_choice == "Compare Both":
    df20, df32 = load_data('20'), load_data('32')
    tab1, tab2 = st.tabs(["2.0 mm Glass", "3.2 mm Glass"])
    with tab1:
        comp20 = render_single(df20, "2.0 mm Glass")
        render_lookup(df20, comp20, "2.0mm")
        render_market_share(comp20, orennia_df, "2.0mm")
        render_value_gap(comp20, "2.0mm")
    with tab2:
        comp32 = render_single(df32, "3.2 mm Glass")
        render_lookup(df32, comp32, "3.2mm")
        render_market_share(comp32, orennia_df, "3.2mm")
        render_value_gap(comp32, "3.2mm")

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

else:
    suffix = '32' if glass_choice == "3.2 mm" else '20'
    df = load_data(suffix)
    computed = render_single(df, glass_choice)
    render_lookup(df, computed, glass_choice)
    render_market_share(computed, orennia_df, glass_choice)
    render_value_gap(computed, glass_choice)

# ─── Parameter Summary ───
st.markdown("---")
st.markdown("### 📊 Current Parameter Summary")
p1, p2, p3 = st.columns(3)
with p1:
    st.markdown(f"| Parameter | Value |\n|---|---|\n| Replacement Cost | **${replacement_cost:.2f}/W** |"
                f"\n| Discount Rate | **{interest_rate:.2f}%** |\n| Annuity Factor | **{new_annuity:.2f}×** |")
with p2:
    st.markdown(f"| Parameter | Value |\n|---|---|\n| Annual Premium | **{annual_premium:.2f}%** |"
                f"\n| Coverage Ratio | **{coverage_ratio}%** |\n| Dev Risk Factor | **{risk_pct}%** |"
                f"\n| Active Layers | **{layers_str}** |")
with p3:
    mkt_str = "| Year | GWdc |\n|---|---|"
    for yr in sorted(market_sizes.keys()):
        mkt_str += f"\n| {yr} | **{market_sizes[yr]:.0f}** |"
    st.markdown(mkt_str)
