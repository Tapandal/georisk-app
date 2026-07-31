"""
GeoRisk Parametric Monitor
==========================
A Streamlit app for live Sentinel-2 anomaly detection.
"""

import streamlit as st
import ee
import folium
from streamlit_folium import st_folium
import requests
import datetime
from datetime import timedelta
import numpy as np
import pandas as pd
import tempfile
import os

# -----------------------------------------------------------------------------
# Page Setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="GeoRisk Parametric Monitor",
    page_icon="🛰️",
    layout="wide",
)

# -----------------------------------------------------------------------------
# Pretty CSS for the cards
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; }
    .score-value {
        font-size: 3.5rem;
        font-weight: 700;
        line-height: 1.1;
        margin: 0.5rem 0;
    }
    .score-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #666;
    }
    .status-pill {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.95rem;
        margin-top: 8px;
    }
    .status-normal { background: #d1fae5; color: #065f46; }
    .status-elevated { background: #fef3c7; color: #92400e; }
    .status-high { background: #ffedd5; color: #9a3412; }
    .status-critical { background: #fee2e2; color: #991b1b; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Connect to Google Earth Engine
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Connecting to satellite database...")
def init_gee():
    """Login to Google Earth Engine."""
    try:
        ee.Initialize(project='my-satellite-app-504119')
        return True
    except Exception:
        pass
    try:
        gee_cfg = st.secrets.get("gee", {})
        if gee_cfg and "service_account" in gee_cfg and "private_key" in gee_cfg:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
                tmp.write(gee_cfg["private_key"])
                key_path = tmp.name
            credentials = ee.ServiceAccountCredentials(
                email=gee_cfg["service_account"], key_file=key_path
            )
            os.unlink(key_path)
            ee.Initialize(credentials)
            return True
    except Exception:
        pass
    st.error("Could not connect to satellite database.")
    st.info("If running locally, type: py -c \"import ee; ee.Authenticate()\"")
    return False


# -----------------------------------------------------------------------------
# Geocoding
# -----------------------------------------------------------------------------
def geocode_pincode(pincode, country="India"):
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {"q": f"{pincode}, {country}", "format": "json", "limit": 1}
        headers = {"User-Agent": "GeoRiskMonitor/1.0"}
        data = requests.get(url, params=params, headers=headers, timeout=12).json()
        if data:
            return float(data[0]["lat"]), float(data[0]["lon"])
    except Exception as exc:
        st.error(f"Geocoding error: {exc}")
    return None, None


# -----------------------------------------------------------------------------
# Satellite Helpers
# -----------------------------------------------------------------------------
def add_ndvi(image):
    ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
    return image.addBands(ndvi)


def get_recent_collection(aoi, days_back):
    end = datetime.date.today()
    start = end - timedelta(days=days_back)
    return (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(str(start), str(end))
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 25))
        .map(add_ndvi)
        .sort("system:time_start", False)
    )


def get_baseline_collection(aoi, start_date, end_date, years=5):
    images = []
    for y in range(1, years + 1):
        yr = datetime.date.today().year - y
        try:
            s = datetime.date(yr, start_date.month, start_date.day)
            e = datetime.date(yr, end_date.month, end_date.day)
        except ValueError:
            s = datetime.date(yr, start_date.month, min(start_date.day, 28))
            e = datetime.date(yr, end_date.month, min(end_date.day, 28))
        yearly = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(aoi)
            .filterDate(str(s), str(e))
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
            .map(add_ndvi)
            .select("NDVI")
            .median()
        )
        images.append(yearly.set("year", yr))
    return ee.ImageCollection.fromImages(images)


def safe_region_stats(image, geometry, scale=30):
    try:
        return image.reduceRegion(
            reducer=ee.Reducer.mean(), geometry=geometry, scale=scale,
            maxPixels=1e9, bestEffort=True,
        ).getInfo() or {}
    except Exception:
        return {}


# -----------------------------------------------------------------------------
# MAIN APP
# -----------------------------------------------------------------------------
def main():
    st.title("🛰️ GeoRisk Parametric Monitor")
    st.caption("Live satellite check for insurance triggers. Compares today's plants vs. 5-year history.")

    # --- Sidebar Inputs ---
    with st.sidebar:
        st.header("📍 Location Input")
        input_mode = st.radio("Input Type", ["Lat / Lon", "Pincode"], horizontal=True)

        if input_mode == "Lat / Lon":
            lat = st.number_input("Latitude", value=26.9124, format="%.6f")
            lon = st.number_input("Longitude", value=75.7873, format="%.6f")
        else:
            pin = st.text_input("Pincode", value="302001")
            country = st.text_input("Country", value="India")
            if st.button("🔍 Find Location"):
                g_lat, g_lon = geocode_pincode(pin, country)
                if g_lat:
                    st.session_state["lat"] = g_lat
                    st.session_state["lon"] = g_lon
                    st.success(f"Found: {g_lat:.5f}, {g_lon:.5f}")
                else:
                    st.error("Pincode not found.")
            lat = st.session_state.get("lat", 26.9124)
            lon = st.session_state.get("lon", 75.7873)

        st.divider()
        st.header("⚙️ Settings")
        buffer_km = st.slider("Check Radius (km)", 1, 10, 3)
        lookback_days = st.slider("How many days back to look?", 30, 120, 60)

        st.divider()
        st.header("🗺️ Map Layers")
        base_map = st.radio(
            "Choose Base Map",
            ["Google Satellite (Sharp)", "OpenStreetMap (Roads)"],
            index=0,
        )
        show_ndvi = st.toggle("🌱 Show Plant Health (NDVI)", value=False)
        show_anomaly = st.toggle("⚠️ Show Anomaly Score", value=False)

    # --- Connect to GEE ---
    if not init_gee():
        st.stop()

    point = ee.Geometry.Point([float(lon), float(lat)])
    aoi = point.buffer(buffer_km * 1000)

    today = datetime.date.today()
    recent_start = today - timedelta(days=lookback_days)
    recent_end = today

    # --- Fetch Data ---
    with st.spinner("Fetching latest satellite photo..."):
        recent_coll = get_recent_collection(aoi, lookback_days)
        recent_count = recent_coll.size().getInfo()
        if recent_count == 0:
            st.warning("No clear photos found. Searching wider...")
            recent_coll = get_recent_collection(aoi, 120)
            recent_count = recent_coll.size().getInfo()
            recent_start = today - timedelta(days=120)

    if recent_count == 0:
        st.error("No satellite photos available for this area.")
        st.stop()

    with st.spinner("Fetching 5-year history..."):
        baseline_coll = get_baseline_collection(aoi, recent_start, recent_end, years=5)
        baseline_median = baseline_coll.median().clip(aoi)
        baseline_std = baseline_coll.reduce(ee.Reducer.stdDev()).clip(aoi)

    recent_median = recent_coll.select("NDVI").median().clip(aoi)

    epsilon = 1e-6
    anomaly = (
        recent_median.subtract(baseline_median)
        .divide(baseline_std.add(epsilon))
        .rename("anomaly")
    )

    with st.spinner("Calculating scores..."):
        recent_stats = safe_region_stats(recent_median, aoi)
        baseline_stats = safe_region_stats(baseline_median, aoi)
        anomaly_stats = safe_region_stats(anomaly, aoi)

    recent_ndvi = recent_stats.get("NDVI")
    baseline_ndvi = baseline_stats.get("NDVI")
    z_score = anomaly_stats.get("anomaly")

    # --- Build Map ---
    col_map, col_metrics = st.columns([3, 2], gap="large")

    with col_map:
        st.subheader("Live Satellite Map")

        # Choose base layer
        if base_map == "Google Satellite (Sharp)":
            m = folium.Map(
                location=[lat, lon], zoom_start=13,
                tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
                attr="Google Satellite",
            )
        else:
            m = folium.Map(location=[lat, lon], zoom_start=13, tiles="OpenStreetMap")

        # Add overlays based on toggles
        if show_ndvi:
            ndvi_vis = {
                "bands": ["NDVI"], "min": -0.2, "max": 0.8,
                "palette": ["#d73027", "#fc8d59", "#fee08b", "#91cf60", "#1a9850"],
            }
            ndvi_map = recent_median.visualize(**ndvi_vis).getMapId()
            folium.TileLayer(
                tiles=ndvi_map["tile_fetcher"].url_format,
                attr="NDVI", name="Plant Health", overlay=True, control=False,
            ).add_to(m)

        if show_anomaly:
            anom_vis = {
                "bands": ["anomaly"], "min": -3, "max": 3,
                "palette": ["#b2182b", "#ef8a62", "#fddbc7", "#d1e5f0", "#67a9cf", "#2166ac"],
            }
            anom_map = anomaly.visualize(**anom_vis).getMapId()
            folium.TileLayer(
                tiles=anom_map["tile_fetcher"].url_format,
                attr="Z-Score", name="Anomaly", overlay=True, control=False,
            ).add_to(m)

        # Always add the analysis circle
        folium.Circle(
            location=[lat, lon], radius=buffer_km * 1000,
            color="#1e3a8a", fill=False, weight=2.5,
        ).add_to(m)

        st_folium(m, width=700, height=550, returned_objects=[])

        # Legend
        if show_ndvi:
            st.caption("🟢 Green = Healthy plants | 🔴 Red = Stressed/Dead plants")
        if show_anomaly:
            st.caption("🔵 Blue = Normal | 🔴 Red = Severely abnormal vs. 5-year history")

    # --- Metrics Panel ---
    with col_metrics:
        st.subheader("Insurance Risk Check")

        m1, m2 = st.columns(2)
        with m1:
            delta = (recent_ndvi - baseline_ndvi) if (recent_ndvi and baseline_ndvi) else 0
            st.metric(
                label="Recent Plant Health",
                value=f"{recent_ndvi:.3f}" if recent_ndvi else "N/A",
                delta=f"{delta:.3f}" if recent_ndvi else None,
                delta_color="inverse",
            )
        with m2:
            st.metric(
                label="5-Year Average",
                value=f"{baseline_ndvi:.3f}" if baseline_ndvi else "N/A",
            )

        st.divider()

        if z_score is not None:
            if z_score <= -2.5:
                status_text, status_color, status_bg, trigger_msg, trigger_level = (
                    "CRITICAL TRIGGER", "#991b1b", "#fee2e2",
                    "PAYOUT CONDITION MET — Claim is eligible.", "error"
                )
            elif z_score <= -1.5:
                status_text, status_color, status_bg, trigger_msg, trigger_level = (
                    "HIGH RISK", "#9a3412", "#ffedd5",
                    "WATCH — Close to automatic payout.", "warning"
                )
            elif z_score <= -0.5:
                status_text, status_color, status_bg, trigger_msg, trigger_level = (
                    "ELEVATED", "#92400e", "#fef3c7",
                    "ADVISORY — Slightly below normal.", "info"
                )
            else:
                status_text, status_color, status_bg, trigger_msg, trigger_level = (
                    "NORMAL", "#065f46", "#d1fae5",
                    "NO TRIGGER — Everything looks normal.", "success"
                )

            st.markdown(
                f"""
                <div style="background: {status_bg}; border-radius: 12px; padding: 24px; text-align: center; margin-bottom: 16px;">
                    <div class="score-label">Anomaly Distance Score</div>
                    <div class="score-value" style="color: {status_color};">{z_score:.2f}</div>
                    <span class="status-pill" style="background: #ffffff; color: {status_color}; border: 1px solid {status_color};">
                        {status_text}
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if trigger_level == "error":
                st.error(f"**{trigger_msg}**")
            elif trigger_level == "warning":
                st.warning(f"**{trigger_msg}**")
            elif trigger_level == "info":
                st.info(f"**{trigger_msg}**")
            else:
                st.success(f"**{trigger_msg}**")
        else:
            st.warning("Could not calculate score. Not enough history.")

        st.divider()
        st.markdown("**5-Year History Chart**")
        chart_data = []
        for y in range(5, 0, -1):
            yr = today.year - y
            img = baseline_coll.filter(ee.Filter.eq("year", yr)).first()
            if img:
                stat = safe_region_stats(img, aoi)
                val = stat.get("NDVI")
                if val is not None:
                    chart_data.append({"Period": str(yr), "NDVI": val})
        if recent_ndvi is not None:
            chart_data.append({"Period": f"{today.year} (Now)", "NDVI": recent_ndvi})
        if chart_data:
            st.bar_chart(pd.DataFrame(chart_data).set_index("Period"), use_container_width=True, color="#3b82f6")
        else:
            st.caption("No history to show.")

        st.caption(f"🛰️ Sentinel-2 | 📍 {buffer_km} km | 🗓️ {recent_start} to {recent_end}")

        if st.button("📄 Export Report", use_container_width=True):
            snapshot = f"""
**GeoRisk Snapshot**
- Coordinates: {lat:.5f}, {lon:.5f}
- Radius: {buffer_km} km
- Recent Health: {recent_ndvi:.4f if recent_ndvi else 'N/A'}
- 5-Year Average: {baseline_ndvi:.4f if baseline_ndvi else 'N/A'}
- Anomaly Score: {z_score:.4f if z_score else 'N/A'}
- Status: {status_text if z_score else 'Unknown'}
            """
            st.code(snapshot, language="markdown")


if __name__ == "__main__":
    main()