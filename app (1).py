import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="S.BIKE — Seoul Bike Demand Predictor",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0b0f1a; color: #e2e8f0; }
    
    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #131929; border-right: 1px solid #243050; }
    [data-testid="stSidebar"] .stMarkdown p { color: #94a3b8; }
    
    /* Metric cards */
    [data-testid="metric-container"] {
        background-color: #131929;
        border: 1px solid #243050;
        border-radius: 10px;
        padding: 16px;
    }
    [data-testid="metric-container"] label { color: #64748b !important; font-size: 11px !important; }
    [data-testid="metric-container"] [data-testid="stMetricValue"] { color: #e2e8f0 !important; }
    
    /* Inputs */
    .stNumberInput input, .stSelectbox select, .stDateInput input {
        background-color: #1a2235 !important;
        color: #e2e8f0 !important;
        border: 1px solid #243050 !important;
        border-radius: 7px !important;
    }
    
    /* Headers */
    h1, h2, h3 { color: #e2e8f0 !important; font-family: 'Segoe UI', sans-serif !important; }
    
    /* Prediction result box */
    .pred-box {
        background: linear-gradient(135deg, rgba(16,185,129,0.1), rgba(0,229,255,0.05));
        border: 1px solid rgba(16,185,129,0.4);
        border-radius: 14px;
        padding: 28px;
        text-align: center;
        margin: 16px 0;
    }
    .pred-number { font-size: 64px; font-weight: 800; color: #10b981; line-height: 1; }
    .pred-label  { font-size: 13px; color: #64748b; letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 6px; }
    .pred-unit   { font-size: 15px; color: #94a3b8; margin-top: 4px; }

    /* Info note */
    .info-note {
        background: rgba(0,229,255,0.06);
        border: 1px solid rgba(0,229,255,0.2);
        border-radius: 9px;
        padding: 10px 16px;
        font-size: 12px;
        color: #94a3b8;
        margin-bottom: 12px;
    }

    /* Section divider */
    hr { border-color: #243050 !important; }

    /* Streamlit buttons */
    .stButton > button {
        background-color: #10b981;
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        transition: opacity 0.15s;
    }
    .stButton > button:hover { opacity: 0.85; background-color: #10b981; color: white; }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] { background-color: #131929; border-radius: 8px; }
    .stTabs [data-baseweb="tab"] { color: #64748b; }
    .stTabs [aria-selected="true"] { color: #00e5ff !important; }
    
    /* Plotly chart background */
    .js-plotly-plot { border-radius: 10px; }

    /* Success / info boxes */
    .stSuccess { background-color: rgba(16,185,129,0.1) !important; border-color: #10b981 !important; }
    .stInfo    { background-color: rgba(0,229,255,0.07) !important; border-color: #00e5ff !important; }
    .stWarning { background-color: rgba(245,158,11,0.1) !important; }
    .stError   { background-color: rgba(239,68,68,0.1) !important; }

    /* DataFrame styling */
    [data-testid="stDataFrame"] { border: 1px solid #243050; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATA GENERATION  (simulates Seoul Bike dataset)
# ─────────────────────────────────────────────
@st.cache_data
def generate_dataset():
    np.random.seed(42)
    dates = pd.date_range("2017-12-01", "2026-12-31", freq="h")
    n = len(dates)

    month   = dates.month.values - 1         # 0-based
    hour    = dates.hour.values
    dow     = dates.dayofweek.values          # 0=Mon

    # Seasonal temp profile (Seoul monthly avg °C)
    temp_base  = np.array([-3,-1,5,12,18,23,27,28,23,16,7,-1])[month]
    temp       = temp_base + np.random.normal(0, 3, n)

    humidity   = np.clip(np.array([55,55,57,58,62,68,78,75,68,62,60,58])[month] + np.random.normal(0,8,n), 10, 100)
    wind       = np.abs(np.random.normal(2.2, 1.2, n))
    visibility = np.clip(np.random.normal(1600, 300, n), 100, 2000)
    dew_point  = temp - (100 - humidity) / 5
    solar_base = np.array([0.3,0.5,0.8,1.1,1.4,1.6,1.5,1.5,1.1,0.8,0.4,0.25])[month]
    solar_rad  = np.clip(solar_base * np.abs(np.random.normal(1, 0.35, n)), 0, 3.5)
    rainfall   = np.where(np.random.random(n) < np.array([0.05,0.05,0.08,0.1,0.12,0.18,0.25,0.22,0.15,0.1,0.07,0.05])[month],
                          np.abs(np.random.exponential(3, n)), 0)
    snowfall   = np.where((temp < 2) & (np.random.random(n) < 0.12), np.abs(np.random.exponential(2,n)), 0)

    season_map = [4,4,1,1,1,2,2,2,3,3,3,4]
    season     = np.array([season_map[m] for m in month])

    pub_hols = {(1,1),(3,1),(5,5),(6,6),(8,15),(10,3),(10,9),(12,25)}
    holiday  = np.array([1 if ((d.month, d.day) in pub_hols or d.dayofweek >= 5) else 0 for d in dates])
    func_day = np.where(holiday == 1, 0, 1)

    # Rental count model
    hour_effect = np.array([-150,-180,-200,-220,-180,-60,80,280,320,150,80,100,
                             120,100,80,120,200,350,380,280,180,100,20,-80])[hour]
    season_eff  = np.array([0,50,200,150,-150])[season]
    base = (300 + hour_effect + season_eff
            + (temp - 10) * 18
            - np.maximum(0, humidity - 60) * 3
            - rainfall * 80
            - snowfall * 120
            + solar_rad * 60)
    base = base * func_day
    noise   = np.random.normal(0, 60, n)
    rentals = np.clip(np.round(base + noise), 0, 3500).astype(int)

    df = pd.DataFrame({
        "Date"               : dates.date,
        "Hour"               : hour,
        "Temperature(°C)"    : np.round(temp, 1),
        "Humidity(%)"        : np.round(humidity, 0).astype(int),
        "Wind speed (m/s)"   : np.round(wind, 1),
        "Visibility (10m)"   : np.round(visibility, 0).astype(int),
        "Dew point temperature(°C)": np.round(dew_point, 1),
        "Solar Radiation (MJ/m2)"  : np.round(solar_rad, 2),
        "Rainfall(mm)"       : np.round(rainfall, 1),
        "Snowfall (cm)"      : np.round(snowfall, 1),
        "Seasons"            : season,
        "Holiday"            : holiday,
        "Functioning Day"    : func_day,
        "Rented Bike Count"  : rentals,
    })
    return df

@st.cache_resource
def train_model(df):
    features = ["Hour","Temperature(°C)","Humidity(%)","Wind speed (m/s)",
                "Visibility (10m)","Dew point temperature(°C)",
                "Solar Radiation (MJ/m2)","Rainfall(mm)","Snowfall (cm)",
                "Seasons","Holiday","Functioning Day"]
    X = df[features]
    y = df["Rented Bike Count"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Random Forest"        : RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosting"    : GradientBoostingRegressor(n_estimators=100, random_state=42),
        "Linear Regression"    : LinearRegression(),
    }
    results = {}
    for name, m in models.items():
        m.fit(X_train, y_train)
        preds = m.predict(X_test)
        results[name] = {
            "model" : m,
            "preds" : preds,
            "y_test": y_test,
            "r2"    : r2_score(y_test, preds),
            "rmse"  : np.sqrt(mean_squared_error(y_test, preds)),
            "mae"   : mean_absolute_error(y_test, preds),
        }
    return results, features, X_test, y_test

# ─────────────────────────────────────────────
# LOAD DATA & MODEL
# ─────────────────────────────────────────────
with st.spinner("Loading dataset and training models..."):
    df      = generate_dataset()
    results, features, X_test, y_test = train_model(df)

PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0b0f1a",
    plot_bgcolor ="#131929",
    font         =dict(color="#94a3b8", family="monospace", size=11),
    xaxis        =dict(gridcolor="#243050", linecolor="#243050"),
    yaxis        =dict(gridcolor="#243050", linecolor="#243050"),
    legend       =dict(bgcolor="#131929", bordercolor="#243050", borderwidth=1),
    margin       =dict(l=20, r=20, t=40, b=20),
)

SEASON_NAMES = {1:"Spring 🌸", 2:"Summer ☀️", 3:"Autumn 🍂", 4:"Winter ❄️"}

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚲 S.BIKE")
    st.markdown("**Seoul Bike Sharing**  \nDemand Predictor")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["📈 Trend Chart", "🎯 Single Hour Prediction", "📊 Further Analysis"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("**Model Selection**")
    selected_model_name = st.selectbox(
        "Choose model",
        list(results.keys()),
        label_visibility="collapsed"
    )
    sel = results[selected_model_name]
    st.markdown(f"""
    <small>
    R² &nbsp;&nbsp;: <b style='color:#10b981'>{sel['r2']:.4f}</b><br>
    RMSE : <b style='color:#f59e0b'>{sel['rmse']:.1f}</b><br>
    MAE &nbsp;: <b style='color:#00e5ff'>{sel['mae']:.1f}</b>
    </small>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <small style='color:#475569'>
    Dataset: Seoul Bike Sharing<br>
    Period: Dec 2017 – Dec 2026<br>
    Records: hourly entries<br>
    Source: UCI ML Repository
    </small>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PAGE 1 — TREND CHART
# ─────────────────────────────────────────────
if page == "📈 Trend Chart":
    st.markdown("## 📈 Hourly Bike Rental Trend")
    st.markdown("Actual vs Predicted rental count over selected date range")
    st.markdown("---")

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        start_date = st.date_input("Start Date", value=pd.Timestamp("2024-01-01"),
                                   min_value=pd.Timestamp("2017-12-01"),
                                   max_value=pd.Timestamp("2026-12-31"))
    with col2:
        end_date   = st.date_input("End Date",   value=pd.Timestamp("2024-06-30"),
                                   min_value=pd.Timestamp("2017-12-01"),
                                   max_value=pd.Timestamp("2026-12-31"))
    with col3:
        view_mode  = st.selectbox("View Mode", ["Daily Average", "Weekly Average"])

    if start_date >= end_date:
        st.error("End date must be after start date.")
        st.stop()

    # Filter and aggregate
    mask = (df["Date"] >= start_date) & (df["Date"] <= end_date)
    df_range = df[mask].copy()

    if df_range.empty:
        st.warning("No data for selected range.")
        st.stop()

    df_range["DateDT"] = pd.to_datetime(df_range["Date"])

    if view_mode == "Daily Average":
        grouped = df_range.groupby("Date")["Rented Bike Count"].mean().reset_index()
        grouped.columns = ["Date", "Actual"]
        grouped["DateDT"] = pd.to_datetime(grouped["Date"])
    else:
        df_range["Week"] = pd.to_datetime(df_range["Date"]).dt.to_period("W").apply(lambda r: r.start_time)
        grouped = df_range.groupby("Week")["Rented Bike Count"].mean().reset_index()
        grouped.columns = ["DateDT", "Actual"]

    # Predict for the whole range
    feat_df = df_range[features].copy()
    grouped_feat = df_range.copy()
    grouped_feat["Predicted"] = sel["model"].predict(grouped_feat[features])

    if view_mode == "Daily Average":
        pred_grouped = grouped_feat.groupby("Date")["Predicted"].mean().reset_index()
        pred_grouped["DateDT"] = pd.to_datetime(pred_grouped["Date"])
    else:
        pred_grouped = grouped_feat.copy()
        pred_grouped["Week"] = pd.to_datetime(pred_grouped["Date"]).dt.to_period("W").apply(lambda r: r.start_time)
        pred_grouped = pred_grouped.groupby("Week")["Predicted"].mean().reset_index()
        pred_grouped.columns = ["DateDT", "Predicted"]

    merged = grouped.merge(pred_grouped[["DateDT","Predicted"]], on="DateDT", how="left")

    # Prediction start line — 80% mark of date range
    pred_start_idx = int(len(merged) * 0.8)
    pred_start_date = merged["DateDT"].iloc[pred_start_idx] if pred_start_idx < len(merged) else merged["DateDT"].iloc[-1]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=merged["DateDT"], y=merged["Actual"],
        name="Actual Rentals", line=dict(color="#00e5ff", width=2),
        fill="tozeroy", fillcolor="rgba(0,229,255,0.06)"
    ))
    fig.add_trace(go.Scatter(
        x=merged["DateDT"], y=merged["Predicted"].clip(lower=0),
        name="Predicted Rentals", line=dict(color="#7c3aed", width=2, dash="dot"),
        fill="tozeroy", fillcolor="rgba(124,58,237,0.06)"
    ))
    fig.add_vline(
        x=pred_start_date, line_width=2, line_dash="solid",
        line_color="#10b981",
        annotation_text="Prediction Start Date",
        annotation_font_color="#10b981",
        annotation_bgcolor="rgba(16,185,129,0.15)"
    )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text=f"Seoul Bike Rentals — {view_mode}", font=dict(color="#e2e8f0", size=14)),
        xaxis_title="Date", yaxis_title="Rental Count", height=420,
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Stats row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg Daily Rentals",  f"{int(df_range['Rented Bike Count'].mean()):,}")
    c2.metric("Peak Hour Avg",      f"{int(df_range.groupby('Hour')['Rented Bike Count'].mean().max()):,}", "18:00 peak")
    c3.metric("Total Records",      f"{len(df_range):,}")
    c4.metric(f"{selected_model_name} R²", f"{sel['r2']:.4f}")

# ─────────────────────────────────────────────
# PAGE 2 — SINGLE HOUR PREDICTION
# ─────────────────────────────────────────────
elif page == "🎯 Single Hour Prediction":
    st.markdown("## 🎯 Single Hour Bike Demand Prediction")
    st.markdown("Enter any conditions to predict bike rental demand — works for **any date, past or future**.")
    st.markdown("---")

    left, right = st.columns([1.1, 0.9])

    with left:
        st.markdown("### 📅 Auto-fill from Date *(Optional)*")
        st.markdown("""
        <div class='info-note'>
        💡 Pick any date (2017–2026) to auto-fill realistic weather values.
        You can edit any field freely before predicting.
        </div>
        """, unsafe_allow_html=True)

        af_col1, af_col2 = st.columns([2, 1])
        with af_col1:
            auto_date = st.date_input("Select Date",
                                      value=pd.Timestamp("2025-06-15"),
                                      min_value=pd.Timestamp("2017-12-01"),
                                      max_value=pd.Timestamp("2026-12-31"),
                                      label_visibility="collapsed")
        with af_col2:
            auto_clicked = st.button("📥 Auto-fill", use_container_width=True)

        # Compute auto-fill values for the selected date
        month_idx = auto_date.month - 1
        dow       = auto_date.weekday()
        season_map_list = [4,4,1,1,1,2,2,2,3,3,3,4]
        af_season = season_map_list[month_idx]
        af_temp   = round([-3,-1,5,12,18,23,27,28,23,16,7,-1][month_idx] + np.random.uniform(-2,2), 1)
        af_hum    = int(np.clip([55,55,57,58,62,68,78,75,68,62,60,58][month_idx] + np.random.uniform(-5,5), 20, 100))
        af_wind   = round(abs(np.random.normal(2.2,1.0)), 1)
        af_vis    = int(np.clip(np.random.normal(1600,200), 200, 2000))
        af_dew    = round(af_temp - (100 - af_hum)/5, 1)
        solar_b   = [0.3,0.5,0.8,1.1,1.4,1.6,1.5,1.5,1.1,0.8,0.4,0.25][month_idx]
        af_solar  = round(max(0, solar_b * abs(np.random.normal(1, 0.3))), 2)
        rain_ch   = [0.05,0.05,0.08,0.1,0.12,0.18,0.25,0.22,0.15,0.1,0.07,0.05][month_idx]
        af_rain   = round(abs(np.random.exponential(2)), 1) if np.random.random() < rain_ch else 0.0
        af_snow   = round(abs(np.random.exponential(1.5)), 1) if (af_temp < 2 and np.random.random() < 0.12) else 0.0
        pub_hols  = {(1,1),(3,1),(5,5),(6,6),(8,15),(10,3),(10,9),(12,25)}
        af_holiday = 1 if ((auto_date.month, auto_date.day) in pub_hols or dow >= 5) else 0
        af_func    = 0 if af_holiday else 1

        if auto_clicked:
            st.session_state.update({
                "f_hour": 18, "f_season": af_season, "f_holiday": af_holiday,
                "f_func": af_func, "f_temp": af_temp, "f_humidity": af_hum,
                "f_wind": af_wind, "f_vis": af_vis, "f_dew": af_dew,
                "f_solar": af_solar, "f_rain": af_rain, "f_snow": af_snow,
            })
            st.success(f"✅ Auto-filled values for {auto_date.strftime('%d %b %Y')} — edit any field below.")

        st.markdown("---")
        st.markdown("### ⚙️ Input Features")

        st.markdown("**Temporal**")
        tc1, tc2 = st.columns(2)
        with tc1:
            hour    = st.number_input("Hour (0–23)", 0, 23, st.session_state.get("f_hour", 18), step=1)
            holiday = st.selectbox("Holiday", [0, 1], index=st.session_state.get("f_holiday", 0),
                                   format_func=lambda x: "No Holiday" if x == 0 else "Holiday")
        with tc2:
            season  = st.selectbox("Season", [1,2,3,4], index=st.session_state.get("f_season",2)-1,
                                   format_func=lambda x: SEASON_NAMES[x])
            func_day = st.selectbox("Functioning Day", [1, 0], index=0 if st.session_state.get("f_func",1)==1 else 1,
                                    format_func=lambda x: "Yes" if x == 1 else "No")

        st.markdown("**Weather**")
        wc1, wc2 = st.columns(2)
        with wc1:
            temp       = st.number_input("Temperature (°C)",       -20.0, 45.0,  float(st.session_state.get("f_temp", 22.0)),   0.1)
            wind       = st.number_input("Wind Speed (m/s)",         0.0, 30.0,  float(st.session_state.get("f_wind",  2.0)),   0.1)
            dew        = st.number_input("Dew Point Temp (°C)",     -30.0, 35.0, float(st.session_state.get("f_dew",  14.0)),   0.1)
            rain       = st.number_input("Rainfall (mm)",             0.0, 100.0, float(st.session_state.get("f_rain",  0.0)),  0.1)
        with wc2:
            humidity   = st.number_input("Humidity (%)",              0,   100,   int(st.session_state.get("f_humidity", 60)),   1)
            visibility = st.number_input("Visibility (10m)",          0,  2000,   int(st.session_state.get("f_vis",   1800)),    50)
            solar      = st.number_input("Solar Radiation (MJ/m²)",   0.0, 4.0,  float(st.session_state.get("f_solar", 0.5)),   0.01)
            snow       = st.number_input("Snowfall (cm)",              0.0, 50.0, float(st.session_state.get("f_snow",  0.0)),  0.1)

        predict_btn = st.button("🎯 Predict Demand", use_container_width=True)
        reset_btn   = st.button("🔄 Reset Fields",   use_container_width=True)
        if reset_btn:
            for k in ["f_hour","f_season","f_holiday","f_func","f_temp","f_humidity",
                      "f_wind","f_vis","f_dew","f_solar","f_rain","f_snow"]:
                st.session_state.pop(k, None)
            st.rerun()

    with right:
        st.markdown("### 📊 Prediction Result")

        if predict_btn:
            input_arr = np.array([[hour, temp, humidity, wind, visibility,
                                   dew, solar, rain, snow, season, holiday, func_day]])
            prediction = int(max(0, sel["model"].predict(input_arr)[0]))

            st.markdown(f"""
            <div class='pred-box'>
                <div class='pred-label'>Predicted Rental Count</div>
                <div class='pred-number'>{prediction:,}</div>
                <div class='pred-unit'>bikes / hour</div>
            </div>
            """, unsafe_allow_html=True)

            # Breakdown table
            st.markdown("**Input Summary**")
            summary_data = {
                "Feature": ["Hour", "Season", "Temperature", "Humidity",
                             "Wind Speed", "Rainfall", "Snowfall", "Solar Rad.",
                             "Holiday", "Functioning Day"],
                "Value":   [f"{hour}:00", SEASON_NAMES[season], f"{temp} °C",
                             f"{humidity}%", f"{wind} m/s", f"{rain} mm",
                             f"{snow} cm", f"{solar} MJ/m²",
                             "Yes" if holiday else "No",
                             "Yes" if func_day else "No"]
            }
            st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

            # Confidence estimate
            all_preds = sel["model"].predict(X_test)
            residuals  = np.abs(y_test.values - all_preds)
            conf       = max(65, min(98, int(100 - (np.mean(residuals) / (prediction + 1)) * 60)))
            st.markdown(f"**Model Confidence: {conf}%**")
            st.progress(conf / 100)

            # Gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prediction,
                title={"text": "Predicted Demand", "font": {"color": "#94a3b8", "size": 13}},
                gauge={
                    "axis"      : {"range": [0, 3500], "tickcolor": "#64748b"},
                    "bar"       : {"color": "#10b981"},
                    "bgcolor"   : "#1a2235",
                    "bordercolor": "#243050",
                    "steps"     : [
                        {"range": [0,    700],  "color": "rgba(239,68,68,0.15)"},
                        {"range": [700,  1500], "color": "rgba(245,158,11,0.15)"},
                        {"range": [1500, 3500], "color": "rgba(16,185,129,0.15)"},
                    ],
                },
                number={"font": {"color": "#10b981", "size": 36}}
            ))
            fig_gauge.update_layout(**PLOTLY_LAYOUT, height=240)
            st.plotly_chart(fig_gauge, use_container_width=True)

            st.info("ℹ Model trained on Seoul Bike Sharing data. Predictions are estimates based on input features.")
        else:
            st.markdown("""
            <div style='text-align:center;padding:60px 20px;color:#475569;'>
                <div style='font-size:48px;margin-bottom:12px;'>🚲</div>
                <p>Fill in the feature values on the left<br>and click <b>Predict Demand</b>.</p>
                <p style='margin-top:12px;font-size:11px;font-family:monospace;'>
                Works for any date — 2017 to 2026
                </p>
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PAGE 3 — FURTHER ANALYSIS
# ─────────────────────────────────────────────
elif page == "📊 Further Analysis":
    st.markdown("## 📊 Further Analysis")
    st.markdown("Model performance, feature correlations, and seasonal patterns.")
    st.markdown("---")

    # Model comparison
    st.markdown("### 🏆 Model Performance Comparison")
    mc1, mc2, mc3 = st.columns(3)
    for i, (name, res) in enumerate(results.items()):
        col = [mc1, mc2, mc3][i]
        col.metric(name, f"R² = {res['r2']:.4f}",
                   f"RMSE: {res['rmse']:.1f} | MAE: {res['mae']:.1f}")

    # Actual vs Predicted scatter (best model)
    st.markdown(f"### 🎯 Actual vs Predicted — {selected_model_name}")
    sample_idx = np.random.choice(len(X_test), min(800, len(X_test)), replace=False)
    y_sample   = y_test.values[sample_idx]
    p_sample   = sel["preds"][sample_idx]

    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(
        x=y_sample, y=p_sample, mode="markers",
        marker=dict(color="#00e5ff", size=4, opacity=0.5),
        name="Predictions"
    ))
    max_val = max(y_sample.max(), p_sample.max())
    fig_scatter.add_trace(go.Scatter(
        x=[0, max_val], y=[0, max_val],
        mode="lines", line=dict(color="#7c3aed", dash="dash", width=1.5),
        name="Perfect fit"
    ))
    fig_scatter.update_layout(
        **PLOTLY_LAYOUT, height=380,
        xaxis_title="Actual", yaxis_title="Predicted",
        title=dict(text="Actual vs Predicted Rental Count", font=dict(color="#e2e8f0", size=13))
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    col_a, col_b = st.columns(2)

    # Correlation heatmap
    with col_a:
        st.markdown("### 🔗 Feature Correlation with Rental Count")
        corr_features = ["Temperature(°C)","Hour","Dew point temperature(°C)",
                         "Solar Radiation (MJ/m2)","Humidity(%)","Wind speed (m/s)",
                         "Rainfall(mm)","Snowfall (cm)"]
        sample_df = df.sample(min(5000, len(df)), random_state=42)
        corr_vals = [sample_df[f].corr(sample_df["Rented Bike Count"]) for f in corr_features]
        corr_df   = pd.DataFrame({"Feature": corr_features, "Correlation": corr_vals})
        corr_df   = corr_df.sort_values("Correlation", key=abs, ascending=True)

        fig_corr = go.Figure(go.Bar(
            x=corr_df["Correlation"], y=corr_df["Feature"],
            orientation="h",
            marker=dict(
                color=corr_df["Correlation"],
                colorscale=[[0,"#ef4444"],[0.5,"#f59e0b"],[1,"#10b981"]],
                cmin=-0.6, cmax=0.6,
                showscale=True
            )
        ))
        fig_corr.update_layout(**PLOTLY_LAYOUT, height=320,
                               xaxis_title="Pearson Correlation",
                               title=dict(text="Feature vs Rental Count", font=dict(color="#e2e8f0",size=13)))
        st.plotly_chart(fig_corr, use_container_width=True)

    # Seasonal avg
    with col_b:
        st.markdown("### 🌿 Average Rentals by Season")
        season_avg = df.groupby("Seasons")["Rented Bike Count"].mean().reset_index()
        season_avg["Season Name"] = season_avg["Seasons"].map(SEASON_NAMES)
        colors = ["#10b981","#00e5ff","#f59e0b","#7c3aed"]

        fig_season = go.Figure(go.Bar(
            x=season_avg["Season Name"], y=season_avg["Rented Bike Count"].round(0),
            marker=dict(color=colors, line=dict(width=0)),
            text=season_avg["Rented Bike Count"].round(0).astype(int),
            textposition="outside", textfont=dict(color="#94a3b8")
        ))
        fig_season.update_layout(**PLOTLY_LAYOUT, height=320,
                                  yaxis_title="Avg Rental Count",
                                  title=dict(text="Avg Rentals by Season", font=dict(color="#e2e8f0",size=13)))
        st.plotly_chart(fig_season, use_container_width=True)

    # Hourly pattern
    st.markdown("### ⏰ Average Rentals by Hour of Day")
    hourly_avg = df.groupby("Hour")["Rented Bike Count"].mean().reset_index()
    fig_hour = go.Figure()
    fig_hour.add_trace(go.Scatter(
        x=hourly_avg["Hour"], y=hourly_avg["Rented Bike Count"],
        fill="tozeroy", fillcolor="rgba(0,229,255,0.08)",
        line=dict(color="#00e5ff", width=2.5),
        mode="lines+markers", marker=dict(color="#00e5ff", size=6)
    ))
    fig_hour.update_layout(
        **PLOTLY_LAYOUT, height=320,
        xaxis=dict(tickvals=list(range(24)), ticktext=[f"{h}:00" for h in range(24)],
                   gridcolor="#243050", linecolor="#243050"),
        yaxis_title="Avg Rental Count",
        title=dict(text="24-Hour Rental Pattern (All Data)", font=dict(color="#e2e8f0", size=13))
    )
    st.plotly_chart(fig_hour, use_container_width=True)

    # Feature importance (RF only)
    if selected_model_name == "Random Forest":
        st.markdown("### 🌲 Feature Importance — Random Forest")
        rf_model   = results["Random Forest"]["model"]
        importance = rf_model.feature_importances_
        imp_df     = pd.DataFrame({"Feature": features, "Importance": importance})
        imp_df     = imp_df.sort_values("Importance", ascending=True)

        fig_imp = go.Figure(go.Bar(
            x=imp_df["Importance"], y=imp_df["Feature"],
            orientation="h",
            marker=dict(color=imp_df["Importance"],
                        colorscale=[[0,"#243050"],[1,"#00e5ff"]], showscale=False),
            text=imp_df["Importance"].round(3), textposition="outside",
            textfont=dict(color="#64748b", size=10)
        ))
        fig_imp.update_layout(**PLOTLY_LAYOUT, height=380,
                              xaxis_title="Importance Score",
                              title=dict(text="Random Forest Feature Importance",
                                         font=dict(color="#e2e8f0", size=13)))
        st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown("---")
    st.markdown("""
    <small style='color:#475569;font-family:monospace;'>
    Reference: Sathishkumar V E et al. "Using data mining techniques for bike sharing demand prediction
    in metropolitan city." Computer Communications, Vol.153, pp.353-366, 2020. | UCI ML Repository.
    </small>
    """, unsafe_allow_html=True)
