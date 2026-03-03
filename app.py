import streamlit as st
import numpy as np
import pandas as pd
import joblib
import folium
from streamlit_folium import st_folium
from catboost import CatBoostRegressor
from sklearn.neighbors import BallTree
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Prices",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg: #0d0f14;
    --surface: #151820;
    --surface2: #1c2030;
    --border: #252a38;
    --accent: #c8ff00;
    --accent2: #00d4ff;
    --text: #e8eaf2;
    --muted: #6b7280;
    --danger: #ff6b6b;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] * {
    color: var(--text) !important;
}

h1, h2, h3 {
    font-family: 'DM Serif Display', serif !important;
    color: var(--text) !important;
}

.metric-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    position: relative;
    overflow: hidden;
}

.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
}

.price-hero {
    background: linear-gradient(135deg, var(--surface2) 0%, #1a2040 100%);
    border: 1px solid var(--border);
    border-radius: 24px;
    padding: 40px;
    text-align: center;
    position: relative;
    overflow: hidden;
}

.price-hero::after {
    content: '';
    position: absolute;
    bottom: -60px; right: -60px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(200,255,0,0.08) 0%, transparent 70%);
    border-radius: 50%;
}

.price-value {
    font-family: 'DM Serif Display', serif;
    font-size: 3.5rem;
    color: var(--accent);
    line-height: 1;
    margin: 8px 0;
}

.price-label {
    font-size: 0.85rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 2px;
}

.range-bar {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 20px 24px;
}

.tag {
    display: inline-block;
    background: rgba(200,255,0,0.1);
    color: var(--accent);
    border: 1px solid rgba(200,255,0,0.2);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.78rem;
    font-weight: 500;
    letter-spacing: 1px;
    text-transform: uppercase;
}

.section-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.4rem;
    color: var(--text);
    margin-bottom: 16px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
}

.stSelectbox > div > div,
.stNumberInput > div > div > input,
.stSlider {
    background-color: var(--surface2) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
}

.stButton > button {
    background: var(--accent) !important;
    color: #0d0f14 !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    font-family: 'DM Sans', sans-serif !important;
    padding: 12px 32px !important;
    width: 100% !important;
    font-size: 1rem !important;
    letter-spacing: 0.5px;
    transition: all 0.2s;
}

.stButton > button:hover {
    background: #d4ff33 !important;
    transform: translateY(-1px);
    box-shadow: 0 8px 24px rgba(200,255,0,0.25) !important;
}

div[data-testid="stCheckbox"] label {
    color: var(--text) !important;
}

.sidebar-section {
    margin-bottom: 20px;
    padding-bottom: 20px;
    border-bottom: 1px solid var(--border);
}

.insight-chip {
    background: rgba(0,212,255,0.08);
    border: 1px solid rgba(0,212,255,0.2);
    border-radius: 10px;
    padding: 10px 16px;
    font-size: 0.85rem;
    color: var(--accent2);
    margin: 6px 0;
}

footer {visibility: hidden;}
#MainMenu {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ── Load artifacts ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    base = Path(__file__).parent / "artifacts"
    assets = joblib.load(base / "assets.joblib")
    model = CatBoostRegressor()
    model.load_model(str(base / "catboost_model.cbm"))
    return model, assets

model, assets = load_artifacts()

EARTH_R = 6371000

def min_dist_m(lat, lon, poi_array):
    if poi_array is None or len(poi_array) == 0:
        return 99999
    tree = BallTree(np.deg2rad(poi_array), metric='haversine')
    X = np.deg2rad([[lat, lon]])
    dist, _ = tree.query(X, k=1)
    return float(dist[0, 0]) * EARTH_R

def count_within(lat, lon, poi_array, radius_m):
    if poi_array is None or len(poi_array) == 0:
        return 0
    tree = BallTree(np.deg2rad(poi_array), metric='haversine')
    X = np.deg2rad([[lat, lon]])
    ind = tree.query_radius(X, r=radius_m / EARTH_R)
    return int(len(ind[0]))


def build_features(inputs: dict) -> pd.DataFrame:
    a = assets
    city  = inputs["city"]
    nbh   = inputs["neighborhood"]

    # Coordenadas
    lat, lon = a["nbh_to_latlon"].get(
        (city, nbh),
        a["city_to_latlon"].get(city, (-23.55, -46.63))
    )

    # Target encodings
    nbh_enc  = a["neighborhood_target_enc"].get(nbh,  a["global_median"])
    city_enc = a["city_target_enc"].get(city, a["global_median"])

    # Geo-clusters
    geo_vals = {}
    for k, km in a["geo_models"].items():
        geo_vals[f"geo_k{k}"] = int(km.predict([[lat, lon]])[0])

    # Features externas
    dist_metro = min_dist_m(lat, lon, a.get("metro_latlon"))
    dist_trem  = min_dist_m(lat, lon, a.get("trem_latlon"))
    poi_latlon = a.get("poi_latlon", {})

    poi_counts = {}
    for name, arr in poi_latlon.items():
        poi_counts[f"n_{name}_500m"]  = count_within(lat, lon, arr, 500)
        poi_counts[f"n_{name}_1000m"] = count_within(lat, lon, arr, 1000)

    has_bank_500  = int(poi_counts.get("n_bank_500m", 0) > 0)
    has_park_500  = int(poi_counts.get("n_park_500m", 0) > 0)

    # Tipo do imóvel → OHE
    imvl_cats = ["casas", "casas-de-condominio", "casas-de-vila",
                 "flat", "quitinetes", "studio"]
    imvl_ohe  = {f"imvl_type_{c}": int(inputs["imvl_type"] == c) for c in imvl_cats}

    # Zona → OHE
    zone_cats = ["Zona Central", "Zona Norte", "Zona Sul"]
    zone_ohe  = {f"zone_{z}": int(inputs.get("zone", "") == z) for z in zone_cats}

    area = inputs["usable_area"]
    row = {
        "listing.address.point.lon": lon,
        "listing.backyard":          int(inputs.get("backyard", False)),
        "listing.barbgrill":         int(inputs.get("barbgrill", False)),
        "listing.bathrooms":         inputs["bathrooms"],
        "listing.bedrooms":          inputs["bedrooms"],
        "listing.pool":              int(inputs.get("pool", False)),
        "listing.suites":            inputs["suites"],
        "listing.usableAreas":       area,
        "iptu_mensal":               inputs.get("iptu_mensal", 0),
        "condo_ratio":               (inputs.get("condo_fee", 0) / inputs.get("price_est", 2000))
                                     if inputs.get("condo_fee", 0) > 50 else 0,
        "total_cost":                inputs.get("condo_fee", 0) if inputs.get("condo_fee", 0) > 50 else 0,
        "log_usableAreas":           np.log1p(area),
        "preco_por_m2":              0,  # será preenchido após previsão
        "neighborhood_target_enc":   nbh_enc,
        "city_target_enc":           city_enc,
        **zone_ohe,
        "has_parking":               int(inputs.get("parking_spaces", 0) > 0),
        **imvl_ohe,
        **{k: geo_vals.get(k, 0) for k in ["geo_k10", "geo_k100"]},
        "log_dist_metro":            np.log1p(dist_metro),
        "log_dist_trem":             np.log1p(dist_trem),
        "n_supermarket_1000m":       poi_counts.get("n_supermarket_1000m", 0),
        "n_hospital_500m":           poi_counts.get("n_hospital_500m", 0),
        "n_restaurant_1000m":        poi_counts.get("n_restaurant_1000m", 0),
        "n_pharmacy_500m":           poi_counts.get("n_pharmacy_500m", 0),
        "n_pharmacy_1000m":          poi_counts.get("n_pharmacy_1000m", 0),
        "n_bank_500m":               poi_counts.get("n_bank_500m", 0),
        "n_bank_1000m":              poi_counts.get("n_bank_1000m", 0),
        "has_bank_500m":             has_bank_500,
        "has_park_500m":             has_park_500,
        "walkability_score":         0,
    }

    df = pd.DataFrame([row])
    # garantir ordem correta
    for col in assets["selected_features"]:
        if col not in df.columns:
            df[col] = assets["feature_medians"].get(col, 0)
    df = df[assets["selected_features"]].fillna(0)
    return df, lat, lon


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 8px 0 24px'>
        <div style='font-family: DM Serif Display, serif; font-size:1.8rem; color:#c8ff00'>
            Smart Prices
        </div>
        <div style='font-size:0.8rem; color:#6b7280; letter-spacing:1px; text-transform:uppercase'>
            Previsão de Aluguel · Brasil
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Localização
    st.markdown("**📍 Localização**")
    city_list = sorted(assets.get("city_list", [])) or ["São Paulo"]
    default_city_idx = city_list.index("São Paulo") if "São Paulo" in city_list else 0
    city = st.selectbox("Cidade", city_list, index=default_city_idx)

    nbh_list = ["(Não informado)"] + sorted(assets.get("neighborhood_list", []))
    neighborhood = st.selectbox("Bairro", nbh_list)
    if neighborhood == "(Não informado)":
        neighborhood = city
    
    zone = st.selectbox("Zona", ["(Não informado)", "Zona Sul", "Zona Norte",
                                  "Zona Oeste", "Zona Central", "Sudeste", "Bairros"])

    st.markdown("---")

    # Características
    st.markdown("**🏠 Características**")
    imvl_options = {
        "Apartamento": "apartamentos",
        "Casa": "casas",
        "Casa de Condomínio": "casas-de-condominio",
        "Cobertura": "cobertura",
        "Flat": "flat",
        "Quitinete": "quitinetes",
        "Studio": "studio",
        "Casa de Vila": "casas-de-vila",
    }
    imvl_label = st.selectbox("Tipo do Imóvel", list(imvl_options.keys()))
    imvl_type  = imvl_options[imvl_label]

    col1, col2 = st.columns(2)
    with col1:
        bedrooms  = st.number_input("Quartos",  min_value=0, max_value=10, value=2)
        bathrooms = st.number_input("Banheiros", min_value=0, max_value=10, value=1)
    with col2:
        suites    = st.number_input("Suítes",   min_value=0, max_value=10, value=0)
        parking   = st.number_input("Vagas",    min_value=0, max_value=10, value=1)

    area = st.slider("Área Útil (m²)", min_value=10, max_value=500, value=70)

    st.markdown("---")

    # Financeiro
    st.markdown("**💰 Financeiro**")
    condo_fee  = st.number_input("Condomínio Mensal (R$)", min_value=0, value=0, step=50)
    iptu_mensal = st.number_input("IPTU Mensal (R$)", min_value=0, value=0, step=10)

    st.markdown("---")

    # Amenidades
    st.markdown("**✨ Amenidades**")
    col1, col2 = st.columns(2)
    with col1:
        pool      = st.checkbox("🏊 Piscina")
        gym       = st.checkbox("🏋️ Academia")
        barbgrill = st.checkbox("🍖 Churrasqueira")
        garden    = st.checkbox("🌿 Jardim")
    with col2:
        backyard  = st.checkbox("🌳 Quintal")
        furnished = st.checkbox("🛋️ Mobiliado")
        sauna     = st.checkbox("🧖 Sauna")
        playground = st.checkbox("🛝 Playground")

    st.markdown("---")
    predict_btn = st.button("🔍 Calcular Aluguel Ideal")


# ── Main area ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style='margin-bottom: 32px'>
    <h1 style='font-size:2.4rem; margin-bottom:4px'>
        Qual é o aluguel ideal?
    </h1>
    <p style='color:#6b7280; font-size:1rem'>
        Preencha as características do imóvel e descubra o valor justo de mercado.
    </p>
</div>
""", unsafe_allow_html=True)

if not predict_btn:
    # Estado inicial
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class='metric-card'>
            <div class='price-label'>Modelo</div>
            <div style='font-size:1.6rem; font-weight:600; margin:8px 0; color:#c8ff00'>CatBoost</div>
            <div style='font-size:0.8rem; color:#6b7280'>Gradient Boosting</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='metric-card'>
            <div class='price-label'>MAE Médio</div>
            <div style='font-size:1.6rem; font-weight:600; margin:8px 0; color:#00d4ff'>R$ 56</div>
            <div style='font-size:0.8rem; color:#6b7280'>Erro médio absoluto</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class='metric-card'>
            <div class='price-label'>R² Score</div>
            <div style='font-size:1.6rem; font-weight:600; margin:8px 0; color:#c8ff00'>99.93%</div>
            <div style='font-size:0.8rem; color:#6b7280'>Variância explicada</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background:#151820; border:1px solid #252a38; border-radius:16px; padding:32px; text-align:center; color:#6b7280'>
        <div style='font-size:3rem; margin-bottom:12px'>🏠</div>
        <div style='font-size:1.1rem; color:#9ca3af'>
            Configure as características do imóvel na barra lateral<br>e clique em <strong style='color:#c8ff00'>Calcular Aluguel Ideal</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)

else:
    inputs = {
        "city":         city,
        "neighborhood": neighborhood,
        "zone":         zone,
        "imvl_type":    imvl_type,
        "bedrooms":     bedrooms,
        "bathrooms":    bathrooms,
        "suites":       suites,
        "parking_spaces": parking,
        "usable_area":  area,
        "condo_fee":    condo_fee,
        "iptu_mensal":  iptu_mensal,
        "pool":         pool,
        "gym":          gym,
        "barbgrill":    barbgrill,
        "garden":       garden,
        "backyard":     backyard,
        "furnished":    furnished,
        "sauna":        sauna,
        "playground":   playground,
        "price_est":    2000,
    }

    with st.spinner("Calculando..."):
        feat_df, lat, lon = build_features(inputs)
        pred_log  = model.predict(feat_df)[0]
        pred_price = float(np.expm1(pred_log))
        pred_min   = pred_price * 0.88
        pred_max   = pred_price * 1.12

    # ── Resultado principal ────────────────────────────────────────────────────
    col_main, col_side = st.columns([3, 2], gap="large")

    with col_main:
        faixa = ("econômico" if pred_price < 1500 else
                 "médio" if pred_price < 4000 else
                 "alto" if pred_price < 10000 else "luxo")
        faixa_color = {"econômico": "#4ade80", "médio": "#00d4ff",
                       "alto": "#f59e0b", "luxo": "#c8ff00"}[faixa]

        st.markdown(f"""
        <div class='price-hero'>
            <div class='price-label'>Aluguel estimado</div>
            <div class='price-value'>R$ {pred_price:,.0f}</div>
            <div style='margin-top:12px'>
                <span class='tag' style='background:rgba({",".join(str(int(faixa_color.lstrip("#")[i:i+2], 16)) for i in (0,2,4))},0.15);
                      color:{faixa_color}; border-color:rgba({",".join(str(int(faixa_color.lstrip("#")[i:i+2], 16)) for i in (0,2,4))},0.3)'>
                    {faixa.upper()}
                </span>
            </div>
            <div style='margin-top:20px; color:#6b7280; font-size:0.85rem'>
                por mês · {city}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Faixa de preço
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>📊 Faixa de Mercado</div>", unsafe_allow_html=True)

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pred_price,
            number={"prefix": "R$ ", "font": {"size": 28, "color": "#c8ff00",
                    "family": "DM Serif Display"}},
            gauge={
                "axis": {"range": [pred_min * 0.7, pred_max * 1.3],
                         "tickcolor": "#6b7280", "tickfont": {"color": "#6b7280", "size": 11}},
                "bar": {"color": "#c8ff00", "thickness": 0.25},
                "bgcolor": "#1c2030",
                "bordercolor": "#252a38",
                "steps": [
                    {"range": [pred_min * 0.7, pred_min], "color": "#1c2030"},
                    {"range": [pred_min, pred_max], "color": "rgba(200,255,0,0.12)"},
                    {"range": [pred_max, pred_max * 1.3], "color": "#1c2030"},
                ],
                "threshold": {
                    "line": {"color": "#00d4ff", "width": 3},
                    "thickness": 0.8,
                    "value": pred_price,
                },
            },
        ))
        fig_gauge.update_layout(
            paper_bgcolor="#0d0f14", plot_bgcolor="#0d0f14",
            font={"color": "#e8eaf2"},
            height=260, margin=dict(t=20, b=10, l=30, r=30),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown(f"""
            <div class='range-bar' style='text-align:center'>
                <div style='font-size:0.75rem; color:#6b7280; text-transform:uppercase; letter-spacing:1px'>Mínimo</div>
                <div style='font-size:1.3rem; font-weight:600; color:#4ade80; margin-top:4px'>R$ {pred_min:,.0f}</div>
            </div>""", unsafe_allow_html=True)
        with col_b:
            st.markdown(f"""
            <div class='range-bar' style='text-align:center; border-color:#c8ff00'>
                <div style='font-size:0.75rem; color:#6b7280; text-transform:uppercase; letter-spacing:1px'>Estimado</div>
                <div style='font-size:1.3rem; font-weight:600; color:#c8ff00; margin-top:4px'>R$ {pred_price:,.0f}</div>
            </div>""", unsafe_allow_html=True)
        with col_c:
            st.markdown(f"""
            <div class='range-bar' style='text-align:center'>
                <div style='font-size:0.75rem; color:#6b7280; text-transform:uppercase; letter-spacing:1px'>Máximo</div>
                <div style='font-size:1.3rem; font-weight:600; color:#f59e0b; margin-top:4px'>R$ {pred_max:,.0f}</div>
            </div>""", unsafe_allow_html=True)

    with col_side:
        # Mapa
        st.markdown("<div class='section-title'>📍 Localização</div>", unsafe_allow_html=True)
        m = folium.Map(location=[lat, lon], zoom_start=13,
                       tiles="CartoDB dark_matter")
        folium.CircleMarker(
            location=[lat, lon], radius=12,
            color="#c8ff00", fill=True, fill_color="#c8ff00", fill_opacity=0.8,
            popup=f"<b>{neighborhood}</b><br>R$ {pred_price:,.0f}/mês",
        ).add_to(m)
        folium.Circle(
            location=[lat, lon], radius=1000,
            color="#c8ff00", fill=True, fill_color="#c8ff00",
            fill_opacity=0.05, weight=1, dash_array="5",
        ).add_to(m)
        st_folium(m, height=320, use_container_width=True)

        # Insights
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>💡 Insights</div>", unsafe_allow_html=True)

        preco_m2 = pred_price / area
        st.markdown(f"""
        <div class='insight-chip'>💰 R$ {preco_m2:,.0f}/m² · {area}m² de área útil</div>
        <div class='insight-chip'>🏘️ Bairro: {neighborhood}</div>
        <div class='insight-chip'>🏠 Tipo: {imvl_label} · {bedrooms} quartos</div>
        """, unsafe_allow_html=True)

        if condo_fee > 0:
            total = pred_price + condo_fee + iptu_mensal
            st.markdown(f"""
            <div style='margin-top:12px; background:#1c2030; border:1px solid #252a38;
                        border-radius:12px; padding:16px'>
                <div style='font-size:0.75rem; color:#6b7280; text-transform:uppercase; letter-spacing:1px'>
                    Custo Total Estimado
                </div>
                <div style='font-size:1.5rem; font-weight:600; color:#00d4ff; margin-top:4px'>
                    R$ {total:,.0f}<span style='font-size:0.85rem; color:#6b7280'>/mês</span>
                </div>
                <div style='font-size:0.78rem; color:#6b7280; margin-top:4px'>
                    Aluguel + Condomínio + IPTU
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Comparação com mercado ─────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>📈 Comparação com o Mercado</div>",
                unsafe_allow_html=True)

    market_data = {
        "Econômico\n(até R$1.500)":   1100,
        "Médio\n(R$1.5k–4k)":         2500,
        "Alto\n(R$4k–10k)":            6500,
        "Luxo\n(>R$10k)":             15000,
        f"📍 {neighborhood[:15]}":     pred_price,
    }

    colors = ["#2d3748", "#2d3748", "#2d3748", "#2d3748", "#c8ff00"]
    fig_bar = go.Figure(go.Bar(
        x=list(market_data.keys()),
        y=list(market_data.values()),
        marker_color=colors,
        marker_line_color=["#4a5568"] * 4 + ["#c8ff00"],
        marker_line_width=1,
        text=[f"R$ {v:,.0f}" for v in market_data.values()],
        textposition="outside",
        textfont={"color": "#e8eaf2", "size": 12},
    ))
    fig_bar.update_layout(
        paper_bgcolor="#0d0f14", plot_bgcolor="#0d0f14",
        font={"color": "#e8eaf2", "family": "DM Sans"},
        xaxis={"gridcolor": "#1c2030", "tickfont": {"size": 11}},
        yaxis={"gridcolor": "#252a38", "tickprefix": "R$ ",
               "tickfont": {"size": 11}},
        height=340, margin=dict(t=40, b=20, l=20, r=20),
        showlegend=False,
    )
    st.plotly_chart(fig_bar, use_container_width=True)
