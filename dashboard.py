# dashboard.py — Airbnb Dallas | Pro Dashboard (búsqueda por país + modo claro/oscuro + KPI superhosts + Mapa)
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.optimize import curve_fit
import base64

# ---------------- Config ---------------- #
st.set_page_config(page_title="Airbnb — Pro Dashboard", layout="wide")

ROOT = Path(".")
DEFAULT_CSV = ROOT / "Datos_limpios.csv"
LOGO = ROOT / "airbnb.png"

# ---------------- Utilidades ---------------- #
def _pct_to_float(s):
    if pd.isna(s): return np.nan
    if isinstance(s, (int, float)):
        v = float(s); return v/100.0 if v > 1.5 else v
    s = str(s).strip()
    if s.endswith("%"):
        try: return float(s[:-1]) / 100.0
        except: return np.nan
    try:
        v = float(s); return v/100.0 if v > 1.5 else v
    except: return np.nan

def numeric_cols(df):
    cols = df.select_dtypes(include=["float", "int"]).columns.tolist()
    bad = {"id","lat","lon","latitude","longitude"}  # excluye coords de los análisis numéricos
    return [c for c in cols if all(t not in c.lower() for t in bad)]

def propose_top10_cats(df):
    preferred = [
        "room_type","property_type","bathrooms_text","host_response_time",
        "host_is_superhost","instant_bookable","host_acceptance_rate",
        "host_response_rate","accommodates","bedrooms",
    ]
    cats = [c for c in preferred if c in df.columns]
    if len(cats) < 10:
        for c in df.columns:
            if c in cats: continue
            is_cat = (df[c].dtype.name in ["object","category"]) or (
                pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique(dropna=True) <= 50
            )
            if is_cat and 2 <= df[c].nunique(dropna=True) <= 50 and df[c].isna().mean() < 0.5:
                cats.append(c)
            if len(cats) >= 10: break
    return cats[:10]

# --------- Binarización (igual a tu lógica) --------- #
def apply_binarization(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "instant_bookable" in out.columns:
        if out["instant_bookable"].dtype == object:
            out["instant_bookable"] = out["instant_bookable"].astype(str).str.lower().isin(
                ["t","true","1","yes","y"]
            ).astype(int)
        else:
            out["instant_bookable"] = out["instant_bookable"].astype(int)
    if "host_is_superhost" in out.columns:
        if out["host_is_superhost"].dtype == object:
            out["host_is_superhost"] = out["host_is_superhost"].astype(str).str.lower().isin(
                ["t","true","1","yes","y"]
            ).astype(int)
        else:
            out["host_is_superhost"] = out["host_is_superhost"].astype(int)
    if "room_type" in out.columns:
        out["room_type_binary"] = (out["room_type"] == "Entire home/apt").astype(int)
    if "host_response_time" in out.columns:
        out["response_time_binary"] = (out["host_response_time"].astype(str).str.lower()=="within an hour").astype(int)
    if "availability_365" in out.columns:
        out["availability_binary"] = (pd.to_numeric(out["availability_365"], errors="coerce") > 180).astype(int)
    if "property_type" in out.columns:
        out["property_type_binary"] = (out["property_type"] == "Apartment/Condo").astype(int)

    col_rr = next((c for c in ["host_response_rate","host_response_rate_clean"] if c in out.columns), None)
    if col_rr:
        out["_rr"] = out[col_rr].apply(_pct_to_float)
        out["response_rate_binary"] = (out["_rr"] >= 0.90).astype(int)
        out.drop(columns=["_rr"], inplace=True, errors="ignore")

    col_ar = next((c for c in ["host_acceptance_rate","host_acceptance_rate_clean"] if c in out.columns), None)
    if col_ar:
        out["_ar"] = out[col_ar].apply(_pct_to_float)
        out["acceptance_rate_binary"] = (out["_ar"] >= 0.85).astype(int)
        out.drop(columns=["_ar"], inplace=True, errors="ignore")

    if "price" in out.columns:
        p = pd.to_numeric(out["price"], errors="coerce")
        out["price_binary"] = (p >= p.median(skipna=True)).astype(int)
    if "review_scores_rating" in out.columns:
        r = pd.to_numeric(out["review_scores_rating"], errors="coerce")
        out["rating_binary"] = (r >= 4.5).astype(int)
    return out

BINARY_VARS = [
    "instant_bookable","host_is_superhost","room_type_binary","response_time_binary",
    "availability_binary","property_type_binary","response_rate_binary",
    "acceptance_rate_binary","price_binary","rating_binary",
]

# ---------------- Temas (dark/light) ---------------- #
def set_theme_state():
    if "light_mode" not in st.session_state:
        st.session_state.light_mode = False

AIRBNB_PINKS_DARK = ["#8B5CF6","#60A5FA","#22D3EE","#A78BFA","#F472B6","#34D399","#F59E0B"]
AIRBNB_PINKS_LIGHT = ["#7C3AED","#2563EB","#0891B2","#6D28D9","#DB2777","#059669","#D97706"]

def apply_theme(fig):
    """Aplica el tema coherente según el toggle."""
    if st.session_state.light_mode:
        px.defaults.template = "plotly_white"
        px.defaults.color_discrete_sequence = AIRBNB_PINKS_LIGHT
        fig.update_layout(
            paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
            font=dict(color="#111111"),
            legend_font_color="#111111", legend_title_font_color="#111111",
            xaxis=dict(tickfont=dict(color="#111111"), title_font=dict(color="#111111"), gridcolor="rgba(0,0,0,.08)"),
            yaxis=dict(tickfont=dict(color="#111111"), title_font=dict(color="#111111"), gridcolor="rgba(0,0,0,.08)"),
        )
    else:
        px.defaults.template = "plotly_dark"
        px.defaults.color_discrete_sequence = AIRBNB_PINKS_DARK
        fig.update_layout(
            paper_bgcolor="#0B1220", plot_bgcolor="#0B1220",
            font=dict(color="#E5E7EB"),
            legend_font_color="#E5E7EB", legend_title_font_color="#E5E7EB",
            xaxis=dict(tickfont=dict(color="#CBD5E1"), title_font=dict(color="#E5E7EB"), gridcolor="rgba(226,232,240,.12)"),
            yaxis=dict(tickfont=dict(color="#CBD5E1"), title_font=dict(color="#E5E7EB"), gridcolor="rgba(226,232,240,.12)"),
        )
    return fig

def ensure_titles(fig, title=None, xlab=None, ylab=None):
    tcol = "#111111" if st.session_state.light_mode else "#E5E7EB"
    if title is not None:
        fig.update_layout(title=dict(text=title, x=0.02, xanchor="left", font=dict(color=tcol, size=18)))
    if xlab is not None: fig.update_xaxes(title_text=xlab, title_font=dict(color=tcol))
    if ylab is not None: fig.update_yaxes(title_text=ylab, title_font=dict(color=tcol))
    fig.update_layout(margin=dict(t=60, r=20, l=20, b=40))
    return fig

# ===== Helpers de Mapa =====
def make_map_fig(df, color_col=None, size_col=None, zoom=9, opacity=0.7):
    """
    Crea la figura de mapa usando 'latitude' y 'longitude' (tus nombres exactos).
    Muestra hasta 5000 puntos por rendimiento.
    """
    if "latitude" not in df.columns or "longitude" not in df.columns:
        return None, "No se encontraron columnas 'latitude' y 'longitude' en los datos."

    dfx = df.dropna(subset=["latitude", "longitude"]).copy()
    if len(dfx) > 5000:
        dfx = dfx.sample(5000, random_state=42)

    # Hover enriquecido si existen columnas
    hover_cols = []
    for c in ["name", "price", "review_scores_rating", "room_type", "property_type"]:
        if c in dfx.columns: hover_cols.append(c)

    # Mapbox styles según modo (no requiere token)
    mb_style = "carto-positron" if st.session_state.get("light_mode", False) else "carto-darkmatter"

    fig = px.scatter_mapbox(
        dfx,
        lat="latitude", lon="longitude",
        color=color_col if color_col in dfx.columns else None,
        size=size_col if size_col in dfx.columns else None,
        size_max=18,
        opacity=opacity,
        hover_data=hover_cols,
        zoom=zoom,
        height=620
    )
    fig.update_layout(mapbox_style=mb_style, margin=dict(t=60, r=10, l=10, b=10))
    ensure_titles(fig, "Mapa de listados", "longitud", "latitud")
    return fig, None

# ---------------- Carga de datos ---------------- #
@st.cache_resource
def load_default():
    if not DEFAULT_CSV.exists():
        return pd.DataFrame()
    return pd.read_csv(DEFAULT_CSV)

def combine_sources(default_df, uploads):
    frames = []
    if not default_df.empty:
        frames.append(default_df.assign(_source="Datos_limpios.csv"))
    for f in uploads or []:
        try:
            dfu = pd.read_csv(f)
            frames.append(dfu.assign(_source=f.name))
        except Exception:
            pass
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()

def filter_by_query(df, query):
    if not query: 
        return df, "Todos"
    q = str(query).strip().lower()
    # columnas de posible localización
    candidates = [c for c in df.columns if c.lower() in 
        ["country","pais","country_code","region","state","estado","city","ciudad","market","neighbourhood","neighborhood"]]
    if not candidates:
        return df, "Sin columnas de país/ciudad detectadas"

    mask = pd.Series(False, index=df.index)
    for c in candidates:
        mask = mask | df[c].astype(str).str.lower().str.contains(q, na=False)
    out = df[mask].copy()
    return out, f"Filtro aplicado en: {', '.join(candidates)}"

# ----------------- UI (top + estilos) ----------------- #
set_theme_state()
logo_uri = ""
if LOGO.exists():
    logo_uri = "data:image/png;base64," + base64.b64encode(LOGO.read_bytes()).decode("utf-8")

st.markdown("""
<style>
:root {
  --bg-dark: #0B1220;
  --panel-dark: #0F172A;
  --stroke-dark: rgba(148,163,184,.16);
  --txt-dark: #E5E7EB;
  --muted-dark: #94A3B8;

  --bg-light: #F8FAFC;
  --panel-light: #FFFFFF;
  --stroke-light: rgba(2,6,23,.08);
  --txt-light: #0B1220;
  --muted-light: #475569;
}

/* Tarjetas KPI */
.card {
  border-radius: 14px; padding: 16px; border: 1px solid var(--stroke-dark);
  background: var(--panel-dark);
  transition: transform .18s ease, box-shadow .18s ease;
  position: relative; overflow: hidden;
}
.card:hover { transform: translateY(-4px); box-shadow: 0 14px 30px rgba(0,0,0,.22); }
.card::after {
  content: ""; position: absolute; inset: 0;
  background: linear-gradient(120deg, transparent, rgba(255,255,255,.12), transparent);
  transform: translateX(-120%);
  transition: transform .4s ease;
}
.card:hover::after { transform: translateX(120%); }

.kpi-value { font-size: 28px; font-weight: 800; margin: 6px 0 4px 0; }
.kpi-sub { color: #22c55e; font-size: 12px; font-weight: 600; }

[data-theme="light"] .card{ background: var(--panel-light); border: 1px solid var(--stroke-light);}
[data-theme="light"] .kpi-value{ color: var(--txt-light);}
</style>
""", unsafe_allow_html=True)

# Barra superior (con widgets reales)
top_l, top_c, top_r = st.columns([0.26, 0.48, 0.26])
with top_l:
    st.markdown(
        f"""<div style="display:flex;align-items:center;gap:10px;margin-top:6px">
               {"<img src='"+logo_uri+"' width='22'/>" if logo_uri else ""}
               <div style="font-weight:800">{'Airbnb — Dallas · Pro Dashboard'}</div>
            </div>""",
        unsafe_allow_html=True,
    )
with top_c:
    search_text = st.text_input("Buscar por país/ciudad/estado…", value="", placeholder="México, USA, Dallas, etc.",
                                label_visibility="collapsed")
with top_r:
    st.write("")  # empuja hacia abajo
    st.session_state.light_mode = st.toggle("Modo claro", value=st.session_state.light_mode)

# ----------------- Fuentes de datos (sidebar) ----------------- #
st.sidebar.header("Fuentes de datos")
uploads = st.sidebar.file_uploader("Cargar CSV adicionales", type=["csv"], accept_multiple_files=True)
base_df = load_default()
df_all = combine_sources(base_df, uploads)

if df_all.empty:
    st.error("No hay datos para mostrar. Asegúrate de tener `Datos_limpios.csv` o subir archivos CSV.")
    st.stop()

# Aplicar búsqueda real
df, where_info = filter_by_query(df_all, search_text)

# ----------------- KPIs ----------------- #
colA, colB, colC, colD = st.columns(4)
with colA:
    st.markdown(
        f'<div class="card"><div style="opacity:.8">Listado</div><div class="kpi-value">{len(df):,}</div><div class="kpi-sub">datos filtrados</div></div>',
        unsafe_allow_html=True)

with colB:
    med_price = pd.to_numeric(df.get("price", pd.Series(dtype=float)), errors="coerce").median()
    st.markdown(
        f'<div class="card"><div style="opacity:.8">Precio mediano</div><div class="kpi-value">${0 if np.isnan(med_price) else round(med_price):,}</div><div class="kpi-sub">vs mercado</div></div>',
        unsafe_allow_html=True)

with colC:
    med_rate = pd.to_numeric(df.get("review_scores_rating", pd.Series(dtype=float)), errors="coerce").median()
    st.markdown(
        f'<div class="card"><div style="opacity:.8">Rating mediano</div><div class="kpi-value">{0 if np.isnan(med_rate) else round(med_rate,2)}</div><div class="kpi-sub">últimos 12m</div></div>',
        unsafe_allow_html=True)

with colD:
    # KPI útil: % Superhosts (fallbacks si no está)
    kpi_html = ""
    if "host_is_superhost" in df.columns:
        sup = pd.to_numeric(df["host_is_superhost"], errors="coerce")
        if not np.isin(sup.unique(), [0,1]).all():
            sup = df["host_is_superhost"].astype(str).str.lower().isin(["t","true","1","yes","y"]).astype(int)
        pct = sup.mean()*100 if len(sup) else np.nan
        kpi_html = f"% Superhosts|{0 if np.isnan(pct) else pct:.1f}%|mix actual"
    elif "host_acceptance_rate" in df.columns:
        med_ar = pd.to_numeric(df["host_acceptance_rate"].apply(_pct_to_float), errors="coerce").median()
        kpi_html = f"Aceptación mediana|{0 if np.isnan(med_ar) else med_ar*100:.1f}%|hosts"
    elif "availability_365" in df.columns:
        med_av = pd.to_numeric(df["availability_365"], errors="coerce").median()
        kpi_html = f"Disponibilidad mediana|{0 if np.isnan(med_av) else med_av:.0f} días|anual"
    title, val, sub = kpi_html.split("|")
    st.markdown(f'<div class="card"><div style="opacity:.8">{title}</div><div class="kpi-value">{val}</div><div class="kpi-sub">{sub}</div></div>', unsafe_allow_html=True)

st.caption(where_info)

# ----------------- Tabs de análisis ----------------- #
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏷️ Exploración", "📈 Regresión Lineal", "🧩 Regresión No lineal", "🎯 Logística", "🗺️ Mapa"
])

# ========== Tab 1 ==========
with tab1:
    cats = propose_top10_cats(df)
    left, right = st.columns([0.28, 0.72])
    with left:
        vc = st.selectbox("Variable categórica", options=cats, key="exp_cat")
        show_box = st.checkbox("Boxplot de precio por categoría", True)
    with right:
        frec = (
            df[vc].value_counts(dropna=False).rename_axis("categorias").reset_index(name="frecuencia")
            if vc in df.columns else pd.DataFrame(columns=["categorias","frecuencia"])
        )
        r1c1, r1c2 = st.columns(2)
        with r1c1:
            fig = px.bar(frec, x="categorias", y="frecuencia")
            ensure_titles(fig, "Frecuencia por categoría", "categorías", "frecuencia"); apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)
        with r1c2:
            fig2 = px.pie(frec, names="categorias", values="frecuencia")
            ensure_titles(fig2, "Distribución por categoría"); apply_theme(fig2)
            st.plotly_chart(fig2, use_container_width=True)
        r2c1, r2c2 = st.columns(2)
        with r2c1:
            fig3 = px.pie(frec, names="categorias", values="frecuencia", hole=0.45)
            ensure_titles(fig3, "Dona por categoría"); apply_theme(fig3)
            st.plotly_chart(fig3, use_container_width=True)
        with r2c2:
            fig4 = px.area(frec, x="categorias", y="frecuencia")
            ensure_titles(fig4, "Área acumulada", "categorías", "frecuencia"); apply_theme(fig4)
            st.plotly_chart(fig4, use_container_width=True)

        if show_box and "price" in df.columns:
            st.markdown("")
            figb = px.box(df.dropna(subset=[vc, "price"]), x=vc, y="price", points="outliers")
            ensure_titles(figb, f"Distribución de 'price' por {vc}", vc, "price"); apply_theme(figb)
            st.plotly_chart(figb, use_container_width=True)

# ========== Tab 2 ==========
with tab2:
    nums = numeric_cols(df)
    if len(nums) < 2:
        st.warning("No hay suficientes variables numéricas.")
    else:
        left, right = st.columns([0.28, 0.72])
        with left:
            y_var = st.selectbox("Objetivo (Y)", options=nums, key="lin_y")
            x_var = st.selectbox("Independiente simple (X)", options=[c for c in nums if c != y_var], key="lin_x")
            multi_vars = st.multiselect("Independientes (múltiple)", options=[c for c in nums if c != y_var], key="lin_mult")

            st.markdown("### Correlaciones")
            X = df[[x_var]].dropna(); y = df.loc[X.index, y_var]
            R_simple = np.sqrt(max(LinearRegression().fit(X, y).score(X, y), 0))
            st.info(f"**Simple (R)**: {repr(R_simple)}")
            if multi_vars:
                Xm = df[multi_vars].dropna(); ym = df.loc[Xm.index, y_var]
                Rm = np.sqrt(max(LinearRegression().fit(Xm, ym).score(Xm, ym), 0))
                st.info(f"**Múltiple (R)**: {repr(Rm)}")
            else:
                st.warning("Agrega al menos 1 variable para el modelo múltiple.")

        with right:
            a, b = st.columns(2)
            with a:
                fig = px.scatter(df, x=x_var, y=y_var)
                ensure_titles(fig, "Modelo Lineal Simple", x_var, y_var); apply_theme(fig)
                st.plotly_chart(fig, use_container_width=True)
            with b:
                if multi_vars:
                    tmp = df[[y_var] + multi_vars].dropna()
                    long = tmp.melt(id_vars=y_var, value_vars=multi_vars, var_name="variable", value_name="value")
                    figm = px.scatter(long, x="value", y=y_var, color="variable")
                    ensure_titles(figm, "Modelo Lineal Múltiple", "value", y_var); apply_theme(figm)
                    st.plotly_chart(figm, use_container_width=True)
                else:
                    st.info("Selecciona variables para visualizar (múltiple).")

# ========== Tab 3 ==========
with tab3:
    nums = numeric_cols(df)
    if len(nums) < 2:
        st.warning("No hay suficientes variables numéricas.")
    else:
        left, right = st.columns([0.28, 0.72])
        with left:
            y_nl = st.selectbox("Objetivo (Y)", options=nums, key="nl_y")
            x_nl = st.selectbox("Independiente (X)", options=[c for c in nums if c != y_nl], key="nl_x")
            mtype = st.radio("Modelo", options=["Función cuadrática","Función exponencial","Función logarítmica"])
        with right:
            dfn = df.dropna(subset=[x_nl, y_nl]).copy()
            if mtype == "Función logarítmica":
                dfn = dfn[dfn[x_nl] > 0].copy()

            x = dfn[x_nl].values; y = dfn[y_nl].values

            def fit_nl(x, y, model_name):
                if len(x) < 3: return None, None
                if model_name == "Función cuadrática":
                    def f(z,a,b,c): return a*z**2 + b*z + c
                    p0 = (1.0, 1.0, float(np.nanmean(y)) if len(y)>0 else 0.0)
                elif model_name == "Función exponencial":
                    def f(z,a,b,c): return a*np.exp(-b*z) + c
                    p0 = (1.0, 0.01, float(np.nanmean(y)) if len(y)>0 else 0.0)
                else:
                    def f(z,a,b): return a*np.log(z) + b
                    p0 = (1.0, float(np.nanmean(y)) if len(y)>0 else 0.0)
                try:
                    params,_ = curve_fit(f, x, y, p0=p0, maxfev=20000)
                    yhat = f(x, *params); R2 = r2_score(y, yhat); R = np.sqrt(max(R2,0))
                    return yhat, R
                except: return None, None

            yhat, R = fit_nl(x, y, mtype)
            fig = px.scatter(dfn, x=x_nl, y=y_nl)
            if yhat is not None:
                order = np.argsort(x)
                fig.add_traces(go.Scatter(x=x[order], y=yhat[order], mode="lines", name="Predicción"))
            ensure_titles(fig, f"Modelo No Lineal — {mtype}", x_nl, y_nl); apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)
            st.info(f"**Correlación No lineal (R)**: {repr(R) if R is not None else 'N/D'}")

# ========== Tab 4 ==========
with tab4:
    dfb = apply_binarization(df)
    targets = [c for c in BINARY_VARS if c in dfb.columns]
    if not targets:
        st.error("No se encontraron variables binarias esperadas.")
    else:
        left, right = st.columns([0.28, 0.72])
        with left:
            tgt = st.selectbox("Dependiente (binaria)", options=targets)
            nums_b = dfb.select_dtypes(include=["number"]).columns.tolist()
            Xvars = st.multiselect("Independientes (numéricas)", options=[c for c in nums_b if c != tgt])
        with right:
            if not Xvars:
                st.info("Selecciona al menos una variable para entrenar.")
            else:
                data = dfb[[tgt] + Xvars].dropna().copy()
                X = data[Xvars].apply(pd.to_numeric, errors="coerce").fillna(0.0)
                y = pd.to_numeric(data[tgt], errors="coerce").fillna(0).clip(0,1).astype(int)
                keep = [c for c in X.columns if X[c].nunique() > 1]
                X = X[keep]
                if X.shape[1] == 0:
                    st.error("Las variables seleccionadas tienen varianza cero.")
                else:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.30, random_state=42, stratify=y if y.nunique()==2 else None
                    )
                    scaler = StandardScaler(with_mean=True, with_std=True)
                    X_train_s = scaler.fit_transform(X_train)
                    X_test_s  = scaler.transform(X_test)

                    clf = LogisticRegression(max_iter=1000)
                    clf.fit(X_train_s, y_train)
                    y_pred = clf.predict(X_test_s)

                    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
                    accuracy   = accuracy_score(y_test, y_pred)
                    precision0 = precision_score(y_test, y_pred, pos_label=0)
                    precision1 = precision_score(y_test, y_pred, pos_label=1)
                    sens0      = recall_score(y_test, y_pred, pos_label=0)
                    sens1      = recall_score(y_test, y_pred, pos_label=1)
                    f1macro    = f1_score(y_test, y_pred, average="macro")

                    gL, gR = st.columns([0.58, 0.42])
                    with gL:
                        mat = np.array([[tn, fp], [fn, tp]])
                        fig = go.Figure(data=go.Heatmap(
                            z=mat, x=["Pred 0","Pred 1"], y=["Real 0","Real 1"],
                            colorscale="blues" if st.session_state.light_mode else "Oranges",
                            showscale=True, zmin=0, hoverinfo="z"
                        ))
                        lbl = {(0,0):"TN",(0,1):"FP",(1,0):"FN",(1,1):"TP"}
                        ann=[]
                        for i in range(2):
                            for j in range(2):
                                v = int(mat[i,j])
                                ann.append(dict(x=["Pred 0","Pred 1"][j], y=["Real 0","Real 1"][i],
                                                text=f"{lbl[(i,j)]}: {v}", showarrow=False,
                                                font=dict(color="white" if v > mat.max()/2 else ("#111111" if st.session_state.light_mode else "#0B1220"))))
                        fig.update_layout(annotations=ann, height=520,
                                          title=dict(text="Matriz de confusión", x=0.02, xanchor="left"))
                        apply_theme(fig); st.plotly_chart(fig, use_container_width=True)
                    with gR:
                        st.markdown(
                            f"""
                            <div class="card" style="line-height:1.8">
                              <div style="opacity:.8">Métricas</div>
                              <div class="kpi-value" style="font-size:18px">{accuracy:.3f} exactitud</div>
                              <div class="kpi-sub">F1 (macro): {f1macro:.3f}</div>
                              <div class="kpi-sub">Precisión 0: {precision0:.3f}</div>
                              <div class="kpi-sub">Precisión 1: {precision1:.3f}</div>
                              <div class="kpi-sub">Recall 0: {sens0:.3f}</div>
                              <div class="kpi-sub">Recall 1: {sens1:.3f}</div>
                            </div>
                            """, unsafe_allow_html=True
                        )

# ========== Tab 5: Mapa ==========
with tab5:
    if "latitude" not in df.columns or "longitude" not in df.columns:
        st.error("No se detectaron columnas 'latitude' y 'longitude'.")
    else:
        left, right = st.columns([0.30, 0.70])
        with left:
            st.markdown("#### Opciones del mapa")
            # Color por categórica (si existe)
            cat_cols = [c for c in df.columns if df[c].dtype == "object" and df[c].nunique(dropna=True) <= 30]
            color_default_idx = 0
            if "room_type" in cat_cols:
                color_default_idx = cat_cols.index("room_type") + 1
            color_col = st.selectbox("Color por (categórica)", options=["(ninguno)"] + cat_cols, index=color_default_idx)
            color_col = None if color_col == "(ninguno)" else color_col

            # Tamaño por numérica (si existe)
            num_cols = df.select_dtypes(include=["float", "int"]).columns.tolist()
            size_default_idx = 0
            for cand in ["price", "accommodates"]:
                if cand in num_cols:
                    size_default_idx = num_cols.index(cand) + 1
                    break
            size_col = st.selectbox("Tamaño por (numérica)", options=["(ninguno)"] + num_cols, index=size_default_idx)
            size_col = None if size_col == "(ninguno)" else size_col

            opacity = st.slider("Opacidad de puntos", 0.2, 1.0, 0.7, 0.05)
            zoom = st.slider("Zoom inicial", 3, 15, 9, 1)

            st.caption(f"Mostrando hasta **5,000** puntos por rendimiento. Columnas usadas: **latitude**, **longitude**.")

        with right:
            fig_map, err = make_map_fig(df, color_col=color_col, size_col=size_col, zoom=zoom, opacity=opacity)
            if err:
                st.error(err)
            else:
                apply_theme(fig_map)
                fig_map.update_layout(transition=dict(duration=450, easing="cubic-in-out"))
                st.plotly_chart(fig_map, use_container_width=True)
