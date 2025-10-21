# ============================================
# DASHBOARD Airbnb — Dallas (según plantilla)
# - Extracción de características
# - Regresión Lineal (en app)
# - Regresión No Lineal (cuadrática / exponencial / logarítmica)
# - Regresión Logística (binarizaciones del cuaderno + leyenda estilizada)
# ============================================

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

st.set_page_config(page_title="AIRBNB — Dallas", layout="wide")

# ---------------- Rutas ----------------
ROOT = Path(".")
DATA_CSV = ROOT / "Datos_limpios.csv"

# ---------------- Helpers ----------------
def load_csv_safe(path, **kwargs):
    try:
        if path.exists():
            return pd.read_csv(path, **kwargs)
    except Exception as e:
        st.error(f"Error leyendo {path.name}: {e}")
    return None

def get_numeric_columns(df):
    cols = df.select_dtypes(include=["float", "int"]).columns.tolist()
    bad = ["id", "lat", "lon", "latitude", "longitude"]
    return [c for c in cols if all(t not in c.lower() for t in bad)]

@st.cache_resource
def load_base_df():
    if not DATA_CSV.exists():
        st.warning("No se encontró 'Datos_limpios.csv' en la carpeta actual.")
        return pd.DataFrame()
    df = pd.read_csv(DATA_CSV)

    # Filtrar Dallas si hay columna city/ciudad
    city_cols = [c for c in df.columns if c.lower() in ["city", "ciudad"]]
    if city_cols:
        ccol = city_cols[0]
        mask = df[ccol].astype(str).str.lower() == "dallas"
        if mask.any():
            df = df[mask].copy()
    return df

df_base = load_base_df()

def propose_top10_cats(df):
    preferred = [
        "room_type","property_type","bathrooms_text","host_response_time",
        "host_is_superhost","instant_bookable","host_acceptance_rate",
        "host_response_rate","accommodates","bedrooms",
    ]
    cats = [c for c in preferred if c in df.columns]
    if len(cats) < 10:
        for c in df.columns:
            if c in cats:
                continue
            is_cat = (df[c].dtype.name in ["object", "category"]) or (
                pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique(dropna=True) <= 50
            )
            if is_cat and 2 <= df[c].nunique(dropna=True) <= 50 and df[c].isna().mean() < 0.5:
                cats.append(c)
            if len(cats) >= 10:
                break
    return cats[:10]

Lista_cats = propose_top10_cats(df_base) if not df_base.empty else []

# ---------------- Binarización (tus reglas) ----------------
def _pct_to_float(s):
    """Convierte '90%', '0.9', 0.9 o 90 a float [0,1]."""
    if pd.isna(s):
        return np.nan
    if isinstance(s, (int, float)):
        v = float(s)
        return v/100.0 if v > 1.5 else v
    s = str(s).strip()
    if s.endswith('%'):
        try:
            return float(s[:-1]) / 100.0
        except:
            return np.nan
    try:
        v = float(s);  return v/100.0 if v > 1.5 else v
    except:
        return np.nan

def apply_binarization(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica las reglas de tu notebook para generar variables dicotómicas.
    Dejamos 'instant_bookable' y 'host_is_superhost' tal cual (0/1).
    """
    out = df.copy()

    # Ya binarias (normalizamos a 0/1 si vienen como texto)
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

    # 1) room_type_binary: Entire home/apt = 1, otros = 0
    if "room_type" in out.columns:
        out["room_type_binary"] = (out["room_type"] == "Entire home/apt").astype(int)

    # 2) response_time_binary: within an hour = 1, otros = 0
    if "host_response_time" in out.columns:
        out["response_time_binary"] = (out["host_response_time"].astype(str).str.lower() == "within an hour").astype(int)

    # 3) availability_binary: availability_365 > 180 = 1
    if "availability_365" in out.columns:
        out["availability_binary"] = (pd.to_numeric(out["availability_365"], errors="coerce") > 180).astype(int)

    # 4) property_type_binary: Apartment/Condo = 1
    if "property_type" in out.columns:
        out["property_type_binary"] = (out["property_type"] == "Apartment/Condo").astype(int)

    # 5) response_rate_binary: host_response_rate ≥ 90%
    col_rr = None
    for c in ["host_response_rate", "host_response_rate_clean"]:
        if c in out.columns:
            col_rr = c; break
    if col_rr:
        out["_rr_clean_"] = out[col_rr].apply(_pct_to_float)
        out["response_rate_binary"] = (out["_rr_clean_"] >= 0.90).astype(int)
        out.drop(columns=["_rr_clean_"], inplace=True, errors="ignore")

    # 6) acceptance_rate_binary: host_acceptance_rate ≥ 85%
    col_ar = None
    for c in ["host_acceptance_rate", "host_acceptance_rate_clean"]:
        if c in out.columns:
            col_ar = c; break
    if col_ar:
        out["_ar_clean_"] = out[col_ar].apply(_pct_to_float)
        out["acceptance_rate_binary"] = (out["_ar_clean_"] >= 0.85).astype(int)
        out.drop(columns=["_ar_clean_"], inplace=True, errors="ignore")

    # 7) price_binary: ≥ mediana
    if "price" in out.columns:
        p = pd.to_numeric(out["price"], errors="coerce")
        med = p.median(skipna=True)
        out["price_binary"] = (p >= med).astype(int)

    # 8) rating_binary (target en el cuaderno): review_scores_rating ≥ 4.5
    if "review_scores_rating" in out.columns:
        r = pd.to_numeric(out["review_scores_rating"], errors="coerce")
        out["rating_binary"] = (r >= 4.5).astype(int)

    return out

# Leyendas (se muestran en CUERPO en una caja estilizada)
BIN_EXPLANATIONS = {
    "instant_bookable":       "1 = reserva instantánea; 0 = requiere aprobación.",
    "host_is_superhost":      "1 = Superhost; 0 = no Superhost.",
    "room_type_binary":       "1 = \"Entire home/apt\"; 0 = otros (Private room, Shared room).",
    "response_time_binary":   "1 = Respuesta rápida (within an hour); 0 = más lenta.",
    "availability_binary":    "1 = Alta disponibilidad (> 180 días); 0 = ≤ 180.",
    "property_type_binary":   "1 = \"Apartment/Condo\"; 0 = otros tipos.",
    "response_rate_binary":   "1 = Alta tasa de respuesta (≥ 90%); 0 = < 90%.",
    "acceptance_rate_binary": "1 = Alta tasa de aceptación (≥ 85%); 0 = < 85%.",
    "price_binary":           "1 = Precio alto (≥ mediana); 0 = bajo.",
    "rating_binary":          "1 = Calificación alta (≥ 4.5); 0 = baja.",
}

def render_legend_box(items):
    """items: lista de strings 'Variable: explicación'"""
    if not items:
        return
    html = """
    <div style="
        border: 1px solid rgba(148,163,184,0.35);
        background: rgba(30,41,59,0.35);
        border-radius: 10px;
        padding: 14px 16px;
        margin: 8px 0 18px 0;
    ">
        <div style="font-weight:700; font-size:16px; color:#e5e7eb; margin-bottom:6px;">
            Leyenda de variables dicotómicas
        </div>
        <ul style="margin:0; padding-left:18px; color:#cbd5e1;">
            {}
        </ul>
    </div>
    """
    lis = "".join([f"<li style='margin:4px 0;'>{st._sanitize_html(i)}</li>" for i in items])
    st.markdown(html.format(lis), unsafe_allow_html=True)

# ===== Helpers de TÍTULOS (simples y sin errores) =====
def ensure_titles(fig, title: str = None, xlab: str = None, ylab: str = None):
    """Fuerza título y nombres de ejes para que siempre se vean."""
    if title is not None:
        fig.update_layout(title=dict(text=title, x=0.02, xanchor="left", font=dict(color="#111111", size=18)))
    if xlab is not None:
        fig.update_xaxes(title_text=xlab, title_font=dict(color="#111111"))
    if ylab is not None:
        fig.update_yaxes(title_text=ylab, title_font=dict(color="#111111"))
    # margen suficiente para que el título no se corte
    fig.update_layout(margin=dict(t=60, r=20, l=20, b=40))
    return fig

def theme_fig(fig):
    """Pequeño refuerzo de estilo (no toca propiedades problemáticas)."""
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#111111"),
                      template="plotly_white")
    return fig

# ---------------- Sidebar ----------------
st.sidebar.title("AIRBNB — DALLAS")
View = st.sidebar.selectbox(
    "Tipo de Análisis",
    [
        "Extracción de Características",
        "Regresión Lineal",
        "Regresión No Lineal",
        "Regresión Logística (Resultados)",
    ],
)

# --------------- VISTA 1: Extracción -----------------
if View == "Extracción de Características":
    st.title("Extracción de Características — Dallas (Top 10 categóricas)")
    if df_base.empty or not Lista_cats:
        st.warning("Carga 'Datos_limpios.csv' y verifica que existan variables categóricas.")
    else:
        Variable_Cat = st.sidebar.selectbox("Variable categórica", options=Lista_cats)

        frec = (
            df_base[Variable_Cat]
            .value_counts(dropna=False)
            .rename_axis("categorias")
            .reset_index(name="frecuencia")
        )

        c1, c2 = st.columns(2)
        with c1:
            fig1 = px.bar(frec, x="categorias", y="frecuencia")
            fig1 = ensure_titles(fig1, "Frecuencia por categoría", "categorías", "frecuencia")
            fig1 = theme_fig(fig1)
            st.plotly_chart(fig1, use_container_width=True)

        with c2:
            fig2 = px.pie(frec, names="categorias", values="frecuencia")
            fig2 = ensure_titles(fig2, "Frecuencia por categoría")
            fig2 = theme_fig(fig2)
            st.plotly_chart(fig2, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            fig3 = px.pie(frec, names="categorias", values="frecuencia", hole=0.4)
            fig3 = ensure_titles(fig3, "Frecuencia por categoría")
            fig3 = theme_fig(fig3)
            st.plotly_chart(fig3, use_container_width=True)
        with c4:
            fig4 = px.area(frec, x="categorias", y="frecuencia")
            fig4 = ensure_titles(fig4, "Frecuencia por categoría", "categorías", "frecuencia")
            fig4 = theme_fig(fig4)
            st.plotly_chart(fig4, use_container_width=True)

        if "price" in df_base.columns and pd.api.types.is_numeric_dtype(df_base["price"]):
            fig_box = px.box(df_base.dropna(subset=[Variable_Cat, "price"]),
                             x=Variable_Cat, y="price", points="outliers")
            fig_box = ensure_titles(fig_box, f"Distribución de 'price' por {Variable_Cat}", Variable_Cat, "price")
            fig_box = theme_fig(fig_box)
            st.plotly_chart(fig_box, use_container_width=True)

# ----------- VISTA 2: Regresión Lineal (plantilla) -----------
if View == "Regresión Lineal":
    st.title("Regresión Lineal")

    if df_base.empty:
        st.warning("Carga 'Datos_limpios.csv' para usar esta vista.")
    else:
        Lista_num = get_numeric_columns(df_base)
        if len(Lista_num) < 2:
            st.warning("No hay suficientes variables numéricas.")
        else:
            Variable_y = st.sidebar.selectbox("Variable objetivo (Y)", options=Lista_num)
            Variable_x = st.sidebar.selectbox(
                "Variable independiente del modelo simple (X)",
                options=[c for c in Lista_num if c != Variable_y],
            )

            cTopA, cTopB = st.columns(2)

            # —— Correlación Lineal Simple
            with cTopA:
                X = df_base[[Variable_x]].dropna()
                y = df_base.loc[X.index, Variable_y]
                model = LinearRegression().fit(X, y)
                R_simple = np.sqrt(max(model.score(X, y), 0))
                st.subheader("Correlación Lineal Simple")
                st.write(repr(R_simple))

            # —— Correlación Lineal Múltiple
            with cTopB:
                Variables_x = st.sidebar.multiselect(
                    "Variables independientes del modelo múltiple (X)",
                    options=[c for c in Lista_num if c != Variable_y],
                )
                st.subheader("Correlación Lineal Múltiple")
                if len(Variables_x) == 0:
                    st.info("Selecciona al menos una variable para el modelo múltiple.")
                else:
                    X_M = df_base[Variables_x].dropna()
                    y_M = df_base.loc[X_M.index, Variable_y]
                    model_M = LinearRegression().fit(X_M, y_M)
                    Rm = np.sqrt(max(model_M.score(X_M, y_M), 0))
                    st.write(repr(Rm))

            # —— Gráficos
            cA, cB = st.columns(2)

            with cA:
                st.subheader("Modelo Lineal Simple")
                fig5 = px.scatter(df_base, x=Variable_x, y=Variable_y)
                fig5 = ensure_titles(fig5, "Modelo Lineal Simple", Variable_x, Variable_y)
                fig5 = theme_fig(fig5)
                st.plotly_chart(fig5, use_container_width=True)

            # ======= CORRECCIÓN 1: tipografía negra en gráfico múltiple =======
            with cB:
                st.subheader("Modelo Lineal Múltiple")
                if Variables_x and len(Variables_x) > 0:
                    tmp = df_base[[Variable_y] + Variables_x].dropna()
                    long = tmp.melt(
                        id_vars=Variable_y,
                        value_vars=Variables_x,
                        var_name="variable",
                        value_name="value"
                    )

                    fig6 = px.scatter(long, x="value", y=Variable_y, color="variable")
                    fig6 = ensure_titles(fig6, "Modelo Lineal Múltiple", "value", Variable_y)
                    fig6 = theme_fig(fig6)

                    # Forzar negro en leyenda (título e ítems) y en ejes
                    fig6.update_layout(
                        legend_font_color="#111111",
                        legend_title_font_color="#111111"
                    )
                    fig6.update_xaxes(
                        tickfont=dict(color="#111111"),
                        title_font=dict(color="#111111")
                    )
                    fig6.update_yaxes(
                        tickfont=dict(color="#111111"),
                        title_font=dict(color="#111111")
                    )

                    st.plotly_chart(fig6, use_container_width=True)
                else:
                    st.info("Selecciona variables para visualizar (múltiple).")

# ----------- VISTA 3: Regresión No Lineal -----------
if View == "Regresión No Lineal":
    st.title("Regresión No Lineal")

    if df_base.empty:
        st.warning("Carga 'Datos_limpios.csv' para usar esta vista.")
    else:
        Lista_num = get_numeric_columns(df_base)
        if len(Lista_num) < 2:
            st.warning("No hay suficientes variables numéricas.")
        else:
            Variable_y = st.sidebar.selectbox("Variable objetivo (Y)", options=Lista_num)
            Variable_x = st.sidebar.selectbox(
                "Variable independiente del modelo No lineal (X)",
                options=[c for c in Lista_num if c != Variable_y],
            )
            Lista_mod = ["Función cuadrática", "Función exponencial", "Función logarítmica"]
            Modelo = st.sidebar.selectbox("Modelos No Lineales", options=Lista_mod)

            df_nl = df_base.dropna(subset=[Variable_x, Variable_y]).copy()
            if Modelo == "Función logarítmica":
                df_nl = df_nl[df_nl[Variable_x] > 0].copy()
                if df_nl.empty:
                    st.warning("Para el modelo logarítmico se requieren valores de X > 0.")

            x = df_nl[Variable_x].values
            y = df_nl[Variable_y].values

            def fit_nl(x, y, model_name):
                if len(x) < 3:
                    return None, None
                if model_name == "Función cuadrática":
                    def f(x, a, b, c): return a*x**2 + b*x + c
                    p0 = (1.0, 1.0, float(np.nanmean(y)) if len(y)>0 else 0.0)
                elif model_name == "Función exponencial":
                    def f(x, a, b, c): return a*np.exp(-b*x) + c
                    p0 = (1.0, 0.01, float(np.nanmean(y)) if len(y)>0 else 0.0)
                else:  # logarítmica
                    def f(x, a, b): return a*np.log(x) + b
                    p0 = (1.0, float(np.nanmean(y)) if len(y)>0 else 0.0)
                try:
                    params, _ = curve_fit(f, x, y, p0=p0, maxfev=20000)
                    y_pred = f(x, *params)
                    R2 = r2_score(y, y_pred)
                    R = np.sqrt(max(R2, 0))
                    return y_pred, R
                except Exception:
                    return None, None

            y_pred, R = fit_nl(x, y, Modelo)

            st.subheader("Correlación No lineal")
            st.write(repr(R) if R is not None else "N/D")

            st.subheader(f"Modelo No Lineal — {Modelo}")
            fig = px.scatter(df_nl, x=Variable_x, y=Variable_y)
            if y_pred is not None:
                order = np.argsort(x)
                fig.add_traces(go.Scatter(x=x[order], y=y_pred[order], mode="lines", name="Predicción"))
            fig = ensure_titles(fig, f"Modelo No Lineal — {Modelo}", Variable_x, Variable_y)
            fig = theme_fig(fig)

            # ======= CORRECCIÓN 2: tipografía negra en no lineal (leyenda y ejes) =======
            fig.update_layout(
                legend_font_color="#111111",
                legend_title_font_color="#111111"
            )
            fig.update_xaxes(
                tickfont=dict(color="#111111"),
                title_font=dict(color="#111111")
            )
            fig.update_yaxes(
                tickfont=dict(color="#111111"),
                title_font=dict(color="#111111")
            )

            st.plotly_chart(fig, use_container_width=True)

# ---------- VISTA 4: Logística (target binario; X = solo cuantitativas) ----------
BINARY_VARS = [
    "instant_bookable",
    "host_is_superhost",
    "room_type_binary",
    "response_time_binary",
    "availability_binary",
    "property_type_binary",
    "response_rate_binary",
    "acceptance_rate_binary",
    "price_binary",
    "rating_binary",
]

BIN_EXPLANATIONS = {
    "instant_bookable":       "1 = reserva instantánea; 0 = requiere aprobación.",
    "host_is_superhost":      "1 = Superhost; 0 = no Superhost.",
    "room_type_binary":       "1 = \"Entire home/apt\"; 0 = otros (Private room, Shared room).",
    "response_time_binary":   "1 = Respuesta rápida (within an hour); 0 = más lenta.",
    "availability_binary":    "1 = Alta disponibilidad (> 180 días); 0 = ≤ 180.",
    "property_type_binary":   "1 = \"Apartment/Condo\"; 0 = otros tipos.",
    "response_rate_binary":   "1 = Alta tasa de respuesta (≥ 90%); 0 = < 90%.",
    "acceptance_rate_binary": "1 = Alta tasa de aceptación (≥ 85%); 0 = < 85%.",
    "price_binary":           "1 = Precio alto (≥ mediana); 0 = bajo.",
    "rating_binary":          "1 = Calificación alta (≥ 4.5); 0 = baja.",
}

def render_legend_box(items):
    if not items: return
    html = """
    <div style="
        border: 1px solid rgba(148,163,184,0.35);
        background: rgba(30,41,59,0.35);
        border-radius: 10px;
        padding: 14px 16px;
        margin: 8px 0 18px 0;
    ">
        <div style="font-weight:700; font-size:16px; color:#e5e7eb; margin-bottom:6px;">
            Leyenda de variables dicotómicas
        </div>
        <ul style="margin:0; padding-left:18px; color:#cbd5e1;">
            {}
        </ul>
    </div>
    """
    lis = "".join([f"<li style='margin:4px 0;'><b>{k}</b>: {v}</li>" for k, v in items])
    st.markdown(html.format(lis), unsafe_allow_html=True)

if View == "Regresión Logística (Resultados)":
    st.title("Regresión Logística")

    if df_base.empty:
        st.warning("Carga 'Datos_limpios.csv'.")
    else:
        # 1) Binarizaciones
        df_bin = apply_binarization(df_base)

        # 2) Target: solo binarias disponibles
        avail_targets = [c for c in BINARY_VARS if c in df_bin.columns]
        if len(avail_targets) == 0:
            st.error("No se encontraron las variables binarias esperadas. Revisa apply_binarization().")
        else:
            default_target = "rating_binary" if "rating_binary" in avail_targets else avail_targets[0]
            target = st.sidebar.selectbox(
                "Variable dependiente (binaria)",
                options=avail_targets,
                index=avail_targets.index(default_target),
            )

            # 3) Features: solo cuantitativas
            numeric_cols = df_bin.select_dtypes(include=["number"]).columns.tolist()
            feature_options = sorted([c for c in numeric_cols if c != target])
            X_vars = st.sidebar.multiselect(
                "Variables independientes (solo cuantitativas)",
                options=feature_options,
                default=[]
            )

            # Leyenda
            legend_items = []
            if target in BIN_EXPLANATIONS:
                legend_items.append((target, BIN_EXPLANATIONS[target]))
            for v in X_vars:
                if v in BIN_EXPLANATIONS:
                    legend_items.append((v, BIN_EXPLANATIONS[v]))
            render_legend_box(legend_items)

            # 4) Entrenamiento
            if len(X_vars) == 0:
                st.info("Selecciona al menos una variable cuantitativa para entrenar el modelo.")
            else:
                data = df_bin[[target] + X_vars].dropna().copy()
                X = data[X_vars].apply(pd.to_numeric, errors="coerce").fillna(0.0)
                y = pd.to_numeric(data[target], errors="coerce").fillna(0).clip(0, 1).astype(int)

                keep = [c for c in X.columns if X[c].nunique() > 1]
                X = X[keep]
                if X.shape[1] == 0:
                    st.error("Las variables seleccionadas tienen varianza cero o quedaron inválidas. Elige otras.")
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

                    # Matriz de confusión
                    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

                    # Métricas
                    accuracy   = accuracy_score(y_test, y_pred)
                    precision0 = precision_score(y_test, y_pred, pos_label=0)
                    precision1 = precision_score(y_test, y_pred, pos_label=1)
                    sens0      = recall_score(y_test, y_pred, pos_label=0)
                    sens1      = recall_score(y_test, y_pred, pos_label=1)
                    f1macro    = f1_score(y_test, y_pred, average="macro")

                    left, right = st.columns([0.58, 0.42])

                    with left:
                        st.subheader("Matriz de confusión")
                        mat = np.array([[tn, fp], [fn, tp]])
                        fig = go.Figure(data=go.Heatmap(
                            z=mat, x=["Pred 0","Pred 1"], y=["Real 0","Real 1"],
                            colorscale="Oranges", showscale=True, zmin=0, hoverinfo="z"
                        ))
                        lbl = {(0,0):"TN",(0,1):"FP",(1,0):"FN",(1,1):"TP"}
                        ann=[]
                        for i in range(2):
                            for j in range(2):
                                v = int(mat[i,j])
                                ann.append(dict(x=["Pred 0","Pred 1"][j], y=["Real 0","Real 1"][i],
                                                text=f"{lbl[(i,j)]}: {v}",
                                                showarrow=False,
                                                font=dict(color="white" if v > mat.max()/2 else "black")))
                        fig.update_layout(annotations=ann, height=520, margin=dict(t=60, r=20, l=20, b=40),
                                          title=dict(text="Matriz de confusión", x=0.02, xanchor="left",
                                                     font=dict(color="#111", size=18)))
                        st.plotly_chart(fig, use_container_width=True)

                    with right:
                        st.subheader("Métricas")
                        st.markdown(
                            f"""
                            <div style="
                                border: 1px solid rgba(148,163,184,0.35);
                                background: rgba(30,41,59,0.35);
                                border-radius: 10px;
                                padding: 14px 16px;
                                ">
                                <ul style="margin:0; padding-left:18px; color:#cbd5e1;">
                                    <li><b>Precisión (clase 0)</b>: {precision0}</li>
                                    <li><b>Precisión (clase 1)</b>: {precision1}</li>
                                    <li><b>Sensibilidad (clase 0)</b>: {sens0}</li>
                                    <li><b>Sensibilidad (clase 1)</b>: {sens1}</li>
                                    <li><b>Exactitud del modelo</b>: {accuracy}</li>
                                    <li><b>Puntaje F1 </b>: {f1macro}</li>
                                </ul>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

# ========= ESTILOS (los que ya tenías funcionando) =========
AIRBNB_RAUSCH_PALETTE = ["#FF5A5F","#FF7A7E","#E9494E","#FF9FA2","#C73C40","#FFA7AB","#B83237"]
px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = AIRBNB_RAUSCH_PALETTE

# Logo
_logo_uri = ""
lp = Path("airbnb.png")
if lp.exists():
    _logo_uri = "data:image/png;base64," + base64.b64encode(lp.read_bytes()).decode("utf-8")

st.markdown(
    f"""
    <style>
    #MainMenu {{ display: none !important; }}
    header {{ display: none !important; }}
    footer {{ display: none !important; }}
    [data-testid="stHeader"] {{ display: none !important; }}

    .stApp {{ background-color: #FFF3E7; }}

    h1, h2, h3 {{ color: #000000 !important; font-weight: 800; }}
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3 {{ color:#000000 !important; opacity:1 !important; }}
    .stMarkdown p, .stMarkdown li, .stMarkdown span, .stMarkdown div {{ color:#000000 !important; }}

    .js-plotly-plot .gtitle {{ fill: #111111 !important; opacity:1 !important; }}
    .js-plotly-plot .xtick text, .js-plotly-plot .ytick text {{ fill: #111111 !important; }}

    .stSidebar {{ background-color: #FF5A5F; }}
    .stSidebar h1, .stSidebar h2, .stSidebar h3 {{ color: #FFFFFF !important; font-weight: 800; }}

    section[data-testid="stSidebar"] div[data-baseweb="select"] > div,
    section[data-testid="stSidebar"] input,
    section[data-testid="stSidebar"] textarea {{
        background: #FFFFFF !important;
        color: #000000 !important;
        border-radius: 8px !important;
    }}
    section[data-testid="stSidebar"] div[role="combobox"] * {{ color: #000000 !important; }}
    section[data-testid="stSidebar"] div[data-baseweb="popover"],
    section[data-testid="stSidebar"] div[data-baseweb="popover"] div[role="listbox"] {{
        background: #FFFFFF !important; color: #000000 !important;
    }}
    section[data-testid="stSidebar"] div[data-baseweb="popover"] [role="option"] * {{ color: #000000 !important; }}

    .stButton > button {{
        background-color: #FF5A5F; color: #FFFFFF;
        padding: 10px 20px; border: none; border-radius: 6px; cursor: pointer;
    }}
    .stButton > button:hover {{ filter: brightness(0.92); color: #FFFFFF; }}

    .airbnb-logo {{
        position: fixed; top: 10px; right: 14px; width: 44px; height: auto;
        z-index: 9999; opacity: 0.95; pointer-events: none;
        filter: drop-shadow(0 2px 4px rgba(0,0,0,.25));
    }}
    </style>
    {("<img class='airbnb-logo' src='" + _logo_uri + "' alt='Airbnb logo' />") if _logo_uri else ""}
    """,
    unsafe_allow_html=True
)
