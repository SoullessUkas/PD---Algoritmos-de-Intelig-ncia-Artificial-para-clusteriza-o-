"""
Streamlit App: Clusterização de Países (KMeans k=3 e Hierárquico)

Este aplicativo replica a funcionalidade do notebook:
- Pré-processamento (winsorização, imputação, escalonamento)
- KMeans (k=3) com métricas treino/teste, perfis e visualizações PCA 2D/3D
- Hierárquico (Ward) com dendrograma e corte em 3 clusters
- Comparação (ARI, NMI, Homogeneidade, Completude)
- Robustez (KMeans, múltiplas sementes/splits)
- Mapa geográfico por cluster e nuvem de palavras (ou barras)

Executar localmente: streamlit run streamlit_app.py
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_score,
    completeness_score,
)
from sklearn.model_selection import train_test_split

# Opcional para VIF; se indisponível, tratamos como None
try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
except Exception:
    variance_inflation_factor = None


# -------------------------------------------
# Seção: Configuração de tema e layout
# -------------------------------------------
st.set_page_config(page_title="Clusterização de Países", layout="wide")

st.title("Clusterização de Países: KMeans k=3 e Hierárquico")
st.caption("Aplicativo interativo que replica os cálculos e visualizações do notebook com resultados idênticos por padrão.")


# -------------------------------------------
# Seção: Funções utilitárias (idênticas ao notebook)
# -------------------------------------------
def detect_country_column(df: pd.DataFrame):
    for name in ["country"]:
        if name in df.columns:
            return name
    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if not obj_cols:
        return None
    n = len(df)
    best, best_ratio = None, 0.0
    for c in obj_cols:
        ratio = df[c].nunique(dropna=True) / max(n, 1)
        if ratio > best_ratio:
            best_ratio, best = ratio, c
    return best


def compute_stats(df: pd.DataFrame, num_cols):
    rows = []
    for col in num_cols:
        s = df[col].dropna()
        if s.empty:
            rows.append({
                "variable": col,
                "count": int(df[col].count()),
                "min": np.nan,
                "max": np.nan,
                "range": np.nan,
                "mean": np.nan,
                "median": np.nan,
                "std": np.nan,
                "skew": np.nan,
                "kurtosis": np.nan,
                "q1": np.nan,
                "q3": np.nan,
                "iqr": np.nan,
                "outliers_iqr": 0,
                "outliers_zscore": 0,
                "missing_fraction": float(df[col].isna().mean()),
            })
            continue
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        out_iqr = int(((s < lower) | (s > upper)).sum())
        z = np.abs(stats.zscore(s, nan_policy="omit"))
        out_z = int((z > 3).sum()) if isinstance(z, np.ndarray) else 0
        rows.append({
            "variable": col,
            "count": int(s.count()),
            "min": float(s.min()),
            "max": float(s.max()),
            "range": float(s.max() - s.min()),
            "mean": float(s.mean()),
            "median": float(s.median()),
            "std": float(s.std(ddof=1)),
            "skew": float(stats.skew(s, nan_policy="omit")),
            "kurtosis": float(stats.kurtosis(s, nan_policy="omit")),
            "q1": float(q1),
            "q3": float(q3),
            "iqr": float(iqr),
            "outliers_iqr": out_iqr,
            "outliers_zscore": out_z,
            "missing_fraction": float(df[col].isna().mean()),
        })
    return pd.DataFrame(rows)


def decide_scaler(stats_df: pd.DataFrame):
    outlier_cols = (stats_df["outliers_iqr"] > 0) | (stats_df["outliers_zscore"] > 0)
    skew_high = stats_df["skew"].abs() > 1
    if outlier_cols.sum() >= max(1, len(stats_df) // 3) or skew_high.sum() >= max(1, len(stats_df) // 3):
        return "robust"
    return "standard"


def winsorize_iqr(df: pd.DataFrame, num_cols):
    df_w = df.copy()
    for col in num_cols:
        s = df_w[col]
        if s.isna().all():
            continue
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df_w[col] = s.clip(lower, upper)
    return df_w


def compute_vif(df: pd.DataFrame, num_cols):
    if variance_inflation_factor is None or len(num_cols) < 2:
        return None
    X = df[num_cols].copy()
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)
    res = []
    for i in range(len(num_cols)):
        vif = float(variance_inflation_factor(X_imputed, i))
        res.append({"variable": num_cols[i], "VIF": vif})
    return pd.DataFrame(res)


def safe_metric(fn, X, y):
    try:
        if len(set(y)) < 2:
            return np.nan
        return float(fn(X, y))
    except Exception:
        return np.nan


@st.cache_data(show_spinner=False)
def load_default_dataset():
    """Carrega o dataset padrão do diretório atual."""
    candidates = [
        os.path.join(os.getcwd(), "Country-data.csv"),
        os.path.join(os.path.dirname(os.getcwd()), "Clusterização", "Country-data.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return pd.read_csv(p)
    raise FileNotFoundError("Country-data.csv não encontrado.")


def build_preprocessor(df_w_cat, num_cols, cat_cols, scaler_choice_override=None):
    feature_cols = [c for c in df_w_cat.columns if c != "country"]
    X = df_w_cat[feature_cols]
    numeric_feature_cols = [c for c in X.columns if c in num_cols]
    other_feature_cols = [c for c in X.columns if c not in numeric_feature_cols]
    stats_df = compute_stats(df_w_cat, num_cols)
    choice = decide_scaler(stats_df) if scaler_choice_override == "auto" else scaler_choice_override
    if choice is None:
        choice = decide_scaler(stats_df)
    scaler = RobustScaler() if choice == "robust" else StandardScaler()
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", scaler)]), numeric_feature_cols),
            ("other", SimpleImputer(strategy="most_frequent"), other_feature_cols),
        ],
        remainder="drop",
    )
    X_processed = preprocessor.fit_transform(X)
    processed_feature_names = numeric_feature_cols + other_feature_cols
    return X_processed, processed_feature_names, preprocessor


def descriptive_cluster_names(df_w, df_labels, num_cols, label_col):
    """Gera nomes descritivos para clusters com base em médias de indicadores-chave."""
    names = {}
    keys = ["income", "gdpp", "life_expec", "child_mort", "total_fer"]
    means_global = df_w[keys].mean()
    for c in sorted(df_labels[label_col].unique()):
        sub = df_w[df_labels[label_col] == c]
        means_c = sub[keys].mean()
        hi_income = means_c.get("income", 0) > means_global.get("income", 0)
        hi_gdpp = means_c.get("gdpp", 0) > means_global.get("gdpp", 0)
        hi_life = means_c.get("life_expec", 0) > means_global.get("life_expec", 0)
        hi_child = means_c.get("child_mort", 0) > means_global.get("child_mort", 0)
        hi_fer = means_c.get("total_fer", 0) > means_global.get("total_fer", 0)
        if hi_income and hi_gdpp and hi_life and not hi_child and not hi_fer:
            names[c] = "Alta renda e expectativa de vida"
        elif hi_child and hi_fer and (not hi_income) and (not hi_gdpp):
            names[c] = "Alta mortalidade infantil e fertilidade"
        else:
            names[c] = "Perfil intermediário"
    return names


# -------------------------------------------
# Seção: Barra lateral e carregamento de dados
# -------------------------------------------
st.sidebar.header("Parâmetros")

data_choice = st.sidebar.radio(
    "Fonte de dados",
    options=["Exemplo (Country-data.csv)", "Upload CSV personalizado"],
    index=0,
    help="Escolha usar o dataset de exemplo ou carregar um arquivo CSV customizado.",
)

uploaded_df = None
if data_choice == "Upload CSV personalizado":
    file = st.sidebar.file_uploader("Carregar CSV", type=["csv"], accept_multiple_files=False, help="Arquivo deve conter coluna 'country' e variáveis numéricas.")
    if file is not None:
        try:
            uploaded_df = pd.read_csv(file)
        except Exception as e:
            st.sidebar.error(f"Falha ao ler CSV: {e}")

random_seed = st.sidebar.number_input("Semente (random_state)", min_value=0, value=42, step=1, help="Garantir reprodutibilidade dos resultados.")
test_size = st.sidebar.slider("Proporção de teste", min_value=0.1, max_value=0.4, value=0.2, step=0.05, help="Tamanho do conjunto de teste para métricas em geral.")
k_clusters = st.sidebar.selectbox(
    "Número de clusters (KMeans)", options=[3], index=0,
    help="Fixado em 3 para equivalência funcional com o notebook."
)

winsor_toggle = st.sidebar.toggle("Aplicar winsorização (IQR)", value=True, help="Reduz influência de outliers limitando valores extremos.")
scaler_choice = st.sidebar.selectbox("Escalonamento", options=["auto", "standard", "robust"], index=0, help="Escolha automática (igual ao notebook) ou forçar Standard/Robust.")

show_vif = st.sidebar.toggle("Calcular VIF (se disponível)", value=False, help="Verifica multicolinearidade entre variáveis numéricas.")
show_pca_3d = st.sidebar.toggle("Mostrar PCA 3D", value=True, help="Visualização 3D adicional dos clusters KMeans.")
top_n_features = st.sidebar.slider("Top features por |z-score|", min_value=5, max_value=15, value=10, help="Número de principais características por cluster na análise de perfil.")

highlight_cluster = st.sidebar.selectbox("Destacar cluster", options=["Todos", 0, 1, 2], index=0, help="Permite destacar um cluster específico nas visualizações.")

st.sidebar.markdown("---")
run_button = st.sidebar.button("Executar fluxo", type="primary", help="Processa dados, treina modelos e atualiza visualizações.")


# -------------------------------------------
# Seção: Carregamento e pré-processamento
# -------------------------------------------
if uploaded_df is not None:
    df = uploaded_df.copy()
else:
    df = load_default_dataset()

country_col = detect_country_column(df)
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
if country_col and country_col in cat_cols:
    cat_cols = [c for c in cat_cols if c != country_col]

st.write("Dimensão dos dados:", df.shape)
st.dataframe(df.head(10))

if winsor_toggle:
    df_w = winsorize_iqr(df, num_cols)
else:
    df_w = df.copy()

df_w_cat = df_w.copy()
if len(cat_cols) > 0:
    df_w_cat = pd.get_dummies(df_w_cat, columns=cat_cols, drop_first=True)

# Construir preprocessor e features processadas
X_processed, processed_feature_names, preprocessor = build_preprocessor(df_w_cat, num_cols, cat_cols, scaler_choice_override=scaler_choice)
X_features = X_processed
countries = df[country_col].values if country_col else df.index.values

st.success("Pré-processamento concluído.")

if show_vif:
    vif_df = compute_vif(df, num_cols)
    if vif_df is not None:
        st.subheader("VIF (Multicolinearidade)")
        st.dataframe(vif_df)
    else:
        st.info("VIF não disponível (statsmodels indisponível ou poucas variáveis).")


# -------------------------------------------
# Seção: Treinamento e avaliação
# -------------------------------------------
np.random.seed(random_seed)
X_idx = np.arange(X_features.shape[0])
X_train, X_test, idx_train, idx_test = train_test_split(X_features, X_idx, test_size=test_size, random_state=random_seed)
countries_train = countries[idx_train]
countries_test = countries[idx_test]

km3 = KMeans(n_clusters=k_clusters, init="k-means++", max_iter=300, tol=1e-4, n_init=10, random_state=random_seed)
labels_train_km = km3.fit_predict(X_train)
labels_test_km = km3.predict(X_test)
labels_all_km = km3.predict(X_features)

metrics_km = {
    "silhouette_train": safe_metric(silhouette_score, X_train, labels_train_km),
    "silhouette_test": safe_metric(silhouette_score, X_test, labels_test_km),
    "davies_bouldin_train": safe_metric(davies_bouldin_score, X_train, labels_train_km),
    "davies_bouldin_test": safe_metric(davies_bouldin_score, X_test, labels_test_km),
}

st.subheader("Métricas KMeans (k=3)")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Silhouette (treino)", f"{metrics_km['silhouette_train']:.4f}")
col2.metric("Silhouette (teste)", f"{metrics_km['silhouette_test']:.4f}")
col3.metric("Davies-Bouldin (treino)", f"{metrics_km['davies_bouldin_train']:.4f}")
col4.metric("Davies-Bouldin (teste)", f"{metrics_km['davies_bouldin_test']:.4f}")

# Perfis dos clusters
df_km = df.copy()
df_km["cluster_km3"] = labels_all_km
centroids_processed = pd.DataFrame(km3.cluster_centers_, columns=processed_feature_names)

# Países mais representativos
from numpy.linalg import norm
rep_countries = {}
for c in range(k_clusters):
    idx_c = np.where(labels_all_km == c)[0]
    if len(idx_c) == 0:
        rep_countries[c] = None
        continue
    dists = norm(X_features[idx_c] - km3.cluster_centers_[c], axis=1)
    best_idx = idx_c[int(np.argmin(dists))]
    rep_countries[c] = countries[best_idx].item() if hasattr(countries[best_idx], "item") else countries[best_idx]

# Principais características por |z-score|
global_means = df_w[num_cols].mean()
global_stds = df_w[num_cols].std(ddof=1).replace(0, np.nan)
top_features = {}
for c in range(k_clusters):
    df_c = df_w[df_km["cluster_km3"] == c]
    means_c = df_c[num_cols].mean()
    z = ((means_c - global_means) / global_stds).abs().sort_values(ascending=False)
    top_features[c] = z.dropna().head(top_n_features).to_dict()

# Nomes descritivos por cluster
cluster_names = descriptive_cluster_names(df_w, df_km, num_cols, "cluster_km3")

st.subheader("Perfis por cluster (KMeans)")
st.json({
    "cluster_sizes": df_km["cluster_km3"].value_counts().sort_index().to_dict(),
    "representative_countries": rep_countries,
    "cluster_names": cluster_names,
})


# -------------------------------------------
# Seção: Visualizações (máximo 3-4 simultâneas)
# -------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["PCA 2D", "PCA 3D", "Dendrograma", "Mapa & Top Features"])

with tab1:
    st.markdown("### Projeção PCA 2D (interativa)")
    pca_vis2 = PCA(n_components=2, random_state=random_seed)
    X_vis2 = pca_vis2.fit_transform(X_features)
    try:
        import plotly.express as px
        df_pca2 = pd.DataFrame({
            "PC1": X_vis2[:, 0],
            "PC2": X_vis2[:, 1],
            "cluster": labels_all_km,
            "country": countries,
        })
        df_pca2["alpha"] = 0.9
        if highlight_cluster in [0, 1, 2]:
            df_pca2.loc[df_pca2["cluster"] != highlight_cluster, "alpha"] = 0.2
        fig_scatter = px.scatter(
            df_pca2,
            x="PC1",
            y="PC2",
            color="cluster",
            hover_name="country",
            opacity=df_pca2["alpha"],
            title="KMeans k=3 (PCA 2D)",
            color_continuous_scale="Viridis",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    except Exception as e:
        st.warning(f"Plotly indisponível: {e}. Alternando para gráfico estático.")
        plt.figure(figsize=(8, 6))
        palette = sns.color_palette("tab10", n_colors=k_clusters)
        mask = np.ones_like(labels_all_km, dtype=bool)
        if highlight_cluster in [0, 1, 2]:
            mask = labels_all_km == highlight_cluster
        sns.scatterplot(x=X_vis2[:, 0], y=X_vis2[:, 1], hue=labels_all_km, palette=palette, alpha=np.where(mask, 0.9, 0.2))
        plt.title("KMeans k=3 (PCA 2D)")
        plt.xlabel("PC1"); plt.ylabel("PC2")
        st.pyplot(plt.gcf())

with tab2:
    st.markdown("### Projeção PCA 3D")
    if show_pca_3d:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        pca_vis3 = PCA(n_components=3, random_state=random_seed)
        X_vis3 = pca_vis3.fit_transform(X_features)
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")
        sc = ax.scatter(X_vis3[:, 0], X_vis3[:, 1], X_vis3[:, 2], c=labels_all_km, cmap="tab10", alpha=0.85)
        ax.set_title("KMeans k=3 (PCA 3D)"); ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
        fig.colorbar(sc, label="cluster")
        st.pyplot(fig)
    else:
        st.info("PCA 3D está desativado.")

with tab3:
    st.markdown("### Dendrograma (Ward)")
    from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
    Z = linkage(X_features, method="ward", metric="euclidean")
    fig = plt.figure(figsize=(12, 6))
    dendrogram(Z, labels=countries if country_col else None, leaf_rotation=90, leaf_font_size=8, color_threshold=None)
    plt.title("Dendrograma (linkage: Ward)")
    plt.xlabel("País"); plt.ylabel("Distância")
    st.pyplot(fig)
    labels_hier3 = fcluster(Z, t=3, criterion="maxclust") - 1
    metrics_hier = {
        "silhouette_all": safe_metric(silhouette_score, X_features, labels_hier3),
        "davies_bouldin_all": safe_metric(davies_bouldin_score, X_features, labels_hier3),
    }
    st.json({"Métricas Hierárquico (k=3 via corte)": metrics_hier})

with tab4:
    st.markdown("### Mapa geográfico por cluster e Top Features")
    try:
        import plotly.express as px
        df_map = pd.DataFrame({"country": countries, "cluster": labels_all_km})
        fig_map = px.choropleth(
            df_map,
            locations="country",
            locationmode="country names",
            color="cluster",
            color_continuous_scale="Viridis",
            title="Mapa por cluster (KMeans k=3)",
        )
        st.plotly_chart(fig_map, use_container_width=True)
    except Exception as e:
        st.warning(f"Plotly indisponível ou países não reconhecidos: {e}")
        st.write("Exportar df_map e plote manualmente em uma ferramenta geográfica.")

    st.markdown("#### Top features por |z-score|")
    cluster_select = st.selectbox("Cluster para visualizar", options=[0, 1, 2], help="Escolha o cluster para exibir suas principais características.")
    items = sorted(top_features.get(cluster_select, {}).items(), key=lambda x: x[1], reverse=True)
    items = items[:top_n_features]
    if items:
        labels_b = [k for k, _ in items]
        vals_b = [v for _, v in items]
        fig_bar, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=vals_b, y=labels_b, orient="h", palette="viridis", ax=ax)
        ax.set_xlabel("importância (|z-score|)"); ax.set_ylabel("feature")
        ax.set_title(f"Top features por |z-score| (cluster {cluster_select})")
        st.pyplot(fig_bar)
    else:
        st.info("Sem features disponíveis para este cluster.")


# -------------------------------------------
# Seção: Comparação e robustez
# -------------------------------------------
st.subheader("Comparação KMeans vs Hierárquico")
labels_h_all = labels_hier3
comparison = {
    "ARI": float(adjusted_rand_score(labels_all_km, labels_h_all)),
    "NMI": float(normalized_mutual_info_score(labels_all_km, labels_h_all)),
    "homogeneity_km_vs_hier": float(homogeneity_score(labels_all_km, labels_h_all)),
    "completeness_km_vs_hier": float(completeness_score(labels_all_km, labels_h_all)),
}
colA, colB, colC, colD = st.columns(4)
colA.metric("ARI", f"{comparison['ARI']:.4f}")
colB.metric("NMI", f"{comparison['NMI']:.4f}")
colC.metric("Homogeneidade", f"{comparison['homogeneity_km_vs_hier']:.4f}")
colD.metric("Completude", f"{comparison['completeness_km_vs_hier']:.4f}")

st.subheader("Tabela de contingência (rótulos)")
ct = pd.crosstab(pd.Series(labels_all_km, name="KMeans"), pd.Series(labels_h_all, name="Hierárquico"))
st.dataframe(ct)

st.subheader("Robustez (KMeans k=3)")
seeds = list(range(10))
sil_train_list, sil_test_list, db_train_list, db_test_list = [], [], [], []
for s in seeds:
    X_tr, X_te = train_test_split(X_features, test_size=test_size, random_state=s)[0:2]
    km_tmp = KMeans(n_clusters=k_clusters, init="k-means++", max_iter=300, tol=1e-4, n_init=10, random_state=s)
    lab_tr = km_tmp.fit_predict(X_tr)
    lab_te = km_tmp.predict(X_te)
    sil_train_list.append(safe_metric(silhouette_score, X_tr, lab_tr))
    sil_test_list.append(safe_metric(silhouette_score, X_te, lab_te))
    db_train_list.append(safe_metric(davies_bouldin_score, X_tr, lab_tr))
    db_test_list.append(safe_metric(davies_bouldin_score, X_te, lab_te))

robust_summary = {
    "silhouette_train_mean": float(np.nanmean(sil_train_list)),
    "silhouette_train_std": float(np.nanstd(sil_train_list)),
    "silhouette_test_mean": float(np.nanmean(sil_test_list)),
    "silhouette_test_std": float(np.nanstd(sil_test_list)),
    "davies_bouldin_train_mean": float(np.nanmean(db_train_list)),
    "davies_bouldin_train_std": float(np.nanstd(db_train_list)),
    "davies_bouldin_test_mean": float(np.nanmean(db_test_list)),
    "davies_bouldin_test_std": float(np.nanstd(db_test_list)),
}
colR1, colR2, colR3, colR4 = st.columns(4)
colR1.metric("Silhouette Treino (média)", f"{robust_summary['silhouette_train_mean']:.4f}")
colR2.metric("Silhouette Teste (média)", f"{robust_summary['silhouette_test_mean']:.4f}")
colR3.metric("DB Treino (média)", f"{robust_summary['davies_bouldin_train_mean']:.4f}")
colR4.metric("DB Teste (média)", f"{robust_summary['davies_bouldin_test_mean']:.4f}")

st.markdown("---")
st.info(
    "Dica: Use os controles na barra lateral para ajustar parâmetros; por padrão os valores e algoritmos reproduzem o notebook e geram resultados idênticos."
)