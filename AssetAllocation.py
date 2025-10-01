import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json
import streamlit.components.v1 as components
from datetime import datetime

st.set_page_config(page_title="Backtest Multi-Portefeuilles", layout="wide")

st.markdown(
    """
    <a href="https://www.zonebourse.com/wbfl/livre" target="_blank">
        <img src="https://raw.githubusercontent.com/EtienneNeptune/AssetAllocation/master/Pubpub.png
" width="1500">
    </a>
    """,
    unsafe_allow_html=True
)

# ===============================
# Définition actifs
# ===============================
assets = [
    "Actions mondiales (VT)",
    "Private Equity Global (PSP)",
    "Bitcoin (BTC-USD)",
    "Actions américaines (SPY)",
    "Immobilier international ex US (IFGL)",
    "Immobilier US (IYR)",
    "Obligations d'entreprises américaines (LQD)",
    "Obligations gouvernementales Global (IGLO.L)",
    "Or (GLD)",
    "Métaux industriels (DBB)",
    "Matières agricoles (DBA)",
    "Monétaire Global (IGOV)"
]

tickers = {
    "Actions mondiales (VT)": "VT",        # 2008
    "Private Equity Global (PSP)": "PSP",  # 2008
    "Bitcoin (Attention, ce n'est pas un ETF)": "BTC-USD",  # 2008
    "Actions américaines (SPY)": "SPY",    # 1993
    "Immobilier international ex US (IFGL)": "IFGL",  # 2008
    "Immobilier US (IYR)": "IYR",          # 2000
    "Obligations d'entreprises américaines (LQD)": "LQD",  # 2002
    "Obligations gouvernementales Global (IGLO.L)": "IGLO.L",  # 2009
    "Or (GLD)": "GLD",                     # 2004
    "Métaux industriels (DBB)": "DBB",     # 2007
    "Matières agricoles (DBA)": "DBA",     # 2007
    "Monétaire Global (IGOV)": "IGOV"      # 2009
}

# Pour l’affichage initial du tableau de sélection
inception_years_hint = {
    "VT": 2008, "PSP": 2008, "BTC-USD": 2014, "SPY": 1993,
    "IFGL": 2008, "IYR": 2000, "LQD": 2002, "IGLO.L": 2009,
    "GLD": 2004, "DBB": 2007, "DBA": 2007, "IGOV": 2009
}

# ===============================
# Sélection des actifs (tableau avec cases à cocher)
# ===============================
st.title("Je sélectionne les ETFs que je souhaite analyser / backtester")

default_checked = {"VT", "PSP", "IYR", "LQD", "GLD"}


hdr = st.columns([3, 1, 1])
hdr[0].markdown("**Les différentes classes d'actifs (un ETF/ETP pour chaque classe)**")
hdr[1].markdown("**Année de lancement de l'ETF/ETP**")
hdr[2].markdown("**Télécharger ?**")


selected_assets = []
for asset in assets:
    tkr = tickers[asset]
    c1, c2, c3 = st.columns([3, 1, 1])
    c1.write(asset)
    c2.write(inception_years_hint.get(tkr, "n/a"))
    checked = c3.checkbox(
        f"Télécharger {asset}",
        value=(tkr in default_checked),
        key=f"chk_{tkr}",
        label_visibility="collapsed"
    )
    if checked:
        selected_assets.append(asset)

if not selected_assets:
    st.warning("Veuillez sélectionner au moins un actif pour continuer.")
    st.stop()

sel_tickers = [tickers[a] for a in selected_assets]

# ===============================
# Téléchargement des données (cache par univers sélectionné)
# ===============================
if "data_prices" not in st.session_state or st.session_state.get("cached_sel") != tuple(sel_tickers):
    data_raw = yf.download(sel_tickers, start="1990-01-01", auto_adjust=False, progress=False).dropna()
    if isinstance(data_raw.columns, pd.MultiIndex):
        tops = {lvl[0] for lvl in data_raw.columns}
        if "Adj Close" in tops:
            data_prices = data_raw["Adj Close"].copy()
        elif "Close" in tops:
            data_prices = data_raw["Close"].copy()
        else:
            data_prices = data_raw.copy()
    else:
        data_prices = data_raw.copy()

    data_prices = data_prices.sort_index()
    st.session_state["data_prices"] = data_prices
    st.session_state["cached_sel"] = tuple(sel_tickers)

data_prices = st.session_state["data_prices"]

# ===============================
# Dates de première cotation effectives + début commun possible
# ===============================
first_dates = {}
for c in data_prices.columns:
    first_idx = data_prices[c].first_valid_index()
    if first_idx is not None:
        first_dates[c] = first_idx.date()

if not first_dates:
    st.error("Pas de données de prix valides sur la période demandée.")
    st.stop()

common_start = max(first_dates.values())
max_date = data_prices.index.max().date()

# ===============================
# Sélection de période utilisateur
# ===============================
date_range = st.date_input(
    "**Je selectionne ma période d'analyse (par défaut, la date de début = premier jour de cotation en commun dans l'ensemble des ETFs séléctionnés)**",
    value=[common_start, max_date],
    min_value=common_start,
    max_value=max_date
)
start_date, end_date = date_range
data_filtered = data_prices.loc[str(start_date):str(end_date)].copy()

# ===============================
# Graphique global (rebasing)
# ===============================
st.subheader("Évolution des performances des ETFs / Classes d'actifs")

def rebase_series_to_first_valid(s: pd.Series) -> pd.Series:
    s = s.dropna()
    if s.empty:
        return s
    return s / s.iloc[0]

data_norm = data_filtered.apply(rebase_series_to_first_valid, axis=0)

fig_perf = go.Figure()
for asset in selected_assets:
    tkr = tickers[asset]
    if tkr in data_norm.columns:
        fig_perf.add_trace(go.Scatter(x=data_norm.index, y=data_norm[tkr], mode='lines', name=asset))
fig_perf.update_layout(
    title="Performances en base 1. Exemple : 2 = la NAV à été multipliée par 2 (équivaut à une évolution de +100%)",
    xaxis_title="Date",
    yaxis_title="Performance (base 1)",
    template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
)
st.plotly_chart(fig_perf, use_container_width=True)

# ===============================
# Statistiques par classe d'actifs
# ===============================
st.subheader("Statistiques par classe d'actifs")

asset_stats_rows = []
asset_index = []
# rendements log pour stats
asset_logret = np.log(data_filtered).diff().dropna()

for asset in selected_assets:
    tkr = tickers[asset]
    s = data_filtered[tkr].dropna()
    if len(s) < 2:
        continue

    # CAGR sur la période visible
    years = (s.index[-1] - s.index[0]).days / 365.2425
    cagr = (s.iloc[-1] / s.iloc[0]) ** (1 / years) - 1 if years > 0 else np.nan

    # Annualized avg return (depuis log-returns → retour simple annualisé)
    r = asset_logret[tkr].dropna() if tkr in asset_logret else pd.Series(dtype=float)
    ann_avg = np.expm1(r.mean() * 252) if not r.empty else np.nan  # = e^{mu*252}-1

    # Vol annualisée
    vol_annual = r.std() * np.sqrt(252) if not r.empty else np.nan

    # Best/Worst year (rendements simples par année)
    yearly = s.resample('YE').last().pct_change().dropna()
    best_year = yearly.max() if not yearly.empty else np.nan
    worst_year = yearly.min() if not yearly.empty else np.nan

    # Max Drawdown sur le prix
    hwm = s.cummax()
    dd = s / hwm - 1.0
    max_dd = dd.min()

    # Sharpe (rf=0)
    sharpe = (r.mean() / r.std()) * np.sqrt(252) if (not r.empty and r.std() > 0) else np.nan

    asset_index.append(asset)
    asset_stats_rows.append([cagr, ann_avg, vol_annual, best_year, worst_year, max_dd, sharpe])

if asset_stats_rows:
    asset_stats_df = pd.DataFrame(
        asset_stats_rows,
        index=asset_index,
        columns=[
            "CAGR (Rendement moyen annuel composé)", "Rendement moyen annualisé", "Volatilité",
            "Meilleure année", "Pire année", "Chute la plus importante", "Ratio de Sharpe"
        ]
    )
    st.dataframe(
        asset_stats_df.style.format({
            "CAGR (Rendement moyen annuel composé)": "{:.2%}",
            "Rendement moyen annualisé": "{:.2%}",
            "Volatilité": "{:.2%}",
            "Meilleure année": "{:.2%}",
            "Pire année": "{:.2%}",
            "Chute la plus importante": "{:.2%}",
            "Ratio de Sharpe": "{:.2f}",
        }),
        use_container_width=True
    )
else:
    st.info("Pas assez de données pour les statistiques par classe d'actifs.")

# ===============================
# Corrélations (matrice interactive + graph D3 sans perf)
# ===============================
st.subheader("Corrélations - Cherchez les classes d'actifs suceptibles de baisser la volatilité et augmenter votre CAGR !")

# Matrice de corrélation interactive (compacte)
returns = data_filtered.pct_change().dropna()
if returns.empty or returns.shape[1] < 2:
    st.warning("Pas assez de données pour calculer des corrélations.")
    corr_mat = pd.DataFrame()
else:
    corr_mat = returns.corr()

if not corr_mat.empty:
    fig_corr = px.imshow(
        corr_mat,
        zmin=-1, zmax=1,
        color_continuous_scale="RdBu_r",
        aspect="auto"
    )
    fig_corr.update_traces(
        hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>ρ = %{z:.2f}<extra></extra>"
    )
    fig_corr.update_layout(
        title="Matrice de corrélation (-1 à 1)",
        height=320,
        width=400,
        margin=dict(l=30, r=30, t=40, b=30),
        coloraxis_colorbar=dict(title="ρ"),
        font=dict(size=11)
    )
    fig_corr.update_xaxes(tickangle=45)
    st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("**Réseau de corrélation (une invention Zonebourse). Jouez avec le seuil pour faire apparaître / disparaître des liens !**")
    corr_threshold = st.slider("Seuil de corrélation (|ρ| ≥ seuil)", 0.0, 1.0, 0.35, 0.05)

    # Préparer nodes / links (sans performance)
    nodes, links = [], []
    ticker_to_asset = {v: k for k, v in tickers.items() if k in assets}

    present_tickers = [c for c in corr_mat.columns]  # ceux disponibles dans la corr
    # Nodes à rayon constant, labels = nom actif uniquement
    for tkr in present_tickers:
        if tkr not in ticker_to_asset:
            continue
        nodes.append({
            "id": tkr,
            "name": ticker_to_asset[tkr],
            "radius": 9.0
        })

    for i in range(len(present_tickers)):
        for j in range(i + 1, len(present_tickers)):
            a, b = present_tickers[i], present_tickers[j]
            c = corr_mat.loc[a, b]
            if pd.isna(c):
                continue
            if abs(c) >= corr_threshold:
                # value plus grand si corr plus faible => lien plus long
                value = float(5 * (1 - (c + 1) / 2))
                value = max(0.0, min(5.0, value))
                links.append({"source": a, "target": b, "value": value})

    graph_payload = {"nodes": nodes, "links": links}

    html = f"""
    <div id="corr-stage" style="width:100%; height:600px;"></div>
    <style>
      .node-label {{
        font: 11px ui-sans-serif, system-ui;
        fill: #333;
        pointer-events: none;
      }}
    </style>
    <script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
    <script>
      const DATA = {json.dumps(graph_payload).replace("</", "<\\/")};
      const container = document.getElementById('corr-stage');
      const WIDTH = container.clientWidth || 1000;
      const HEIGHT = container.clientHeight || 600;

      const svg = d3.select(container).append("svg")
        .attr("width", WIDTH)
        .attr("height", HEIGHT)
        .attr("viewBox", [0,0,WIDTH,HEIGHT]);

      const color = d3.scaleOrdinal(d3.schemeCategory10);

      const link = svg.append("g")
        .attr("stroke", "#999")
        .attr("stroke-opacity", 0.6)
        .selectAll("line")
        .data(DATA.links)
        .join("line")
        .attr("stroke-width", d => Math.sqrt(1 + d.value));

      const node = svg.append("g")
        .attr("stroke", "#fff")
        .attr("stroke-width", 1.2)
        .selectAll("circle")
        .data(DATA.nodes)
        .join("circle")
        .attr("r", d => d.radius || 8)
        .attr("fill", d => color(d.id))
        .call(d3.drag()
          .on("start", dragstarted)
          .on("drag", dragged)
          .on("end", dragended));

      const labels = svg.append("g")
        .selectAll("text")
        .data(DATA.nodes)
        .join("text")
        .attr("class", "node-label")
        .text(d => d.name)
        .attr("text-anchor", "middle")
        .attr("dy", -10);

      const simulation = d3.forceSimulation(DATA.nodes)
        .force("link", d3.forceLink(DATA.links).id(d => d.id).distance(d => 100 + 50*d.value).strength(0.6))
        .force("charge", d3.forceManyBody().strength(-220).distanceMax(300))
        .force("center", d3.forceCenter(WIDTH/2, HEIGHT/2))
        .force("collision", d3.forceCollide().radius(d => (d.radius || 8) + 6))
        .on("tick", ticked);

      function ticked() {{
        link
          .attr("x1", d => d.source.x)
          .attr("y1", d => d.source.y)
          .attr("x2", d => d.target.x)
          .attr("y2", d => d.target.y);

        node
          .attr("cx", d => d.x)
          .attr("cy", d => d.y);

        labels
          .attr("x", d => d.x)
          .attr("y", d => d.y - 10);
      }}

      function dragstarted(event) {{
        if (!event.active) simulation.alphaTarget(0.3).restart();
        event.subject.fx = event.subject.x;
        event.subject.fy = event.subject.y;
      }}
      function dragged(event) {{
        event.subject.fx = event.x;
        event.subject.fy = event.y;
      }}
      function dragended(event) {{
        if (!event.active) simulation.alphaTarget(0);
        event.subject.fx = null;
        event.subject.fy = null;
      }}
    </script>
    """
    components.html(html, height=620, scrolling=True)

# ===============================
# Backtest (Buy & Hold ou Rebalancing N jours)
# ===============================
st.title("Configuration des portefeuilles")

# Mémoire de session
if 'table_rows' not in st.session_state:
    st.session_state.table_rows = 4
if 'portfolios' not in st.session_state:
    st.session_state.portfolios = ["Portefeuille 1"]

# Gestion lignes/colonnes
c1, c2 = st.columns(2)
if c1.button("➕ Ajouter un actif (ligne)"):
    st.session_state.table_rows += 1
if c2.button("➖ Supprimer un actif (ligne)") and st.session_state.table_rows > 1:
    st.session_state.table_rows -= 1

c3, c4 = st.columns(2)
if c3.button("➕ Ajouter un portefeuille (colonne)"):
    st.session_state.portfolios.append(f"Portefeuille {len(st.session_state.portfolios)+1}")
if c4.button("➖ Supprimer un portefeuille (colonne)") and len(st.session_state.portfolios) > 1:
    st.session_state.portfolios.pop()

# Noms et capital
st.subheader("Nom et capital initial")
cap_cols = st.columns(len(st.session_state.portfolios))
start_capitals = []
for i in range(len(st.session_state.portfolios)):
    st.session_state.portfolios[i] = cap_cols[i].text_input("Nom", value=st.session_state.portfolios[i], key=f"name_{i}")
    start_capitals.append(cap_cols[i].number_input("Capital initial (€)", value=10000, step=1000, key=f"capital_{i}"))

# Tableau d'allocations
st.subheader("Tableau d'allocations")
tab_cols = st.columns(len(st.session_state.portfolios) + 1)
tab_cols[0].markdown("**Classes d'actifs**")
for j, pname in enumerate(st.session_state.portfolios):
    tab_cols[j + 1].markdown(f"**{pname} (%)**")

alloc_table = []
used_assets = set()

for i in range(st.session_state.table_rows):
    available_assets = [a for a in selected_assets if a not in used_assets]
    # On ajoute une "sentinelle" si plus d’option disponible
    if not available_assets:
        available_assets = ["No Option to select"]

    asset_select = tab_cols[0].selectbox(
        f"Actif {i}",
        options=available_assets,
        index=0,
        key=f"asset_{i}"
    )

    if asset_select != "No Option to select":
        used_assets.add(asset_select)

    row = [asset_select]
    for j in range(len(st.session_state.portfolios)):
        w = tab_cols[j + 1].number_input(
            f"{st.session_state.portfolios[j]} alloc {i}",
            min_value=0, max_value=100, value=0,
            step=5,
            key=f"alloc_{i}_{j}",
            disabled=(asset_select == "No Option to select")  # 🔒 poids bloqués
        )
        row.append(w)
    alloc_table.append(row)

# ===============================
# Paramètres par portefeuille
# ===============================
st.subheader("Stratégie d'investissement des portefeuilles")

# helper fréquence → jours
def freq_to_days(label: str) -> int | None:
    if label == "Mensuel (30 j)":
        return 30
    if label == "Hebdomadaire (7 j)":
        return 7
    if label == "Trimestriel (90 j)":
        return 90
    return None

# Pour chaque portefeuille, on choisit la stratégie et ses paramètres
port_configs = []
for j, pname in enumerate(st.session_state.portfolios):
    with st.expander(f"⚙️ Paramètres – {pname}", expanded=(len(st.session_state.portfolios) == 1)):
        cc1, cc2, cc3 = st.columns(3)

        strategy_j = cc1.selectbox(
            "Stratégie",
            ["Buy & Hold", "Rebalancing (N jours)", "DCA (apport périodique)"],
            index=0,
            key=f"strategy_{j}"
        )

        # Paramètres par stratégie
        reb_days_j = None
        dca_amount_j = 0.0
        dca_days_j = None

        if strategy_j == "Rebalancing (N jours)":
            reb_days_j = int(cc2.number_input("Intervalle N (jours)", value=60, min_value=2, max_value=365, step=1, key=f"reb_{j}"))
        elif strategy_j == "DCA (apport périodique)":
            dca_amount_j = float(cc2.number_input("Montant DCA (€)", value=500.0, step=100.0, min_value=0.0, key=f"dca_amt_{j}"))
            freq_label = cc3.selectbox("Fréquence", ["Mensuel (30 j)", "Hebdomadaire (7 j)", "Trimestriel (90 j)", "Chaque N jours"], index=0, key=f"dca_freq_{j}")
            dca_days_j = freq_to_days(freq_label)
            if dca_days_j is None:  # "Chaque N jours"
                dca_days_j = int(st.number_input("N jours", value=30, min_value=2, max_value=365, step=1, key=f"dca_n_{j}"))

        # Frais par portefeuille (si tu veux les garder globaux, mets-le ailleurs)
        costs_bps_j = float(st.number_input("Frais de courtage (bps)", value=0, min_value=0, max_value=200, step=1, key=f"costs_{j}"))

        port_configs.append({
            "strategy": strategy_j,
            "reb_days": reb_days_j,
            "dca_amount": dca_amount_j,
            "dca_days": dca_days_j,
            "costs_bps": costs_bps_j,
        })

# Fonction de backtest. Attention ici, pour faciliter les calculs, on va faire comme si on pouvait acheter des fractions d'actions... Grosse simplification.
# On fait ça pour pas que la perf soit dépendante du capital initial dans le portefeuille.
def backtest_with_rebalance_or_dca(
    prices: pd.DataFrame,
    weights_pct: dict,
    start_capital: float,
    reb_days: int | None = None,
    costs_bps: float = 0.0,
    dca_amount: float = 0.0,
    dca_days: int | None = None
) -> tuple[pd.Series, pd.Series]:
    cols = list(weights_pct.keys())
    prices = prices[cols].dropna().copy()
    w = (pd.Series(weights_pct, dtype=float) / 100.0).reindex(cols).fillna(0.0)

    dates = prices.index
    holdings = (start_capital * w) / prices.iloc[0]
    port_val = pd.Series(index=dates, dtype=float)
    flows = pd.Series(0.0, index=dates)

    port_val.iloc[0] = float((holdings * prices.iloc[0]).sum())

    # Choix de la cadence d’événement
    event_days = None
    if dca_amount > 0 and dca_days is not None:
        event_days = dca_days
    elif reb_days is not None:
        event_days = reb_days

    last_event = dates[0]

    for t in range(1, len(dates)):
        pv = float((holdings * prices.iloc[t]).sum())
        event_due = (event_days is not None) and ((dates[t] - last_event).days >= event_days)

        if event_due:
            # DCA éventuel : on ajoute le cash à la valeur
            if dca_amount > 0:
                pv += dca_amount
                flows.iloc[t] += dca_amount  # flux externe

            # Rebalance systématique à l’événement (même en DCA)
            target_val = pv * w
            current_val = holdings * prices.iloc[t]
            trade_val = target_val - current_val

            if costs_bps > 0:
                cost = trade_val.abs().sum() * (costs_bps / 10_000.0)
                pv_after_cost = pv - cost
                target_val = pv_after_cost * w

            holdings = target_val / prices.iloc[t]
            last_event = dates[t]
            pv = float((holdings * prices.iloc[t]).sum())

        port_val.iloc[t] = pv

    return port_val, flows

# ===============================
# Lancer backtest
# ===============================
if st.button("🚀 Lancer le backtest"):
    # Vérif totaux = 100% par portefeuille
    valid = True
    for j in range(len(st.session_state.portfolios)):
        total = sum([row[j + 1] for row in alloc_table])
        if total != 100:
            st.error(f"Le portefeuille **{st.session_state.portfolios[j]}** fait {total}%, il doit faire 100%.")
            valid = False

    if valid:
        use_cols = [tickers[a] for a in selected_assets]
        data_subset = data_filtered[use_cols].dropna()
        if data_subset.empty:
            st.warning("Pas assez de données sur la période sélectionnée pour exécuter le backtest.")
            st.stop()

        capital_data = pd.DataFrame(index=data_subset.index)
        flow_data = pd.DataFrame(0.0, index=data_subset.index, columns=st.session_state.portfolios)

        for j, pname in enumerate(st.session_state.portfolios):
            weights_pct = {tickers[row[0]]: row[j + 1] for row in alloc_table if (row[0] != "No Option to select" and row[j + 1] > 0)}
            if not weights_pct:
                st.warning(f"Aucun poids non nul pour {pname}.")
                continue

            cfg = port_configs[j]
            series_val, series_flow = backtest_with_rebalance_or_dca(
                prices=data_subset,
                weights_pct=weights_pct,
                start_capital=start_capitals[j],
                reb_days=cfg["reb_days"],
                costs_bps=cfg["costs_bps"],
                dca_amount=cfg["dca_amount"],
                dca_days=cfg["dca_days"]
            )
            capital_data[pname] = series_val
            flow_data[pname] = series_flow

        # Courbe de capital
        st.subheader("Évolution du capital")
        fig_bt = go.Figure()
        for col in capital_data.columns:
            fig_bt.add_trace(go.Scatter(x=capital_data.index, y=capital_data[col], mode='lines', name=col))
        fig_bt.update_layout(
            title="Backtest Multi-Portefeuilles",
            xaxis_title="Date", yaxis_title="Capital (€)",
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
        )
        st.plotly_chart(fig_bt, use_container_width=True)

        # ===== Camemberts d’allocations (couleurs consistantes) =====
        from plotly.colors import qualitative
        palette = qualitative.Plotly + qualitative.D3 + qualitative.Set3
        labels_master = [row[0] for row in alloc_table]
        unique_labels = [l for i,l in enumerate(labels_master) if l not in labels_master[:i]]
        color_map = {label: palette[i % len(palette)] for i, label in enumerate(unique_labels)}

        st.subheader("Allocations par portefeuille")
        pie_cols = st.columns(len(st.session_state.portfolios))
        for j, pname in enumerate(st.session_state.portfolios):
            labels = [row[0] for row in alloc_table]
            values = [row[j + 1] for row in alloc_table]
            colors = [color_map[l] for l in labels]
            fig_pie = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3, marker=dict(colors=colors))])
            fig_pie.update_layout(title_text=pname)
            pie_cols[j].plotly_chart(fig_pie, use_container_width=True)

        # ===== Stats TWRR (neutralise les apports DCA) =====
        st.subheader("Statistiques des portefeuilles")

        # Rendements journaliers TWRR
        twrr_daily = (capital_data.diff() - flow_data).div(capital_data.shift(1))
        twrr_daily = twrr_daily.replace([np.inf, -np.inf], np.nan).dropna()
        twrr_wi = (1 + twrr_daily).cumprod()

        stats_rows = []
        for col in capital_data.columns:
            series_cap = capital_data[col].dropna()
            r = twrr_daily[col].dropna()
            wi = twrr_wi[col].dropna()
            if len(series_cap) < 2 or r.empty or wi.empty:
                continue

            years = (r.index[-1] - r.index[0]).days / 365.2425
            cagr = wi.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan

            vol_annual = r.std() * np.sqrt(252) if r.std() > 0 else np.nan
            sharpe = (r.mean() / r.std()) * np.sqrt(252) if r.std() > 0 else np.nan

            yearly = (1 + r).resample("YE").prod() - 1
            best_year = yearly.max() if not yearly.empty else np.nan
            worst_year = yearly.min() if not yearly.empty else np.nan

            # MDD sur WI TWRR
            mdd = (wi / wi.cummax() - 1).min()

            stats_rows.append([
                series_cap.iloc[0], series_cap.iloc[-1], cagr, vol_annual,
                best_year, worst_year, mdd, sharpe
            ])

        if stats_rows:
            stats_df = pd.DataFrame(
                stats_rows,
                columns=["Capital de départ (€)", "Capital de fin (€)", "CAGR (TWRR)", "Volatilité (TWRR)",
                         "Meilleure année (TWRR)", "Pire année (TWRR)", "Chute la plus importante (TWRR)", "Sharpe (TWRR, Rf=0)"],
                index=capital_data.columns
            )
            st.dataframe(stats_df.style.format({
                "Capital de départ (€)": "{:,.0f}",
                "Capital de fin (€)": "{:,.0f}",
                "CAGR (TWRR)": "{:.2%}",
                "Volatilité (TWRR)": "{:.2%}",
                "Meilleure année (TWRR)": "{:.2%}",
                "Pire année (TWRR)": "{:.2%}",
                "Chute la plus importante (TWRR)": "{:.2%}",
                "Sharpe (TWRR, Rf=0)": "{:.2f}",
            }), use_container_width=True)
        else:
            st.info("Pas assez de points pour calculer les statistiques.")

