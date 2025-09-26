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

# ===============================
# Définition actifs
# ===============================
assets = [
    "Actions mondiales (VT)",
    "Private Equity Global (PSP)",
    "Actions françaises (CAC.PA)",
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
    "Actions françaises (CAC.PA)": "CAC.PA",  # 2008
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
    "VT": 2008, "PSP": 2008, "CAC.PA": 2008, "SPY": 1993,
    "IFGL": 2008, "IYR": 2000, "LQD": 2002, "IGLO.L": 2009,
    "GLD": 2004, "DBB": 2007, "DBA": 2007, "IGOV": 2009
}

# ===============================
# Sélection des actifs (tableau avec cases à cocher)
# ===============================
st.title("Sélection des actifs")

hdr = st.columns([3, 1, 1])
hdr[0].markdown("**Actif**")
hdr[1].markdown("**Début (indicatif)**")
hdr[2].markdown("**Télécharger ?**")

default_checked = {"VT", "LQD", "GLD"}
selected_assets = []
for asset in assets:
    tkr = tickers[asset]
    c1, c2, c3 = st.columns([3, 1, 1])
    c1.write(asset)
    c2.write(inception_years_hint.get(tkr, "n/a"))
    checked = c3.checkbox(
        "",
        value=(tkr in default_checked),
        key=f"chk_{tkr}"
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
    data_raw = yf.download(sel_tickers, start="1990-01-01", progress=False).dropna(how="all")
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
    "Sélectionnez une période",
    value=[common_start, max_date],
    min_value=common_start,
    max_value=max_date
)
start_date, end_date = date_range
data_filtered = data_prices.loc[str(start_date):str(end_date)].copy()

# ===============================
# Graphique global (rebasing)
# ===============================
st.subheader("Évolution des performances des classes d'actifs")

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
# Corrélations (matrice interactive + graph D3 sans perf)
# ===============================
st.subheader("Corrélations")

# Matrice de corrélation interactive (compacte)
returns = data_filtered.pct_change().dropna(how="all").dropna(axis=1, how="any")
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

    st.markdown("**Réseau de corrélations (force-directed)**")
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
tab_cols[0].markdown("**Actif**")
for j, pname in enumerate(st.session_state.portfolios):
    tab_cols[j + 1].markdown(f"**{pname} (%)**")

alloc_table = []
for i in range(st.session_state.table_rows):
    row = []
    asset_select = tab_cols[0].selectbox(f"Actif {i}", selected_assets, key=f"asset_{i}")
    row.append(asset_select)
    for j in range(len(st.session_state.portfolios)):
        w = tab_cols[j + 1].number_input(
            f"{st.session_state.portfolios[j]} alloc {i}",
            min_value=0, max_value=100, value=0, key=f"alloc_{i}_{j}"
        )
        row.append(w)
    alloc_table.append(row)

# Paramètres de backtest
st.subheader("Paramètres de backtest")
pb1, pb2, pb3 = st.columns(3)
strategy = pb1.selectbox("Stratégie", ["Buy & Hold", "Rebalancing (N jours)"], index=0)
reb_days = None
if strategy.startswith("Rebalancing"):
    reb_days = int(pb2.number_input("Intervalle N (jours)", value=63, min_value=2, max_value=252, step=1))
costs_bps = float(pb3.number_input("Frais de réallocation (bps)", value=0, min_value=0, max_value=200, step=1))

# Fonction de backtest
def backtest_with_rebalance(prices: pd.DataFrame,
                            weights_pct: dict,
                            start_capital: float,
                            reb_days: int | None = None,
                            costs_bps: float = 0.0) -> pd.Series:
    cols = list(weights_pct.keys())
    prices = prices[cols].copy()
    rets = prices.pct_change().fillna(0.0)
    w = (pd.Series(weights_pct, dtype=float) / 100.0).reindex(cols).fillna(0.0)

    dates = prices.index
    holdings = (start_capital * w) / prices.iloc[0]
    port_val = pd.Series(index=dates, dtype=float)
    port_val.iloc[0] = float((holdings * prices.iloc[0]).sum())

    last_reb = dates[0]
    for t in range(1, len(dates)):
        # évolution naturelle
        holdings *= (1.0 + rets.iloc[t])
        pv = float((holdings * prices.iloc[t]).sum())

        # rebalancement si demandé
        if reb_days and (dates[t] - last_reb).days >= reb_days:
            target_val = pv * w
            target_hold = target_val / prices.iloc[t]
            trade = target_hold - holdings
            if costs_bps > 0:
                trade_cash = (trade.abs() * prices.iloc[t]).sum()
                cost = trade_cash * (costs_bps / 10_000.0)
                pv -= cost
                target_val = pv * w
                target_hold = target_val / prices.iloc[t]
                trade = target_hold - holdings
            holdings += trade
            last_reb = dates[t]

        port_val.iloc[t] = float((holdings * prices.iloc[t]).sum())

    return port_val

# ===============================
# Lancer backtest
# ===============================
if st.button("🚀 Lancer le backtest"):
    # Vérif totaux = 100%
    valid = True
    for j in range(len(st.session_state.portfolios)):
        total = sum([row[j + 1] for row in alloc_table])
        if total != 100:
            st.error(f"Le portefeuille **{st.session_state.portfolios[j]}** fait {total}%, il doit faire 100%.")
            valid = False

    if valid:
        # Sous-ensemble propre (intersection des séries)
        use_cols = [tickers[a] for a in selected_assets]
        data_subset = data_filtered[use_cols].dropna()

        if data_subset.empty:
            st.warning("Pas assez de données sur la période sélectionnée pour exécuter le backtest.")
            st.stop()

        capital_data = pd.DataFrame(index=data_subset.index)

        for j, pname in enumerate(st.session_state.portfolios):
            weights_pct = {tickers[row[0]]: row[j + 1] for row in alloc_table if row[j + 1] > 0}
            if not weights_pct:
                st.warning(f"Aucun poids non nul pour {pname}.")
                continue

            series_val = backtest_with_rebalance(
                prices=data_subset,
                weights_pct=weights_pct,
                start_capital=start_capitals[j],
                reb_days=None if strategy == "Buy & Hold" else reb_days,
                costs_bps=costs_bps
            )
            capital_data[pname] = series_val

        # Évolution du capital
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

        # ===== Camemberts d’allocations =====
        st.subheader("Allocations par portefeuille")
        pie_cols = st.columns(len(st.session_state.portfolios))
        for j, pname in enumerate(st.session_state.portfolios):
            labels = [row[0] for row in alloc_table]
            values = [row[j + 1] for row in alloc_table]
            fig_pie = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
            fig_pie.update_layout(title_text=pname)
            pie_cols[j].plotly_chart(fig_pie, use_container_width=True)

        # ===== Tableau de statistiques =====
        st.subheader("Statistiques des portefeuilles")
        stats_rows = []
        daily = capital_data.pct_change().dropna(how="all")
        for col in capital_data.columns:
            series = capital_data[col].dropna()
            if len(series) < 2:
                continue
            start_balance = series.iloc[0]
            end_balance = series.iloc[-1]
            years = (series.index[-1] - series.index[0]).days / 365.25
            cagr = (end_balance / start_balance) ** (1 / years) - 1 if start_balance > 0 and years > 0 else np.nan

            r = daily[col].dropna()
            vol_annual = r.std() * np.sqrt(252) if not r.empty else np.nan

            yearly = series.resample('Y').last().pct_change().dropna()
            best_year = yearly.max() if not yearly.empty else np.nan
            worst_year = yearly.min() if not yearly.empty else np.nan

            if not r.empty:
                cum = (1 + r).cumprod()
                hwm = cum.cummax()
                drawdown = (cum / hwm) - 1
                max_dd = drawdown.min()
            else:
                max_dd = np.nan

            sharpe = (r.mean() / r.std()) * np.sqrt(252) if (not r.empty and r.std() > 0) else np.nan
            downside_std = r[r < 0].std() * np.sqrt(252) if not r.empty else np.nan
            sortino = (r.mean() / downside_std) * np.sqrt(252) if (downside_std is not None and downside_std > 0) else np.nan

            stats_rows.append([
                start_balance, end_balance, cagr, vol_annual,
                best_year, worst_year, max_dd, sharpe, sortino
            ])

        if stats_rows:
            stats_df = pd.DataFrame(
                stats_rows,
                columns=["Start Balance (€)", "End Balance (€)", "CAGR", "Volatility",
                         "Best Year", "Worst Year", "Max Drawdown", "Sharpe Ratio", "Sortino Ratio"],
                index=capital_data.columns
            )
            st.dataframe(stats_df.style.format({
                "Start Balance (€)": "{:,.0f}",
                "End Balance (€)": "{:,.0f}",
                "CAGR": "{:.2%}",
                "Volatility": "{:.2%}",
                "Best Year": "{:.2%}",
                "Worst Year": "{:.2%}",
                "Max Drawdown": "{:.2%}",
                "Sharpe Ratio": "{:.2f}",
                "Sortino Ratio": "{:.2f}"
            }), use_container_width=True)

            # ===== Téléchargement CSV =====
            st.download_button(
                label="📥 Télécharger les résultats (CSV)",
                data=capital_data.to_csv().encode("utf-8"),
                file_name="backtest_results.csv",
                mime="text/csv"
            )
        else:
            st.info("Pas assez de points pour calculer les statistiques.")
