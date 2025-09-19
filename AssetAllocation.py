import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
import streamlit.components.v1 as components
from datetime import datetime

st.set_page_config(page_title="Backtest Multi-Portefeuilles", layout="wide")

# ===============================
# Définition actifs
# ===============================
assets = [
    "Actions mondiales (VT)",
    "Immobilier international ex US (IFGL)",
    "Immobilier US (IYR)",
    "Obligations d'entreprises Global (CORP.L)",
    "Obligations gouvernementales Global (IGLO.L)",
    "Bitcoin",
    "Or (GLD)",
    "Métaux industriels (DBB)",
    "Matières agricoles (DBA)",
    "Private Equity Global (PSP)",
    "Monétaire Global (IGOV)"
]

tickers = {
    "Actions mondiales (VT)": "VT",  # 2008
    "Immobilier international ex US (IFGL)": "IFGL",  # 2008
    "Immobilier US (IYR)": "IYR",  # 2000
    "Obligations d'entreprises Global (CORP.L)": "CORP.L",  # 2012
    "Obligations gouvernementales Global (IGLO.L)": "IGLO.L",  # 2009
    "Bitcoin": "BTC-USD",  # 2014
    "Or (GLD)": "GLD",  # 2004
    "Métaux industriels (DBB)": "DBB",  # 2007
    "Matières agricoles (DBA)": "DBA",  # 2007
    "Private Equity Global (PSP)": "PSP",  # 2006
    "Monétaire Global (IGOV)": "IGOV"  # 2009
}

# ===============================
# Téléchargement unique des données
# ===============================
if "data_prices" not in st.session_state:
    data_raw = yf.download(list(tickers.values()), start="1990-01-01", progress=False).dropna()

    # Gestion robuste Adj Close / Close
    if isinstance(data_raw.columns, pd.MultiIndex):
        top_levels = set([lvl[0] for lvl in data_raw.columns])
        if "Adj Close" in top_levels:
            data_prices = data_raw["Adj Close"].copy()
        elif "Close" in top_levels:
            data_prices = data_raw["Close"].copy()
        else:
            data_prices = data_raw.copy()
    else:
        data_prices = data_raw.copy()

    data_prices = data_prices.sort_index()
    st.session_state["data_prices"] = data_prices

data_prices = st.session_state["data_prices"]

# ===============================
# Sélection de période utilisateur
# ===============================
min_date = data_prices.index.min().date()
max_date = data_prices.index.max().date()

st.title("Analyse des Classes d'Actifs")
date_range = st.date_input(
    "Sélectionnez une période",
    value=[min_date, max_date],
    min_value=min_date,
    max_value=max_date
)
start_date, end_date = date_range
data_filtered = data_prices.loc[str(start_date):str(end_date)].copy()

# ===============================
# Graphique global (rebasing par série)
# ===============================
st.subheader("Évolution des performances des classes d'actifs")

def rebase_series_to_first_valid(s: pd.Series) -> pd.Series:
    s = s.dropna()
    if s.empty:
        return s
    return s / s.iloc[0]

data_norm = data_filtered.apply(rebase_series_to_first_valid, axis=0)

fig = go.Figure()
for asset, tkr in tickers.items():
    if tkr in data_norm.columns and data_norm[tkr].notna().any():
        fig.add_trace(go.Scatter(x=data_norm.index, y=data_norm[tkr], mode='lines', name=asset))
fig.update_layout(
    title="Performances normalisées (base 100 à la première cotation disponible)",
    xaxis_title="Date", yaxis_title="Indice (base 100)",
    template="plotly_white"
)
st.plotly_chart(fig, use_container_width=True)

# ===============================
# Tableau statistiques (fenêtre >= 2 ans)
# ===============================
delta_years = (end_date - start_date).days / 365.25
if delta_years >= 2:
    st.subheader("Statistiques par classe d'actifs")

    stats = []
    # Rendements : garder les NaN là où pas de données, puis drop par colonne au besoin
    returns = data_filtered.pct_change()

    for asset, tkr in tickers.items():
        if tkr not in data_filtered.columns:
            continue
        prices = data_filtered[tkr].dropna()
        if len(prices) < 2:
            continue

        start_val = prices.iloc[0]
        end_val = prices.iloc[-1]
        years = (prices.index[-1] - prices.index[0]).days / 365.25
        cagr = (end_val / start_val) ** (1 / years) - 1 if start_val > 0 and years > 0 else np.nan

        r = returns[tkr].dropna()
        if r.empty:
            continue
        annual_return = r.mean() * 252
        vol_annual = r.std() * np.sqrt(252)

        cum_returns = (1 + r).cumprod()
        high_water_mark = cum_returns.cummax()
        drawdown = (cum_returns / high_water_mark) - 1
        max_drawdown = drawdown.min() if not drawdown.empty else np.nan

        sharpe = (r.mean() / r.std()) * np.sqrt(252) if r.std() > 0 else np.nan

        stats.append([asset, cagr, annual_return, vol_annual, max_drawdown, sharpe])

    if stats:
        stats_df = pd.DataFrame(stats, columns=[
            "Actif", "CAGR", "Rendement Annuel Moyen", "Volatilité", "Max Drawdown", "Sharpe Ratio"
        ]).set_index("Actif")
        st.dataframe(stats_df.style.format({
            "CAGR": "{:.2%}",
            "Rendement Annuel Moyen": "{:.2%}",
            "Volatilité": "{:.2%}",
            "Max Drawdown": "{:.2%}",
            "Sharpe Ratio": "{:.2f}"
        }))

# ===============================
# Visualisation corrélations (D3)
# ===============================
st.subheader("Réseau de corrélations (force-directed)")
colB, colC = st.columns(2)

with colB:
    corr_threshold = st.slider("Seuil de corrélation", 0.0, 1.0, 0.35, 0.05)
with colC:
    perf_lookback_days = st.slider("Fenêtre perf (jours)", 60, 252, 126, 7)

# Préparation des données corrélation
ret = data_filtered.pct_change()
corr_mat = ret.corr(min_periods=60)

nodes, links = [], []
ticker_to_asset = {v: k for k, v in tickers.items()}

# nodes
for tkr in data_filtered.columns:
    if tkr not in ticker_to_asset:
        continue
    series = data_filtered[tkr].dropna()
    if series.empty:
        continue

    lb = min(perf_lookback_days, len(series) - 1) if len(series) > 1 else 1
    perf6m = (series.iloc[-1] / series.iloc[-lb] - 1) if lb > 0 else np.nan

    asset_name = ticker_to_asset[tkr]

    r_series = ret[tkr].dropna()
    vol_ann = r_series.std() * np.sqrt(252) if not r_series.empty else 0.0
    radius = float(np.clip(6 + (vol_ann * 80), 6, 18))

    nodes.append({
        "id": tkr,
        "name": asset_name,  # afficher nom complet
        "perf6m": float(perf6m) if np.isfinite(perf6m) else 0.0,
        "radius": radius
    })

# links
present_tickers = [n["id"] for n in nodes]
for i in range(len(present_tickers)):
    for j in range(i + 1, len(present_tickers)):
        a, b = present_tickers[i], present_tickers[j]
        c = corr_mat.loc[a, b] if (a in corr_mat.index and b in corr_mat.columns) else np.nan
        if pd.isna(c):
            continue
        strength = abs(c)
        if strength >= corr_threshold:
            value = float(5 * (1 - (c + 1) / 2))
            value = max(0.0, min(5.0, value))
            links.append({"source": a, "target": b, "value": value})

meta = {
    "start_corr": str(ret.index.min().date()) if not ret.dropna(how="all").empty else "",
    "end_corr": str(ret.index.max().date()) if not ret.dropna(how="all").empty else "",
    "start_perf6m": str((data_filtered.index[-1] - pd.tseries.offsets.BDay(perf_lookback_days)).date()) if len(data_filtered) else "",
    "end_perf6m": str(data_filtered.index[-1].date()) if len(data_filtered) else "",
    "n_nodes": len(nodes),
    "n_links": len(links)
}

graph_payload = {
    "nodes": nodes,
    "links": links,
    "meta": meta
}

# HTML + JS inline
html = f"""
<div id="corr-app" style="width:100%;">
  <div id="toolbar" style="margin:8px 0; display:flex; gap:8px; align-items:center;">
    <div id="meta" style="font-family: ui-sans-serif, system-ui; font-size:14px; opacity:0.8;"></div>
  </div>
  <div id="stage" style="position:relative;"></div>
</div>

<style>
  #stage .tooltip {{
    position: fixed;
    padding: 8px 10px;
    background: rgba(20,20,20,0.9);
    color: #fff;
    border-radius: 6px;
    pointer-events: none;
    font-size: 12px;
    line-height: 1.2;
  }}
  .node-label {{
    font: 11px ui-sans-serif, system-ui;
    fill: #333;
    pointer-events: none;
  }}
</style>

<script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
<script>
  const DATA = {json.dumps(graph_payload).replace("</", "<\\/")};

  const WIDTH = document.getElementById('stage').clientWidth || 1100;
  const HEIGHT = 720;

  const stage = document.getElementById('stage');
  const meta = document.getElementById('meta');

  function renderForceGraph(container, data) {{
    container.innerHTML = '';

    const color = d3.scaleOrdinal(d3.schemeCategory10);

    const svg = d3.select(container)
      .append('svg')
      .attr('width', WIDTH)
      .attr('height', HEIGHT)
      .attr('viewBox', [0, 0, WIDTH, HEIGHT]);

    const tooltip = d3.select(container)
      .append('div')
      .attr('class', 'tooltip')
      .style('opacity', 0);

    const nodes = data.nodes.map(d => ({{ ...d }}));
    const links = data.links.map(d => ({{ ...d }}));

    const K_LINK_BASE = 100;
    const K_LINK_MULT = 50;
    const CHARGE = -220;
    const COLLIDE_PAD = 6;

    const simulation = d3.forceSimulation(nodes)
      .force('link', d3.forceLink(links)
        .id(d => d.id)
        .distance(d => K_LINK_BASE + K_LINK_MULT * d.value)
        .strength(0.6)
      )
      .force('charge', d3.forceManyBody().strength(CHARGE).distanceMax(300))
      .force('center', d3.forceCenter(WIDTH/2, HEIGHT/2))
      .force('collision', d3.forceCollide().radius(d => (d.radius || 6) + COLLIDE_PAD))
      .on('tick', ticked);

    const link = svg.append('g')
      .attr('stroke', '#999')
      .attr('stroke-opacity', 0.6)
      .selectAll('line')
      .data(links)
      .join('line')
      .attr('stroke-width', d => Math.sqrt(1 + d.value));

    const node = svg.append('g')
      .attr('stroke', '#fff')
      .attr('stroke-width', 1.2)
      .selectAll('circle')
      .data(nodes)
      .join('circle')
      .attr('r', d => d.radius || 6)
      .attr('fill', d => color(d.id))  // couleur par ticker
      .call(d3.drag()
        .on('start', dragstarted)
        .on('drag', dragged)
        .on('end', dragended))
      .on('mousemove', (event, d) => {{
        tooltip.style('opacity', 1)
          .style('left', `${{event.clientX + 10}}px`)
          .style('top', `${{event.clientY + 10}}px`)
          .html(`<b>${{d.name}}</b><br/>Perf 6m: ${{(d.perf6m*100).toFixed(2)}}%`);
      }})
      .on('mouseout', () => tooltip.style('opacity', 0));

    const labels = svg.append('g')
      .selectAll('text')
      .data(nodes)
      .join('text')
      .attr('class', 'node-label')
      .text(d => `${{d.name}}  ${{(d.perf6m*100).toFixed(1)}}%`)
      .attr('text-anchor', 'middle')
      .attr('dy', d => - (d.radius + 4));

    function ticked() {{
      link
        .attr('x1', d => d.source.x)
        .attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x)
        .attr('y2', d => d.target.y);

      node
        .attr('cx', d => d.x)
        .attr('cy', d => d.y);

      labels
        .attr('x', d => d.x)
        .attr('y', d => d.y - (d.radius + 4));
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
  }}

  const m = DATA.meta || {{}};
  document.getElementById('meta').innerHTML =
    `Fenêtre corr: ${{m.start_corr}} → ${{m.end_corr}} · Fenêtre perf: ${{m.start_perf6m}} → ${{m.end_perf6m}} · N=${{m.n_nodes}} · L=${{m.n_links}}`;

  renderForceGraph(stage, DATA);
</script>
"""

components.html(html, height=820, scrolling=True)


# ===============================
# --- Simulation de portefeuilles (inchangée) ---
# ===============================
st.title("Configuration des portefeuilles")

# Mémoire de session
if 'table_rows' not in st.session_state:
    st.session_state.table_rows = 4
if 'portfolios' not in st.session_state:
    st.session_state.portfolios = ["Portefeuille 1", "Portefeuille 2"]

# Gestion lignes
col1, col2 = st.columns(2)
if col1.button("➕ Ajouter un actif (ligne)"):
    st.session_state.table_rows += 1
if col2.button("➖ Supprimer un actif (ligne)"):
    if st.session_state.table_rows > 1:
        st.session_state.table_rows -= 1

# Gestion colonnes
col3, col4 = st.columns(2)
if col3.button("➕ Ajouter un portefeuille (colonne)"):
    new_name = f"Portefeuille {len(st.session_state.portfolios) + 1}"
    st.session_state.portfolios.append(new_name)
if col4.button("➖ Supprimer un portefeuille (colonne)"):
    if len(st.session_state.portfolios) > 1:
        st.session_state.portfolios.pop()

# Noms et capital initial
st.subheader("Nom et capital initial des portefeuilles")
cap_cols = st.columns(len(st.session_state.portfolios))
start_capitals = []
for i in range(len(st.session_state.portfolios)):
    st.session_state.portfolios[i] = cap_cols[i].text_input("Nom", value=st.session_state.portfolios[i], key=f"name_{i}")
for i in range(len(st.session_state.portfolios)):
    capital = cap_cols[i].number_input("Capital initial (€)", value=10000, step=1000, key=f"capital_{i}")
    start_capitals.append(capital)

# Tableau d'allocations
st.subheader("Tableau d'allocations")
columns = st.columns(len(st.session_state.portfolios) + 1)
columns[0].markdown("**Actif**")
for j, portfolio_name in enumerate(st.session_state.portfolios):
    columns[j + 1].markdown(f"**{portfolio_name} (%)**")

alloc_table = []
for i in range(st.session_state.table_rows):
    row = []
    asset_select = columns[0].selectbox(f"Actif {i}", assets, key=f"asset_{i}")
    row.append(asset_select)
    for j in range(len(st.session_state.portfolios)):
        weight_input = columns[j + 1].number_input(
            f"{st.session_state.portfolios[j]} alloc {i}",
            min_value=0, max_value=100, value=0, key=f"alloc_{i}_{j}"
        )
        row.append(weight_input)
    alloc_table.append(row)

# Lancer backtest
if st.button("🚀 Lancer le backtest"):

    # Vérification totaux 100%
    valid = True
    for j in range(len(st.session_state.portfolios)):
        total = sum([row[j + 1] for row in alloc_table])
        if total != 100:
            valid = False
            st.error(
                f"La somme des allocations du portefeuille **{st.session_state.portfolios[j]}** est {total}%, elle doit faire 100%."
            )

    if valid:
        selected_assets = list(set([row[0] for row in alloc_table]))
        data_subset = data_prices[[tickers[a] for a in selected_assets]].dropna()

        data_norm_bt = data_subset / data_subset.iloc[0]
        capital_data = pd.DataFrame(index=data_norm_bt.index)

        for j, portfolio_name in enumerate(st.session_state.portfolios):
            perf = pd.Series(0.0, index=data_norm_bt.index)
            for row in alloc_table:
                asset = row[0]
                weight = row[j + 1] / 100
                perf += data_norm_bt[tickers[asset]] * weight
            capital_data[portfolio_name] = perf * start_capitals[j]

        st.subheader("Évolution du capital des portefeuilles")
        fig = go.Figure()
        for col in capital_data.columns:
            fig.add_trace(go.Scatter(x=capital_data.index, y=capital_data[col], mode='lines', name=col))
        fig.update_layout(title="Backtest Multi-Portefeuilles", xaxis_title="Date", yaxis_title="Capital (€)", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

        # Camemberts
        st.subheader("Allocations des portefeuilles")
        pie_cols = st.columns(len(st.session_state.portfolios))
        for j, portfolio_name in enumerate(st.session_state.portfolios):
            labels = []
            values = []
            for row in alloc_table:
                labels.append(row[0])
                values.append(row[j + 1])
            pie_fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
            pie_fig.update_layout(title_text=portfolio_name)
            pie_cols[j].plotly_chart(pie_fig, use_container_width=True)

        # Statistiques
        st.subheader("Statistiques des portefeuilles")
        stats = []
        returns = capital_data.pct_change().dropna()

        for col in capital_data.columns:
            start_balance = capital_data[col].iloc[0]
            end_balance = capital_data[col].iloc[-1]
            years = (capital_data.index[-1] - capital_data.index[0]).days / 365.25
            cagr = (end_balance / start_balance) ** (1 / years) - 1 if start_balance > 0 and years > 0 else np.nan
            vol_annual = returns[col].std() * np.sqrt(252)
            yearly_perf = capital_data[col].resample('Y').last().pct_change().dropna()
            best_year = yearly_perf.max() if not yearly_perf.empty else np.nan
            worst_year = yearly_perf.min() if not yearly_perf.empty else np.nan
            cum_returns = (1 + returns[col]).cumprod()
            high_water_mark = cum_returns.cummax()
            drawdown = (cum_returns / high_water_mark) - 1
            max_drawdown = drawdown.min() if not drawdown.empty else np.nan
            sharpe = returns[col].mean() / returns[col].std() * np.sqrt(252) if returns[col].std() > 0 else np.nan
            downside_std = returns[col][returns[col] < 0].std() * np.sqrt(252)
            sortino = returns[col].mean() / downside_std * np.sqrt(252) if downside_std > 0 else np.nan

            stats.append(
                [start_balance, end_balance, cagr, vol_annual, best_year, worst_year, max_drawdown, sharpe, sortino]
            )

        stats_df = pd.DataFrame(stats, columns=[
            "Start Balance (€)", "End Balance (€)", "CAGR", "Volatility", "Best Year", "Worst Year",
            "Max Drawdown", "Sharpe Ratio", "Sortino Ratio"
        ], index=capital_data.columns)

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
        }))
