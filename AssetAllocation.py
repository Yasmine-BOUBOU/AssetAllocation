import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Backtest Multi-Portefeuilles", layout="wide")

assets = ["Gold", "US stocks", "Treasuries 20y+", "Commodities"]
tickers = {"Gold": "IAU", "US stocks": "QQQ", "Treasuries 20y+": "TLT", "Commodities": "USCI"}

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

st.title("Configuration des portefeuilles")

# Noms et capital initial
st.subheader("Nom et capital initial des portefeuilles")
cap_cols = st.columns(len(st.session_state.portfolios))
start_capitals = []
for i in range(len(st.session_state.portfolios)):
    st.session_state.portfolios[i] = cap_cols[i].text_input("Nom", value=st.session_state.portfolios[i],
                                                            key=f"name_{i}")
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
                f"La somme des allocations du portefeuille **{st.session_state.portfolios[j]}** est {total}%, elle doit faire 100%.")

    if valid:
        selected_assets = list(set([row[0] for row in alloc_table]))
        data_raw = yf.download([tickers[a] for a in selected_assets], start="2015-01-01")

        if "Adj Close" in data_raw:
            data_prices = data_raw["Adj Close"].copy()
        else:
            data_prices = data_raw["Close"].copy()

        data_prices = data_prices.dropna()
        data_norm = data_prices / data_prices.iloc[0]

        capital_data = pd.DataFrame(index=data_norm.index)

        for j, portfolio_name in enumerate(st.session_state.portfolios):
            perf = pd.Series(0.0, index=data_norm.index)
            for row in alloc_table:
                asset = row[0]
                weight = row[j + 1] / 100
                perf += data_norm[tickers[asset]] * weight
            capital_data[portfolio_name] = perf * start_capitals[j]

        st.subheader("Évolution du capital des portefeuilles")
        fig = go.Figure()
        for col in capital_data.columns:
            fig.add_trace(go.Scatter(x=capital_data.index, y=capital_data[col],
                                     mode='lines', name=col))
        fig.update_layout(title="Backtest Multi-Portefeuilles",
                          xaxis_title="Date", yaxis_title="Capital (€)",
                          template="plotly_white")
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
            cagr = (end_balance / start_balance) ** (1 / years) - 1
            vol_annual = returns[col].std() * np.sqrt(252)
            yearly_perf = capital_data[col].resample('Y').last().pct_change().dropna()
            best_year = yearly_perf.max()
            worst_year = yearly_perf.min()
            cum_returns = (1 + returns[col]).cumprod()
            high_water_mark = cum_returns.cummax()
            drawdown = (cum_returns / high_water_mark) - 1
            max_drawdown = drawdown.min()
            sharpe = returns[col].mean() / returns[col].std() * np.sqrt(252)
            downside_std = returns[col][returns[col] < 0].std() * np.sqrt(252)
            sortino = returns[col].mean() / downside_std * np.sqrt(252) if downside_std > 0 else np.nan

            stats.append(
                [start_balance, end_balance, cagr, vol_annual, best_year, worst_year, max_drawdown, sharpe, sortino])

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
