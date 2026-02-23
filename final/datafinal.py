import pandas as pd
import sqlite3
import plotly.express as px
from dash import Dash, dcc, html

# =========================================
# ÉTAPE 1 — CHARGEMENT & EXPLORATION
# =========================================

# Chargement des CSV
logs         = pd.read_csv("logs.csv")
network      = pd.read_csv("network_logs.csv")
clients      = pd.read_csv("clients.csv")
transactions = pd.read_csv("transactions.csv")

# Tables disponibles dans la base brute
conn = sqlite3.connect("datalake.db")
print("Tables dans datalake.db :")
print(pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn))
conn.close()

# Fonction réutilisable pour explorer un dataframe
def explorer_df(nom, df):
    print(f"\n=== {nom} ===")
    print(f"  Dimensions   : {df.shape}")
    print(f"  Types        :\n{df.dtypes}")
    print(f"  Valeurs nulles :\n{df.isnull().sum()}")
    print(f"  Doublons     : {df.duplicated().sum()}")

explorer_df("LOGS",         logs)
explorer_df("NETWORK",      network)
explorer_df("CLIENTS",      clients)
explorer_df("TRANSACTIONS", transactions)

# Vérification des plages valides
print("\n=== VALIDATIONS MÉTIER ===")
print(f"Ports       — min: {network['port'].min()}, max: {network['port'].max()}")
print(f"Bytes sent  — min: {network['bytes_sent'].min()}")
print(f"Bytes recv  — min: {network['bytes_received'].min()}")
print(f"Montants    — min: {transactions['amount'].min()}, max: {transactions['amount'].max()}")

# Valeurs uniques des colonnes catégorielles
print("\n=== VALEURS UNIQUES ===")
print(f"Events logs      : {logs['event'].unique()}")
print(f"Status logs      : {logs['status'].unique()}")
print(f"Status network   : {network['status'].unique()}")
print(f"Pays clients     : {clients['country'].unique()}")
print(f"Modes paiement   : {transactions['pmntMode'].unique()}")

# =========================================
# ÉTAPE 2 — NETTOYAGE
# =========================================

# Renommer les colonnes mal nommées
clients      = clients.rename(columns={"Usr-Ag": "user_age"})
transactions = transactions.rename(columns={"tX_ID": "transaction_id", "pmntMode": "payment_mode"})

# Convertir les dates
logs["timestamp"]     = pd.to_datetime(logs["timestamp"])
transactions["date"]  = pd.to_datetime(transactions["date"], dayfirst=True)

# Isoler et archiver les données aberrantes
ports_aberrants    = network[(network["port"] < 0) | (network["port"] > 65535)]
bytes_aberrants    = network[network["bytes_sent"] < 0]
montants_aberrants = transactions[transactions["amount"] < 0]

ports_aberrants.to_csv("archive_ports.csv", index=False)
bytes_aberrants.to_csv("archive_bytes.csv", index=False)
montants_aberrants.to_csv("archive_montants.csv", index=False)

print(f"\nAberrants — ports: {len(ports_aberrants)}, bytes: {len(bytes_aberrants)}, montants: {len(montants_aberrants)}")

# Données nettoyées
network_clean      = network[(network["port"].between(0, 65535)) & (network["bytes_sent"] >= 0)]
transactions_clean = transactions[transactions["amount"] >= 0]

print("\n=== DONNÉES PROPRES ===")
for nom, df in [("logs", logs), ("network", network_clean), ("clients", clients), ("transactions", transactions_clean)]:
    print(f"  {nom} : {df.shape}")

# =========================================
# ÉTAPE 3 — STOCKAGE EN BASE SQL
# =========================================

conn = sqlite3.connect("datalake_clean.db")

logs.to_sql("logs",          conn, if_exists="replace", index=False)
network_clean.to_sql("network_log",  conn, if_exists="replace", index=False)
clients.to_sql("client",     conn, if_exists="replace", index=False)
transactions_clean.to_sql("transactions", conn, if_exists="replace", index=False)

# Index pour les jointures
for sql in [
    "CREATE INDEX IF NOT EXISTS idx_logs_user_id   ON logs(user_id)",
    "CREATE INDEX IF NOT EXISTS idx_network_log_id ON network_log(log_id)",
    "CREATE INDEX IF NOT EXISTS idx_client_id      ON client(client_id)",
    "CREATE INDEX IF NOT EXISTS idx_transactions   ON transactions(date)",
]:
    conn.execute(sql)

# Vues SQL
for sql in [
    """CREATE VIEW IF NOT EXISTS vue_top_users AS
       SELECT user_id, COUNT(*) as nb_connexions
       FROM logs GROUP BY user_id ORDER BY nb_connexions DESC""",
    """CREATE VIEW IF NOT EXISTS vue_top_ip AS
       SELECT src_ip, COUNT(*) as nb
       FROM network_log GROUP BY src_ip ORDER BY nb DESC""",
    """CREATE VIEW IF NOT EXISTS vue_trafic_heure AS
       SELECT SUBSTR(timestamp, 1, 13) as heure, COUNT(*) as nb_connexions
       FROM logs GROUP BY heure ORDER BY heure""",
]:
    conn.execute(sql)
print("Vues créées ✓")

conn.commit()
print("\nTables en base :")
print(pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn))
conn.close()

# =========================================
# ÉTAPE 4 — KPI
# =========================================

conn = sqlite3.connect("datalake_clean.db")

kpi1 = pd.read_sql("""
    SELECT user_id, COUNT(*) as nb_connexions
    FROM logs GROUP BY user_id ORDER BY nb_connexions DESC LIMIT 10
""", conn)

kpi2 = pd.read_sql("""
    SELECT SUBSTR(timestamp, 1, 13) as heure, COUNT(*) as nb_connexions
    FROM logs GROUP BY heure ORDER BY heure
""", conn)

kpi3 = pd.read_sql("""
    SELECT src_ip, COUNT(*) as nb
    FROM network_log GROUP BY src_ip ORDER BY nb DESC LIMIT 10
""", conn)

kpi4 = pd.read_sql("""
    SELECT port, COUNT(*) as nb
    FROM network_log GROUP BY port ORDER BY nb DESC LIMIT 10
""", conn)

kpi5 = pd.read_sql("""
    SELECT payment_mode, COUNT(*) as nb, ROUND(SUM(amount), 2) as total
    FROM transactions GROUP BY payment_mode ORDER BY nb DESC
""", conn)

for nom, kpi in [("TOP 10 UTILISATEURS", kpi1), ("TRAFIC PAR HEURE", kpi2),
                 ("TOP 10 IP SOURCES", kpi3), ("TOP 10 PORTS", kpi4), ("PAIEMENTS", kpi5)]:
    print(f"\n=== {nom} ===")
    print(kpi)

# Sauvegarder les KPI dans la base
kpi1.to_sql("kpi_top_users",    conn, if_exists="replace", index=False)
kpi2.to_sql("kpi_trafic_heure", conn, if_exists="replace", index=False)
kpi3.to_sql("kpi_top_ip",       conn, if_exists="replace", index=False)
kpi4.to_sql("kpi_top_ports",    conn, if_exists="replace", index=False)
kpi5.to_sql("kpi_paiements",    conn, if_exists="replace", index=False)

conn.close()

# =========================================
# ÉTAPE 5 — DASHBOARD INTERACTIF
# =========================================

# Heatmap : top 15 IP sources × top 15 ports
conn = sqlite3.connect("datalake_clean.db")
heatmap_data = pd.read_sql("""
    SELECT src_ip, port, COUNT(*) as nb
    FROM network_log GROUP BY src_ip, port
""", conn)
conn.close()

top_ips   = heatmap_data.groupby("src_ip")["nb"].sum().nlargest(15).index
top_ports = heatmap_data.groupby("port")["nb"].sum().nlargest(15).index
pivot = (
    heatmap_data[heatmap_data["src_ip"].isin(top_ips) & heatmap_data["port"].isin(top_ports)]
    .pivot_table(index="src_ip", columns="port", values="nb", fill_value=0)
)

# Graphiques — Base SQL
fig1 = px.line(kpi2, x="heure",       y="nb_connexions", title="Trafic réseau par heure", markers=True)
fig2 = px.bar( kpi1, x="user_id",     y="nb_connexions", title="Top 10 utilisateurs actifs")
fig3 = px.bar( kpi3, x="src_ip",      y="nb",            title="Top 10 IP sources")
fig4 = px.bar( kpi4, x="port",        y="nb",            title="Top 10 ports utilisés")
fig5 = px.pie( kpi5, names="payment_mode", values="total", title="Répartition des paiements")
fig6 = px.imshow(pivot, title="Heatmap — IP sources × Ports", aspect="auto",
                 color_continuous_scale="Blues", labels={"color": "Connexions"})

# Graphiques — Analyse CSV (valeurs nulles par colonne)
def fig_nulls(nom, df):
    nulls = df.isnull().sum().reset_index()
    nulls.columns = ["colonne", "nulls"]
    return px.bar(nulls, x="colonne", y="nulls", title=f"Valeurs nulles — {nom}")

fig_csv1 = fig_nulls("Logs",         logs)
fig_csv2 = fig_nulls("Network",      network)
fig_csv3 = fig_nulls("Clients",      clients)
fig_csv4 = fig_nulls("Transactions", transactions_clean)

# Dashboard avec onglets
app = Dash(__name__)
app.layout = html.Div([
    html.H2("Dashboard Réseau & Transactions", style={"textAlign": "center", "fontFamily": "sans-serif"}),
    dcc.Tabs([
        dcc.Tab(label="Base — Trafic/heure",  children=[dcc.Graph(figure=fig1)]),
        dcc.Tab(label="Base — Utilisateurs",  children=[dcc.Graph(figure=fig2)]),
        dcc.Tab(label="Base — Top IP",        children=[dcc.Graph(figure=fig3)]),
        dcc.Tab(label="Base — Top ports",     children=[dcc.Graph(figure=fig4)]),
        dcc.Tab(label="Base — Paiements",     children=[dcc.Graph(figure=fig5)]),
        dcc.Tab(label="Base — Heatmap",       children=[dcc.Graph(figure=fig6)]),
        dcc.Tab(label="CSV — Logs",           children=[dcc.Graph(figure=fig_csv1)]),
        dcc.Tab(label="CSV — Network",        children=[dcc.Graph(figure=fig_csv2)]),
        dcc.Tab(label="CSV — Clients",        children=[dcc.Graph(figure=fig_csv3)]),
        dcc.Tab(label="CSV — Transactions",   children=[dcc.Graph(figure=fig_csv4)]),
    ])
])

print("\nDashboard disponible sur http://127.0.0.1:8050")
app.run(debug=False)
