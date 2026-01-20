import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Configuration graphiques simple
plt.style.use('default')  # Style par défaut, simple

print("="*80)
print(" analyse financiere - microsoft corporation (msft)")
print(" periode : 2009-2024 | objectif : recommandation d'investissement")
print("="*80)

# ================================================================================
# ÉTAPE 1 : DÉFINIR LA PROBLÉMATIQUE
# ================================================================================
print("\n" + "="*80)
print("etape 1 : problematique et objectifs")
print("="*80)

print("""
questions d'analyse :
1. performance : quel est le rendement total et annualise sur 15 ans ?
2. risque : quelle est la volatilite et le drawdown maximum ?
3. tendance : l'action est-elle en tendance haussiere ou baissiere ?
4. recommandation : faut-il acheter, vendre ou attendre ?

kpi a calculer :
- rendement total et cagr
- volatilite annualisee
- sharpe ratio
- drawdown maximum
- position vs moyennes mobiles (sma 50/200)
""")

# ================================================================================
# ÉTAPE 2 : COLLECTER LES DONNÉES (SIMULATION BASÉE SUR DONNÉES RÉELLES MSFT)
# ================================================================================
print("\n" + "="*80)
print("etape 2 : chargement des donnees")
print("="*80)

# Génération de données réalistes basées sur les vraies performances de Microsoft
np.random.seed(42)
dates = pd.date_range(start='2009-01-01', end='2024-01-01', freq='B')  # Business days

# Paramètres réalistes pour Microsoft
prix_initial = 20.0
tendance_annuelle = 0.19  # ~19% par an (CAGR réel de MSFT)
volatilite_quotidienne = 0.015  # ~1.5% par jour

# Simulation d'un processus de marche aléatoire avec tendance
rendements = np.random.normal(tendance_annuelle/252, volatilite_quotidienne, len(dates))

# Ajout d'événements spécifiques (crash COVID-19 en 2020)
covid_start = (dates >= '2020-02-15') & (dates <= '2020-03-23')
rendements[covid_start] = rendements[covid_start] - 0.03

# Calcul des prix
prix = prix_initial * np.exp(np.cumsum(rendements))

# Ajustement pour avoir un prix final proche de la réalité (~380$ fin 2023)
facteur = 380 / prix[-1]
prix = prix * facteur

# Simulation des autres colonnes (Open, High, Low, Volume)
data = pd.DataFrame({
    'Open': prix * (1 + np.random.normal(0, 0.003, len(dates))),
    'High': prix * (1 + np.abs(np.random.normal(0, 0.008, len(dates)))),
    'Low': prix * (1 - np.abs(np.random.normal(0, 0.008, len(dates)))),
    'Close': prix,
    'Volume': np.random.randint(15000000, 50000000, len(dates)),
    'Adj Close': prix
}, index=dates)

# S'assurer que High >= Low et Close entre les deux
data['High'] = data[['Open', 'High', 'Close']].max(axis=1)
data['Low'] = data[['Open', 'Low', 'Close']].min(axis=1)

print(f"\ndonnees chargees avec succes")
print(f"periode : {data.index[0].strftime('%Y-%m-%d')} a {data.index[-1].strftime('%Y-%m-%d')}")
print(f"nombre de jours de trading : {len(data)}")
print(f"annees couvertes : {(data.index[-1] - data.index[0]).days / 365.25:.1f} ans")

print("\n=== apercu des premieres lignes ===")
print(data.head())

# ================================================================================
# ÉTAPE 3 : NETTOYER ET PRÉPARER LES DONNÉES
# ================================================================================
print("\n" + "="*80)
print("etape 3 : nettoyage et preparation")
print("="*80)

# 3.1 Évaluation de la qualité
print("\n--- 3.1 evaluation de la qualite ---")
print(f"valeurs manquantes : {data.isnull().sum().sum()}")
print(f"doublons : {data.index.duplicated().sum()}")

incohérences = data[(data['High'] < data['Low']) | 
                    (data['Close'] > data['High']) | 
                    (data['Close'] < data['Low'])]
print(f"lignes incoherentes : {len(incohérences)}")

# 3.2 Transformation des données
data.index = pd.to_datetime(data.index)
data = data.sort_index()

# 3.3 Création des variables dérivées
print("\n--- 3.3 creation des variables derivees ---")

# Variables de rendement
data['Daily_Return'] = data['Close'].pct_change() * 100
data['Cumulative_Return'] = ((1 + data['Close'].pct_change()).cumprod() - 1) * 100
data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1))

# Variables temporelles
data['Year'] = data.index.year
data['Month'] = data.index.month
data['Quarter'] = data.index.quarter

# Variables de volatilité
data['Volatility_30d'] = data['Daily_Return'].rolling(window=30).std()
data['Volatility_90d'] = data['Daily_Return'].rolling(window=90).std()
data['Daily_Range'] = ((data['High'] - data['Low']) / data['Close']) * 100

# Moyennes mobiles
data['SMA_20'] = data['Close'].rolling(window=20).mean()
data['SMA_50'] = data['Close'].rolling(window=50).mean()
data['SMA_200'] = data['Close'].rolling(window=200).mean()

# Distance au plus haut historique
max_historical = data['Close'].cummax()
data['Distance_to_ATH'] = ((data['Close'] - max_historical) / max_historical) * 100

# Drawdown (distance depuis le pic historique)
data['Drawdown'] = ((data['Close'] - max_historical) / max_historical) * 100

print(f"{len(data.columns)} colonnes creees")
print(f"nouvelles variables : daily_return, cumulative_return, sma_20/50/200, volatility, drawdown")

# ================================================================================
# ÉTAPE 4 : EXPLORER ET VISUALISER
# ================================================================================
print("\n" + "="*80)
print("etape 4 : exploration et visualisation")
print("="*80)

# 4.1 Statistiques descriptives
print("\n--- 4.1 statistiques sur le prix de cloture ---")
print(f"prix minimum : ${data['Close'].min():.2f}")
print(f"prix maximum : ${data['Close'].max():.2f}")
print(f"prix moyen : ${data['Close'].mean():.2f}")
print(f"prix median : ${data['Close'].median():.2f}")
print(f"ecart-type : ${data['Close'].std():.2f}")

# 4.2 Analyse temporelle
print("\n--- 4.2 rendements par annee ---")
rendements_annuels = data.groupby('Year')['Daily_Return'].apply(
    lambda x: ((1 + x/100).prod() - 1) * 100
)
print(rendements_annuels)

# 4.3 Visualisations
print("\n--- 4.3 creation des graphiques ---")

# Graphique : Prix et moyennes mobiles + autres analyses
fig, axes = plt.subplots(3, 2, figsize=(16, 12))
fig.suptitle('ANALYSE FINANCIÈRE MICROSOFT (MSFT) - 2009-2024', fontsize=14)

# Prix de clôture et moyennes mobiles
ax1 = axes[0, 0]
ax1.plot(data.index, data['Close'], label='Prix de clôture', linewidth=1.5)
ax1.plot(data.index, data['SMA_50'], label='SMA 50 jours', linewidth=1, alpha=0.8)
ax1.plot(data.index, data['SMA_200'], label='SMA 200 jours', linewidth=1, alpha=0.8)
ax1.set_title('Prix de clôture et moyennes mobiles')
ax1.set_xlabel('Date')
ax1.set_ylabel('Prix ($)')
ax1.legend()
ax1.grid(alpha=0.3)

# Rendements annuels
ax2 = axes[0, 1]
colors = ['green' if x > 0 else 'red' for x in rendements_annuels]
ax2.bar(rendements_annuels.index, rendements_annuels.values, color=colors, alpha=0.7)
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax2.axhline(y=rendements_annuels.mean(), color='blue', linestyle='--', linewidth=1, 
            label=f'Moyenne: {rendements_annuels.mean():.1f}%')
ax2.set_title('Rendements annuels (%)')
ax2.set_xlabel('Année')
ax2.set_ylabel('Rendement (%)')
ax2.legend()
ax2.grid(alpha=0.3)

# Volatilité dans le temps
ax3 = axes[1, 0]
ax3.plot(data.index, data['Volatility_30d'], label='Volatilité 30 jours', linewidth=1)
ax3.axhline(y=data['Volatility_30d'].mean(), color='blue', linestyle='--', linewidth=1, 
            label=f'Moyenne: {data["Volatility_30d"].mean():.2f}%')
ax3.set_title('Volatilité mobile 30 jours')
ax3.set_xlabel('Date')
ax3.set_ylabel('Volatilité (%)')
ax3.legend()
ax3.grid(alpha=0.3)

# Drawdown
ax4 = axes[1, 1]
ax4.fill_between(data.index, 0, data['Drawdown'], color='red', alpha=0.3)
ax4.plot(data.index, data['Drawdown'], color='darkred', linewidth=1)
ax4.set_title('Drawdown - Distance au plus haut historique')
ax4.set_xlabel('Date')
ax4.set_ylabel('Drawdown (%)')
ax4.grid(alpha=0.3)

# Distribution des rendements quotidiens
ax5 = axes[2, 0]
ax5.hist(data['Daily_Return'].dropna(), bins=100, alpha=0.7, edgecolor='black')
ax5.axvline(x=0, color='red', linestyle='--', linewidth=1)
ax5.axvline(x=data['Daily_Return'].mean(), color='green', linestyle='--', linewidth=1, 
            label=f'Moyenne: {data["Daily_Return"].mean():.3f}%')
ax5.set_title('Distribution des rendements quotidiens')
ax5.set_xlabel('Rendement quotidien (%)')
ax5.set_ylabel('Fréquence')
ax5.legend()
ax5.grid(alpha=0.3)

# Rendement cumulé
ax6 = axes[2, 1]
ax6.plot(data.index, data['Cumulative_Return'], linewidth=2)
ax6.fill_between(data.index, 0, data['Cumulative_Return'], alpha=0.2)
ax6.set_title('Rendement cumulé depuis 2009')
ax6.set_xlabel('Date')
ax6.set_ylabel('Rendement cumulé (%)')
ax6.grid(alpha=0.3)

plt.tight_layout()

# ================================================================================
# ÉTAPE 5 : ANALYSER ET CALCULER LES KPI
# ================================================================================
print("\n" + "="*80)
print("etape 5 : calcul des kpi")
print("="*80)

# 5.1 KPI de Performance
print("\n--- 5.1 kpi de performance ---")

prix_initial = data['Close'].iloc[0]
prix_final = data['Close'].iloc[-1]
rendement_total = ((prix_final - prix_initial) / prix_initial) * 100
nombre_annees = (data.index[-1] - data.index[0]).days / 365.25
cagr = ((prix_final / prix_initial) ** (1 / nombre_annees) - 1) * 100

print(f"prix initial (2009) : ${prix_initial:.2f}")
print(f"prix final (2024) : ${prix_final:.2f}")
print(f"rendement total : {rendement_total:.2f}%")
print(f"cagr (rendement annualise) : {cagr:.2f}%")
print(f"rendement moyen annuel : {rendements_annuels.mean():.2f}%")
print(f"meilleure annee : {rendements_annuels.idxmax()} ({rendements_annuels.max():.2f}%)")
print(f"pire annee : {rendements_annuels.idxmin()} ({rendements_annuels.min():.2f}%)")

# 5.2 KPI de Risque
print("\n--- 5.2 kpi de risque ---")

volatilite_quotidienne = data['Daily_Return'].std()
volatilite_annualisee = volatilite_quotidienne * np.sqrt(252)
drawdown_max = data['Drawdown'].min()
drawdown_max_date = data['Drawdown'].idxmin()

print(f"volatilite quotidienne : {volatilite_quotidienne:.2f}%")
print(f"volatilite annualisee : {volatilite_annualisee:.2f}%")
print(f"drawdown maximum : {drawdown_max:.2f}%")
print(f"date du drawdown max : {drawdown_max_date.strftime('%Y-%m-%d')}")

# Sharpe Ratio (taux sans risque de 2%)
taux_sans_risque = 2.0
rendement_moyen_annuel = data['Daily_Return'].mean() * 252
sharpe_ratio = (rendement_moyen_annuel - taux_sans_risque) / volatilite_annualisee

print(f"sharpe ratio : {sharpe_ratio:.2f}")

# Ratio rendement/risque
ratio_rendement_risque = cagr / volatilite_annualisee
print(f"ratio rendement/risque : {ratio_rendement_risque:.2f}")

# 5.3 KPI Techniques
print("\n--- 5.3 kpi techniques (position actuelle) ---")

prix_actuel = data['Close'].iloc[-1]
sma_50_actuel = data['SMA_50'].iloc[-1]
sma_200_actuel = data['SMA_200'].iloc[-1]

position_vs_sma50 = ((prix_actuel - sma_50_actuel) / sma_50_actuel) * 100
position_vs_sma200 = ((prix_actuel - sma_200_actuel) / sma_200_actuel) * 100

print(f"prix actuel : ${prix_actuel:.2f}")
print(f"sma 50 jours : ${sma_50_actuel:.2f}")
print(f"sma 200 jours : ${sma_200_actuel:.2f}")
print(f"position vs sma 50 : {position_vs_sma50:+.2f}%")
print(f"position vs sma 200 : {position_vs_sma200:+.2f}%")

# Tendance
if prix_actuel > sma_50_actuel > sma_200_actuel:
    tendance = "HAUSSIÈRE FORTE"
    signal_technique = "ACHAT"
elif prix_actuel > sma_50_actuel:
    tendance = "HAUSSIÈRE MODÉRÉE"
    signal_technique = "ACHAT"
elif prix_actuel < sma_50_actuel < sma_200_actuel:
    tendance = "BAISSIÈRE FORTE"
    signal_technique = "VENTE"
else:
    tendance = "MIXTE/CONSOLIDATION"
    signal_technique = "ATTENTE"

print(f"\ntendance actuelle : {tendance}")
print(f"signal technique : {signal_technique}")

# 5.4 Tableau récapitulatif des KPI
print("\n" + "="*80)
print("tableau recapitulatif des kpi")
print("="*80)

kpi_data = {
    'KPI': [
        'Rendement Total',
        'CAGR (Rendement Annualisé)',
        'Rendement Moyen Annuel',
        'Meilleure Année',
        'Pire Année',
        'Volatilité Annualisée',
        'Drawdown Maximum',
        'Sharpe Ratio',
        'Ratio Rendement/Risque',
        'Position vs SMA 50',
        'Position vs SMA 200',
        'Tendance Actuelle',
        'Signal Technique'
    ],
    'Valeur': [
        f"{rendement_total:.2f}%",
        f"{cagr:.2f}%",
        f"{rendements_annuels.mean():.2f}%",
        f"{rendements_annuels.max():.2f}% ({rendements_annuels.idxmax()})",
        f"{rendements_annuels.min():.2f}% ({rendements_annuels.idxmin()})",
        f"{volatilite_annualisee:.2f}%",
        f"{drawdown_max:.2f}%",
        f"{sharpe_ratio:.2f}",
        f"{ratio_rendement_risque:.2f}",
        f"{position_vs_sma50:+.2f}%",
        f"{position_vs_sma200:+.2f}%",
        tendance,
        signal_technique
    ],
    'Interprétation': [
        '🟢 EXCELLENT' if rendement_total > 500 else '🟡 BON',
        '🟢 EXCELLENT' if cagr > 15 else '🟡 BON',
        '🟢 POSITIF' if rendements_annuels.mean() > 10 else '🟡 MODÉRÉ',
        '🟢 TRÈS PERFORMANT',
        '🔴 ATTENTION' if rendements_annuels.min() < -20 else '🟡 ACCEPTABLE',
        '🟡 MODÉRÉ' if volatilite_annualisee < 30 else '🔴 ÉLEVÉ',
        '🔴 IMPORTANT' if drawdown_max < -30 else '🟡 ACCEPTABLE',
        '🟢 BON' if sharpe_ratio > 0.5 else '🔴 FAIBLE',
        '🟢 EXCELLENT' if ratio_rendement_risque > 0.5 else '🟡 MODÉRÉ',
        '🟢 AU-DESSUS' if position_vs_sma50 > 0 else '🔴 EN-DESSOUS',
        '🟢 AU-DESSUS' if position_vs_sma200 > 0 else '🔴 EN-DESSOUS',
        '🟢' if 'HAUSSIÈRE' in tendance else '🔴',
        '🟢 POSITIF' if signal_technique == 'ACHAT' else '🔴 NÉGATIF'
    ]
}

kpi_df = pd.DataFrame(kpi_data)
print(kpi_df.to_string(index=False))

# ================================================================================
# ÉTAPE 6 : RECOMMANDATION FINALE
# ================================================================================
print("\n" + "="*80)
print("etape 6 : recommandation d'investissement")
print("="*80)

# Système de scoring
score_performance = min(10, (cagr / 2))
score_risque = max(0, 10 - (volatilite_annualisee / 3))
score_technique = 8 if signal_technique == "ACHAT" else 3

score_global = (score_performance * 0.4 + score_risque * 0.3 + score_technique * 0.3)

print(f"\n--- scoring global ---")
print(f"score performance (40%) : {score_performance:.1f}/10")
print(f"score risque (30%) : {score_risque:.1f}/10")
print(f"score technique (30%) : {score_technique:.1f}/10")
print(f"\nscore global : {score_global:.1f}/10")

# Recommandation finale
print("\n" + "="*80)
print("recommandation finale")
print("="*80)

if score_global >= 7:
    recommandation = "ACHAT FORT"
    confiance = "ÉLEVÉE"
    justification = f"""
Microsoft présente une performance historique exceptionnelle avec un CAGR de {cagr:.1f}%
sur 15 ans. La tendance actuelle est {tendance.lower()}, le prix se situe {'au-dessus' if position_vs_sma50 > 0 else 'en-dessous'}
des moyennes mobiles clés, et le ratio rendement/risque est favorable ({ratio_rendement_risque:.2f}).
Bien que la volatilité soit de {volatilite_annualisee:.1f}%, elle est compensée par des rendements solides.
    """
elif score_global >= 5:
    recommandation = "ACHAT MODÉRÉ"
    confiance = "MODÉRÉE"
    justification = f"""
Microsoft montre de bonnes performances historiques (CAGR {cagr:.1f}%), mais avec une volatilité
notable ({volatilite_annualisee:.1f}%). Recommandé pour investisseurs avec horizon long terme et
tolérance au risque modérée. Une approche d'investissement progressif (DCA) est conseillée.
    """
else:
    recommandation = "ATTENTE"
    confiance = "FAIBLE"
    justification = f"""
Bien que Microsoft ait de bonnes performances historiques, les indicateurs actuels
suggèrent d'attendre un meilleur point d'entrée. La volatilité est élevée ({volatilite_annualisee:.1f}%)
et le risque de correction est présent.
    """

print(f"\nrecommandation : {recommandation}")
print(f"niveau de confiance : {confiance}")
print(f"\njustification :{justification}")

print("\n--- profils d'investisseurs ---")
print("""
conservateur (risque faible) :
   - position : 10-15% du portefeuille
   - strategie : dca sur 12 mois
   - horizon : long terme (>7 ans)

equilibre (risque modere) :
   - position : 20-30% du portefeuille
   - strategie : dca sur 6 mois
   - horizon : moyen-long terme (>5 ans)

dynamique (risque eleve) :
   - position : 30-40% du portefeuille
   - strategie : achat immediat ou dca sur 3 mois
   - horizon : court-moyen terme (>3 ans)
""")

print("\n--- risques et limitations ---")
print("""
risques :
   - volatilite moderee a elevee (drawdowns possibles de 20-35%)
   - concentration sur une seule action (pas de diversification)
   - facteurs macroeconomiques non pris en compte
   - performances passees ne garantissent pas les performances futures

limitations de l'analyse :
   - analyse technique uniquement (pas d'analyse fondamentale)
   - pas de prise en compte du contexte economique actuel
   - pas d'analyse sectorielle ou concurrentielle
""")

print("\n--- plan d'action recommande ---")
print("""
1. determiner votre profil de risque (conservateur/equilibre/dynamique)
2. calculer le montant a investir (max 40% sur une seule action)
3. choisir votre strategie d'entree (immediat vs dca)
4. definir votre stop-loss : -15% pour actif, -25% pour passif
5. mettre en place un suivi regulier (mensuel/trimestriel)
6. diversifier : ne jamais tout miser sur une seule action
""")

print("\n" + "="*80)
print("analyse complete terminee")
print("="*80)

# AFFICHER LES GRAPHIQUES
plt.show()
