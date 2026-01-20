import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Téléchargement des données
data = yf.download('AAPL', start='2020-01-01', end='2024-01-01')

# Aplatir les colonnes multi-index si nécessaire
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.droplevel(1)

# Calcul des SMA
data['SMA_50'] = data['Close'].rolling(window=50).mean()
data['SMA_200'] = data['Close'].rolling(window=200).mean()

# Initialisation de la colonne Signal
data['Signal'] = 0

# Détection des signaux
data.loc[(data['SMA_50'] > data['SMA_200']) & (data['SMA_50'].shift(1) <= data['SMA_200'].shift(1)), 'Signal'] = 1 
data.loc[(data['SMA_50'] < data['SMA_200']) & (data['SMA_50'].shift(1) >= data['SMA_200'].shift(1)), 'Signal'] = -1 

# RSI (Relative Strength Index)
data["RSI"] = ta.momentum.RSIIndicator(data["Close"], window=14).rsi()

# MACD (Moving Average Convergence Divergence)
macd = ta.trend.MACD(data["Close"])
data["MACD"] = macd.macd()
data["MACD_Signal"] = macd.macd_signal()

# Bollinger Bands
bollinger = ta.volatility.BollingerBands(data["Close"])
data["Bollinger_High"] = bollinger.bollinger_hband()
data["Bollinger_Low"] = bollinger.bollinger_lband()

# Affichages
print("=== SMA ===")
print(data[['Close', 'SMA_50', 'SMA_200']].tail(10))

print("\n=== SIGNAUX D'ACHAT (Golden Cross) ===")
print(data[data['Signal'] == 1][['Close', 'SMA_50', 'SMA_200']])

print("\n=== SIGNAUX DE VENTE (Death Cross) ===")
print(data[data['Signal'] == -1][['Close', 'SMA_50', 'SMA_200']])

print("\n=== INDICATEURS TECHNIQUES ===")
print(data[["Close", "RSI", "MACD", "MACD_Signal", "Bollinger_High", "Bollinger_Low"]].tail())

# Score de confiance des signaux
data["Buy_Signal"] = (data["SMA_50"] > data["SMA_200"]).astype(int) + \
 (data["RSI"] < 30).astype(int) + \
 (data["MACD"] > data["MACD_Signal"]).astype(int) + \
 (data["Close"] < data["Bollinger_Low"]).astype(int)

data["Sell_Signal"] = (data["SMA_50"] < data["SMA_200"]).astype(int) + \
 (data["RSI"] > 70).astype(int) + \
 (data["MACD"] < data["MACD_Signal"]).astype(int) + \
 (data["Close"] > data["Bollinger_High"]).astype(int)

# Ajout d'un score de confiance
data["Signal_Strength"] = data["Buy_Signal"] - data["Sell_Signal"]

# ATR (volatilité)
data["ATR"] = ta.volatility.AverageTrueRange(data["High"], data["Low"], data["Close"]).average_true_range()

# Calcul du Sharpe Ratio (rendement ajusté au risque)
risk_free_rate = 0.02 # Hypothèse de taux sans risque à 2%
returns = data["Close"].pct_change()
excess_returns = returns - risk_free_rate / 252 # Ajustement journalier
sharpe_ratio = excess_returns.mean() / excess_returns.std()

print(f"\nSharpe Ratio: {sharpe_ratio:.2f}")

# Générer un résumé des signaux
last_row = data.iloc[-1]
recommendation = "Attente ⚖️"

if last_row["Signal_Strength"] >= 3:
    recommendation = "Achat fort 🔥"
elif last_row["Signal_Strength"] >= 1:
    recommendation = "Achat modéré ✅"
elif last_row["Signal_Strength"] <= -3:
    recommendation = "Vente forte 🚨"
elif last_row["Signal_Strength"] <= -1:
    recommendation = "Vente modérée ⚠️"

# Affichage du rapport
print("\n🔹 **Synthèse des indicateurs** 🔹")
print(f"- 📊 **RSI**: {last_row['RSI']:.2f}")
print(f"- 📈 **MACD**: {last_row['MACD']:.2f}, **MACD Signal**: {last_row['MACD_Signal']:.2f}")
print(f"- 🔄 **Moyennes Mobiles**: SMA_50 = {last_row['SMA_50']:.2f}, SMA_200 = {last_row['SMA_200']:.2f}")
print(f"- 📉 **ATR (Volatilité)**: {last_row['ATR']:.2f}")
print(f"- 📊 **Sharpe Ratio**: {sharpe_ratio:.2f}")
print(f"✅ **Recommandation finale**: {recommendation}")

# ===== MACHINE LEARNING =====

# Définir la cible : on veut prédire le prix de clôture du lendemain
data["Target"] = data["Close"].shift(-1)

# Supprimer les valeurs NaN après décalage
data = data.dropna()

# Sélection des features
features = ["Open", "High", "Low", "Close", "Volume", "SMA_50", "SMA_200"]
X = data[features]
y = data["Target"]

# Normalisation des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Séparer en train/test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialisation du modèle
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Entraînement
model.fit(X_train, y_train)

# Prédiction
y_pred = model.predict(X_test)

# Évaluation du modèle
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\nMAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

# Prédiction sur les 30 prochains jours
future_days = 30
last_known_data = X.iloc[-1:].values # Dernière ligne des features connues
predictions = []

for _ in range(future_days):
    pred = model.predict(last_known_data)[0]
    predictions.append(pred)
    
    # Mise à jour des features en remplaçant les plus anciennes valeurs
    new_features = np.roll(last_known_data, -1)
    new_features[0, -1] = pred # Ajout de la nouvelle valeur prédite
    last_known_data = new_features

# Création d'une série temporelle pour les prédictions
future_dates = pd.date_range(start=data.index[-1], periods=future_days+1, freq='B')[1:]
future_df = pd.DataFrame({"Date": future_dates, "Predicted_Close": predictions})

future_price = predictions[-1]
current_price = data["Close"].iloc[-1]

if future_price > current_price:
    decision = "📈 Achat recommandé"
else:
    decision = "📉 Vente recommandée"

print(f"\nPrix actuel: {current_price:.2f} $")
print(f"Prix prédit dans {future_days} jours: {future_price:.2f} $")
print(f"📌 Recommandation : {decision}")

# ===== GRAPHIQUES =====

# Graphique 1: Prix avec moyennes mobiles
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Close'], label='Prix Close')
plt.plot(data.index, data['SMA_50'], label='SMA 50')
plt.plot(data.index, data['SMA_200'], label='SMA 200')
plt.plot(data.index, data["RSI"], label='RSI')
plt.plot(data.index, data["MACD"], label='MACD')
plt.plot(data.index, data["MACD_Signal"], label='MACD_signal')

buy_signals = data[data['Signal'] == 1]
sell_signals = data[data['Signal'] == -1]
plt.scatter(buy_signals.index, buy_signals['Close'], color='green', marker='^', s=150, label='Achat', zorder=5)
plt.scatter(sell_signals.index, sell_signals['Close'], color='red', marker='v', s=150, label='Vente', zorder=5)

plt.xlabel('Date')
plt.ylabel('Prix ($)')
plt.title('Prix Apple avec moyennes mobiles')
plt.legend() 
plt.show()
