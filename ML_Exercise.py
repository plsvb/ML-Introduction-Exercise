import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Daten einlesen
url = 'https://raw.githubusercontent.com/WHPAN0108/BHT-DataScience-S24/main/regression/data/Fish.csv'
data = pd.read_csv(url)

# Überprüfen auf fehlende Werte
print(data.isnull().sum())

# Die originale 'Species'-Spalte für die stratified split behalten
species = data['Species'].copy()

# Kategorische Variable in numerische umwandeln
data = pd.get_dummies(data, columns=['Species'], drop_first=True)

# Features und Zielvariable definieren
X = data.drop('Weight', axis=1)
y = data['Weight']

# Task 1: Zufälliger Split
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X, y, test_size=0.3, random_state=42)

# Lineare Regression Task 1
lr_1 = LinearRegression()
lr_1.fit(X_train_1, y_train_1)
y_pred_lr_1 = lr_1.predict(X_test_1)

# Random Forest Task 1
rf_1 = RandomForestRegressor(random_state=42)
rf_1.fit(X_train_1, y_train_1)
y_pred_rf_1 = rf_1.predict(X_test_1)

# Bewertung Task 1
rmse_lr_1 = np.sqrt(mean_squared_error(y_test_1, y_pred_lr_1))
r2_lr_1 = r2_score(y_test_1, y_pred_lr_1)

rmse_rf_1 = np.sqrt(mean_squared_error(y_test_1, y_pred_rf_1))
r2_rf_1 = r2_score(y_test_1, y_pred_rf_1)

# Task 2: Stratified Split
split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
for train_index, test_index in split.split(X, species):
    X_train_2 = X.iloc[train_index]
    X_test_2 = X.iloc[test_index]
    y_train_2 = y.iloc[train_index]
    y_test_2 = y.iloc[test_index]

# Lineare Regression Task 2
lr_2 = LinearRegression()
lr_2.fit(X_train_2, y_train_2)
y_pred_lr_2 = lr_2.predict(X_test_2)

# Random Forest Task 2
rf_2 = RandomForestRegressor(random_state=42)
rf_2.fit(X_train_2, y_train_2)
y_pred_rf_2 = rf_2.predict(X_test_2)

# Bewertung Task 2
rmse_lr_2 = np.sqrt(mean_squared_error(y_test_2, y_pred_lr_2))
r2_lr_2 = r2_score(y_test_2, y_pred_lr_2)

rmse_rf_2 = np.sqrt(mean_squared_error(y_test_2, y_pred_rf_2))
r2_rf_2 = r2_score(y_test_2, y_pred_rf_2)

# Ergebnisse anzeigen
print("Task 1:")
print(f"Linear Regression RMSE: {rmse_lr_1}, R²: {r2_lr_1}")
print(f"Random Forest RMSE: {rmse_rf_1}, R²: {r2_rf_1}")

print("\nTask 2:")
print(f"Linear Regression RMSE: {rmse_lr_2}, R²: {r2_lr_2}")
print(f"Random Forest RMSE: {rmse_rf_2}, R²: {r2_rf_2}")

# Visualisierung der Ergebnisse
plt.figure(figsize=(14, 12))

plt.subplot(2, 2, 1)
sns.scatterplot(x=y_test_1, y=y_pred_lr_1)
plt.plot([y_test_1.min(), y_test_1.max()], [y_test_1.min(), y_test_1.max()], 'k--', lw=3)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Linear Regression Task 1')

plt.subplot(2, 2, 2)
sns.scatterplot(x=y_test_1, y=y_pred_rf_1)
plt.plot([y_test_1.min(), y_test_1.max()], [y_test_1.min(), y_test_1.max()], 'k--', lw=3)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Random Forest Task 1')

plt.subplot(2, 2, 3)
sns.scatterplot(x=y_test_2, y=y_pred_lr_2)
plt.plot([y_test_2.min(), y_test_2.max()], [y_test_2.min(), y_test_2.max()], 'k--', lw=3)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Linear Regression Task 2')

plt.subplot(2, 2, 4)
sns.scatterplot(x=y_test_2, y=y_pred_rf_2)
plt.plot([y_test_2.min(), y_test_2.max()], [y_test_2.min(), y_test_2.max()], 'k--', lw=3)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Random Forest Task 2')

plt.tight_layout()
plt.show()
