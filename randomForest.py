import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('Ratings.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Training
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=10, random_state=0)
regressor.fit(X, y)

# Prediction
predicted_rating = regressor.predict([[6.5]])
print("Predicted Rating:", predicted_rating[0])

# Visualisation
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Performance Rating Prediction (Random Forest Regression)')
plt.xlabel('Years of Experience')
plt.ylabel('Performance Rating')
plt.show()

# 5-fold cross validation to check performance

from sklearn.model_selection import cross_val_score
regressor = RandomForestRegressor(n_estimators=10, random_state=0)
scores = cross_val_score(regressor, X, y, cv=5, scoring='neg_mean_squared_error')
rmse_scores = np.sqrt(-scores)
print("Cross-Validation RMSE Scores:", rmse_scores)
