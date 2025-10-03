import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LassoCV
from sklearn.model_selection import RepeatedKFold
from scipy.stats.mstats import normaltest

# Load the dataset and check for NULL values
wine = pd.read_csv('winequality.csv', sep=';', quotechar='"')
print(f"Total NULL values:\n{wine.isnull().sum()}\n")

# Correlation analysis: Use absolute values and sort
wine_num = wine.select_dtypes(include=['float64', 'int64'])
wine_num_corr = wine_num.corr()['quality'][:-1].sort_values(ascending=False)
print(f"There are {len(wine_num_corr)} correlated values with 'quality':\n{wine_num_corr}\n")

# Select top 5 features
# top_features = ['alcohol', 'density', 'chlorides', 'volatile acidity', 'total sulfur dioxide']
top_features = wine_num_corr.abs().sort_values(ascending=False).head(5).index.tolist()
print(f"Selected top 5 features: {top_features}\n")

# Check normality and skewness of the target variable 'quality'
print(f"Normaltest: {normaltest(wine['quality'])}")
print(f"Skewness: {wine['quality'].skew():.4f}\n")

# Scatter plots for all variables against 'quality'
for i in range(0, len(wine_num.columns), 4):
    pairplot = sns.pairplot(
        data=wine_num,
        x_vars=wine_num.columns[i:i+4],
        y_vars='quality', height=4, aspect=1, kind='scatter'
    )
    for ax in pairplot.axes.flat:
        ax.set_yticks(range(3, 10))
plt.show()

# Correlation heatmap of the dataset
plt.figure(figsize=(10, 8))
sns.heatmap(wine.corr(), cmap='YlGnBu', annot=True, fmt='.2f')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()

# Train the OLS regression model using the selected features
X = wine[top_features]
y = wine['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)
X_train_sm = sm.add_constant(X_train)
lr = sm.OLS(y_train, X_train_sm).fit()

# Predictions and residuals on the training set
y_train_pred = lr.predict(X_train_sm)
residuals = y_train - y_train_pred

# Plot residuals distribution for OLS
plt.figure()
sns.histplot(residuals, kde=True)
plt.title('OLS Residuals Distribution')
plt.xlabel('Residuals')
plt.show()

# Check residuals' mean and variance
mean_residuals = np.mean(residuals)
variance_residuals = np.var(residuals)
print(f'Mean of residuals: {mean_residuals:.4f}')
print(f'Variance of residuals: {variance_residuals:.4f}\n')

# Evaluate OLS model on the test set
X_test_sm = sm.add_constant(X_test)
y_test_pred = lr.predict(X_test_sm)

# Calculate RMSE and R-squared for OLS
rmse_ols = np.sqrt(mean_squared_error(y_test, y_test_pred))
r2_ols = r2_score(y_test, y_test_pred)
print(f"OLS Test RMSE: {rmse_ols}")
print(f"OLS Test R-squared: {r2_ols}")


# Train a LASSO regression model with the selected features
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
lasso_model = LassoCV(alphas=np.arange(0.001, 1, 0.01), cv=cv, n_jobs=-1)
lasso_model.fit(X_train, y_train)

print(f"\nBest alpha chosen by LassoCV: {lasso_model.alpha_:.6f}\n")

# Predict on the test set using LASSO
y_test_pred_lasso = lasso_model.predict(X_test)

# Calculate RMSE and R-squared for LASSO
rmse_lasso = np.sqrt(mean_squared_error(y_test, y_test_pred_lasso))
r2_lasso = r2_score(y_test, y_test_pred_lasso)
print(f"LASSO Test RMSE: {rmse_lasso}")
print(f"LASSO Test R-squared: {r2_lasso}\n")

# Comparison of OLS and LASSO results
print("Comparison of OLS and LASSO Regression:")
print(f"OLS Test RMSE: {rmse_ols}")
print(f"OLS Test R-squared: {r2_ols}")
print(f"LASSO Test RMSE: {rmse_lasso}")
print(f"LASSO Test R-squared: {r2_lasso}")

# Predict 'quality' for new data using LASSO model
pred_wine = pd.read_csv('pred_wine.csv', sep=';')
X_pred = pred_wine[top_features]

pred_wine_quality_lasso = lasso_model.predict(X_pred)

# Save predictions to a new CSV file
pred_wine['Predicted Quality (LASSO)'] = pred_wine_quality_lasso # [value for value in pred_wine_quality_lasso]
pred_wine.to_csv('pred_wine_quality_top_features.csv', index=False)