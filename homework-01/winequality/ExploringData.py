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


# Load the dataset and check for any NULL values
wine = pd.read_csv('winequality.csv', sep=';', quotechar='"')
print(f"Total NULL values:\n{wine.isnull().sum()}\n")

# Correlation analysis between independent variables and the target variable 'quality'
wine_num = wine.select_dtypes(include=['float64', 'int64'])
wine_num_corr = wine_num.corr()['quality'][:-1].sort_values(ascending=False)
print("There is {} correlated values with 'quality':\n{}\n".format(len(wine_num_corr), wine_num_corr))
print("Out of which the strongest correlation is with '{}' -> {}\n".format(wine_num_corr.index[0], round(wine_num_corr.iloc[0], 2)))

# Check whether the 'quality' variable is normally distributed and evaluate skewness
print(f"Normaltest: {normaltest(wine['quality'])}") # Data is not normally distributed => transformation is needed
print("Skewness: %f" % wine['quality'].skew())

# Visualize scatter plots for each feature against 'quality'
for i in range(0, len(wine_num.columns), 3):
    pairplot = sns.pairplot(
        data=wine_num,
        x_vars=wine_num.columns[i:i+3],
        y_vars='quality', height=4, aspect=1, kind='scatter'
    )
plt.show()

# Display a heatmap of the correlation matrix to analyze the relationships between all variables
plt.figure(figsize=(10, 8))
sns.heatmap(wine.corr(), cmap='YlGnBu', annot=True, fmt='.2f')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()

# Train an OLS (Ordinary Least Squares) regression model using 'alcohol' as the predictor
X = wine['alcohol']
y = wine['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)
X_train_sm = sm.add_constant(X_train) # Add an intercept term for the OLS model
lr = sm.OLS(y_train, X_train_sm).fit() # Fit the OLS regression model

print(f"\nLinear Regression parameters:\n{lr.params}\n")

# Predict on the training set and calculate residuals
y_train_pred = lr.predict(X_train_sm)
residuals = y_train - y_train_pred

# Calculate the mean and variance of the residuals to evaluate the model's fit
mean_residuals = np.mean(residuals)
variance_residuals = np.var(residuals)
print(f'Mean of residuals for OLS: {mean_residuals:.4f}') # mean of residuals = 0, which is good
print(f'Variance of residuals for OLS: {variance_residuals:.4f}\n')

# Evaluate the OLS model on the test set
X_test_sm = sm.add_constant(X_test)
y_test_pred = lr.predict(X_test_sm)

# Calculate RMSE and R-squared metrics for model evaluation
rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
r2 = r2_score(y_test, y_test_pred)
print(f"OLS Test RMSE: {rmse}")
print(f"OLS Test R-squared: {r2}")

# Visualize the model's predictions on the test data
plt.scatter(X_test, y_test, label='Test data')
plt.plot(X_test, y_test_pred, color='red', label='Regression Line')
plt.xlabel('Alcohol')
plt.ylabel('Quality')
plt.title('Simple Linear Regression on Test Data: Alcohol vs Quality')
plt.legend()
plt.show()

# Train a LASSO regression model using 'alcohol' as the predictor
X = wine[['alcohol']]
y = wine['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)

cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

# Define and train the LASSO model with cross-validation to tune the hyperparameter alpha
lasso_model = LassoCV(alphas=np.arange(0.001, 1, 0.01), cv=cv, n_jobs=-1)

lasso_model.fit(X_train, y_train)

print(f"\nBest alpha chosen by LassoCV: {lasso_model.alpha_:.6f}\n")

# Predict on the test set using the LASSO model
y_test_pred_lasso = lasso_model.predict(X_test)
residuals = y_test - y_test_pred_lasso

# Calculate the mean and variance of the residuals for LASSO
mean_residuals = np.mean(residuals)
variance_residuals = np.var(residuals)
print(f'Mean of residuals for LASSO: {mean_residuals:.4f}') # mean of residuals = 0, which is good
print(f'Variance of residuals for LASSO: {variance_residuals:.4f}\n')

# Evaluate the LASSO model using RMSE and R-squared metrics
rmse_lasso = np.sqrt(mean_squared_error(y_test, y_test_pred_lasso))
r2_lasso = r2_score(y_test, y_test_pred_lasso)

# Visualize the LASSO model's predictions on the test data
plt.scatter(X_test, y_test, label='Test data')
plt.plot(X_test, y_test_pred_lasso, color='green', label='LASSO Regression Line')
plt.xlabel('Alcohol')
plt.ylabel('Quality')
plt.title('LASSO Regression on Test Data: Alcohol vs Quality')
plt.legend()
plt.show()

# Compare the performance of OLS and LASSO regression models
print("Comparison of OLS and LASSO Regression:")
print(f"OLS Test RMSE: {rmse}")
print(f"OLS Test R-squared: {r2}")
print(f"LASSO Test RMSE: {rmse_lasso}")
print(f"LASSO Test R-squared: {r2_lasso}")

# Predict the 'quality' variable for the new dataset (pred_wine.csv) using the LASSO model
pred_wine = pd.read_csv('pred_wine.csv', sep=';')

# Use the 'alcohol' feature for prediction
X_pred = pred_wine[['alcohol']]

# Predict the 'quality' using the trained LASSO model
pred_wine_quality_lasso = lasso_model.predict(X_pred)

# Visualize predictions on the new data
plt.scatter(X_pred, pred_wine_quality_lasso, label='Test data')
plt.plot(X_pred, pred_wine_quality_lasso, color='purple', label='LASSO Regression Line')
plt.xlabel('Alcohol')
plt.ylabel('Quality')
plt.title('LASSO Regression on pred_wine.csv data: Alcohol vs Quality')
plt.legend()
plt.show()

# Save the predictions to a new CSV file
pred_wine['Predicted Quality (LASSO)'] = [round(value) for value in pred_wine_quality_lasso]
pred_wine.to_csv('pred_wine_result_lasso.csv', index=False)