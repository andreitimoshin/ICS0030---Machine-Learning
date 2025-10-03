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


### Check NULL values -> 0
wine = pd.read_csv('winequality.csv', sep=';', quotechar='"')
print(f"Total NULL values:\n{wine.isnull().sum()}\n")


### Create a correlation analysis of independent variables against dependent variable, 'quality'
wine_num = wine.select_dtypes(include=['float64', 'int64'])
wine_num_corr = wine_num.corr()['quality'][:-1].sort_values(ascending=False)
print("There is {} correlated values with 'quality':\n{}\n".format(len(wine_num_corr), wine_num_corr))
print("Out of which the strongest correlation is with '{}' -> {}\n".format(wine_num_corr.index[0], round(wine_num_corr.iloc[0], 2)))


### Inspect whether our 'quality' data are normally distributed and not skewed
print(f"Normaltest: {normaltest(wine['quality'])}") # Data is not normally distributed => transformation is needed
print("Skewness: %f" % wine['quality'].skew())


### Analyze how 'quality' values are related with other variables using scatter plot
for i in range(0, len(wine_num.columns), 4):
    pairplot = sns.pairplot(
        data=wine_num,
        x_vars=wine_num.columns[i:i+4],
        y_vars='quality', height=4, aspect=1, kind='scatter'
    )
    for ax in pairplot.axes.flat:
        ax.set_yticks(range(3, 10))
plt.show()

### Analyze the correlation between different variables using heatmap.
plt.figure(figsize=(10, 8))
sns.heatmap(wine.corr(), cmap='YlGnBu', annot=True, fmt='.2f')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()



### Train a simple linear regression model (Ordinary Least Squares OLS)
X = wine['alcohol']
y = wine['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)
X_train_sm = sm.add_constant(X_train)
lr = sm.OLS(y_train, X_train_sm).fit()

print(f"\nLinear Regression parameters:\n{lr.params}\n")

y_train_pred = lr.predict(X_train_sm)

plt.scatter(X_train, y_train, label='Training data')
plt.plot(X_train, y_train_pred, color='red', label='Regression Line')
plt.xlabel('Alcohol')
plt.ylabel('Quality')
plt.title('Simple Linear Regression: Alcohol vs Quality')
plt.legend()
plt.show()

residuals = y_train - y_train_pred

### Plotting residuals
plt.figure()
sns.histplot(residuals, kde=True)
plt.title('Residuals Distribution')
plt.xlabel('Residuals')
plt.show()

### Check mean and variance of the residuals
mean_residuals = np.mean(residuals) # mean of residuals = 0, which is good
variance_residuals = np.var(residuals)
print(f'Mean of residuals: {mean_residuals:.4f}')
print(f'Variance of residuals: {variance_residuals:.4f}\n')


### Evaluate the model on the test set
X_test_sm = sm.add_constant(X_test) # Add constant to test set
y_test_pred = lr.predict(X_test_sm)

### Calculate RMSE and R-squared
rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
r2 = r2_score(y_test, y_test_pred)
print(f"Test RMSE: {rmse}")
print(f"Test R-squared: {r2}")


### Visualize the fit on the test data
plt.scatter(X_test, y_test, label='Test data')
plt.plot(X_test, y_test_pred, color='red', label='Regression Line')
plt.xlabel('Alcohol')
plt.ylabel('Quality')
plt.title('Simple Linear Regression on Test Data: Alcohol vs Quality')
plt.legend()
plt.show()



### Evaluate an LASSO regression model on the dataset
X = wine[['alcohol']]
y = wine['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)

cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

# Define the LassoCV model, with automatic hyperparameter tuning
lasso_model = LassoCV(alphas=np.arange(0.001, 1, 0.01), cv=cv, n_jobs=-1)

lasso_model.fit(X_train, y_train)

print(f"\nBest alpha chosen by LassoCV: {lasso_model.alpha_:.6f}\n")

# Predict on the test set
y_test_pred_lasso = lasso_model.predict(X_test)

# Calculate RMSE and R-squared for the LASSO model
rmse_lasso = np.sqrt(mean_squared_error(y_test, y_test_pred_lasso))
r2_lasso = r2_score(y_test, y_test_pred_lasso)

plt.scatter(X_test, y_test, label='Test data')
plt.plot(X_test, y_test_pred_lasso, color='green', label='LASSO Regression Line')
plt.xlabel('Alcohol')
plt.ylabel('Quality')
plt.title('LASSO Regression on Test Data: Alcohol vs Quality')
plt.legend()
plt.show()


# Comparison of both models' results
print("Comparison of OLS and LASSO Regression:")
print(f"OLS Test RMSE: {rmse}")
print(f"OLS Test R-squared: {r2}")
print(f"LASSO Test RMSE: {rmse_lasso}")
print(f"LASSO Test R-squared: {r2_lasso}")


### Use pred_wine.csv set and chosen model set (LASSO) for predicting 'quality' values
pred_wine = pd.read_csv('pred_wine.csv', sep=';')

# Extract the 'alcohol' column as the predictor
X_pred = pred_wine[['alcohol']]

# Predict the 'quality' using the trained LASSO model
pred_wine_quality_lasso = lasso_model.predict(X_pred)

# Save the predictions to a new CSV file
pred_wine['Predicted Quality (LASSO)'] = [round(value) for value in pred_wine_quality_lasso]
pred_wine.to_csv('pred_wine_result_lasso.csv', index=False)