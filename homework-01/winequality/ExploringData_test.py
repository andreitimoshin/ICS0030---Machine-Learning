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
from scipy.stats import boxcox


### Check NULL values -> 0
wine = pd.read_csv('winequality.csv', sep=';', quotechar='"')
print(f"Total NULL values:\n{wine.isnull().sum()}\n")


### Create a correlation analysis of independent variables against dependent variable, 'quality'
wine_num = wine.select_dtypes(include=['float64', 'int64'])
wine_num_corr = wine_num.corr()['quality'][:-1].sort_values(ascending=False)
print("There is {} correlated values with 'quality':\n{}\n".format(len(wine_num_corr), wine_num_corr))
print("Out of which the strongest correlation is with '{}' -> {}\n".format(wine_num_corr.index[0], round(wine_num_corr.iloc[0], 2)))


### Inspect whether our 'quality' data are normally distributed and not skewed
print(f"Normaltest before log: {normaltest(wine.quality.values)}")
quality_untransformed = sns.displot(wine['quality'])
plt.show()
log_transformed = np.log(wine['quality'])
quality_transformed = sns.displot(log_transformed)
plt.show()
print("Skewness before log: %f" % wine['quality'].skew())
print(f"Normaltest after log: {normaltest(log_transformed)}")
print("Skewness after log: %f" % log_transformed.skew())
sqrt_transformed = np.sqrt(wine.quality.values)
print(f"Normaltest after sqrt: {normaltest(sqrt_transformed)}")
# boxcox_result = boxcox(wine.quality.values)
# wine['quality'] = boxcox_result[0]
# lam = boxcox_result[1]
# print(f"Normaltest after boxcox: {normaltest(wine.quality.values)}")

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

y_train_pred = lr.predict(X_train_sm) # before: 2.458960 + 0.324411 * X_train
# y_train_pred = 2.458960 + 0.324411 * X_train

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
mean_residuals = np.mean(residuals)
variance_residuals = np.var(residuals)
print(f'Mean of residuals: {mean_residuals:.4f}')
print(f'Variance of residuals: {variance_residuals:.4f}\n')

### Check for patterns in residuals
plt.scatter(X_train, residuals)
plt.title('Residuals vs Alcohol')
plt.xlabel('Alcohol')
plt.ylabel('Residuals')
plt.show()


### Evaluate the model on the test set
X_test_sm = sm.add_constant(X_test) # Add constant to test set
# y_test_pred = 1.501824 + 0.197872 * X_test # before: 2.458960 + 0.324411 * X_train
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
X = wine[['alcohol']]  # Using a DataFrame to retain the 2D shape
y = wine['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)

cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

# Define the LassoCV model, with automatic hyperparameter tuning
lasso_model = LassoCV(alphas=np.arange(0, 1, 0.01), cv=cv, n_jobs=-1)

# Fit the model
lasso_model.fit(X_train, y_train)

print(f"Best alpha chosen by LassoCV: {lasso_model.alpha_:.6f}\n")

# Predict on the training and test set
y_train_pred_lasso = lasso_model.predict(X_train)
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


### ----------------------------------------------------------------------------------------------------------------###

# y_test = [inv_boxcox(value, lam) for value in y_test]
# y_test_pred_lasso = [inv_boxcox(value, lam) for value in y_test_pred_lasso]
# plt.scatter(X_test, y_test, label='Test data')
# plt.plot(X_test, y_test_pred_lasso, color='green', label='LASSO Regression Line')
# plt.xlabel('Alcohol')
# plt.ylabel('Quality')
# plt.title('LASSO Regression on Test Data: Alcohol vs Quality')
# plt.legend()
# plt.show()

# pred_wine_quality_original_scale = [round(inv_boxcox(value, lam)) for value in pred_wine_quality_lasso] # in case the Box-Cox is used

# pred_wine['Predicted Quality (LASSO)'] = pred_wine_quality_original_scale # in case the Box-Cox is used

### Capture 'quality' before and after transformation
# quality_untransformed = sns.displot(wine['quality']).set_axis_labels("Quality", "Count")
# plt.tight_layout()
# plt.show()
# boxcox_transformed = boxcox(wine.quality.values)
# quality_transformed = sns.displot(boxcox_transformed[0]).set_axis_labels("Quality", "Count")
# plt.tight_layout()
# plt.show()

### Transforming data with Box-Cox (Log and Square root did not yield any good results)
# wine['quality'] = boxcox_transformed[0]
# lam = boxcox_transformed[1]
# print(f"Normaltest after Box-Cox: {normaltest(wine.quality.values)}") # better result