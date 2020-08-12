import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.feature_selection import f_regression

from internationaltourism import tourismcovid

ipp = pd.read_csv('income_per_person.csv')
ipp = ipp[['country', '2020']]
ipp.columns = ['country', 'income']
ipp['log_income'] = np.log(ipp['income'])

ley = pd.read_csv('life_expectancy_years.csv')
ley = ley[['country', '2020']]
ley.columns = ['country', 'life_expectancy']
ley['log_le'] = np.log(ley['life_expectancy'])

popden = pd.read_csv('popden.csv')
popden = popden[['country', '2020']]
popden.columns = ['country', 'pop_density']
popden['log_pd'] = np.log(popden['pop_density'])

tc2 = tourismcovid[['country', 'log_deaths_per_cap', 'deaths_per_cap', 'tourists_per_cap', 'log_tourists_per_cap']]

df = ipp.merge(ley, how='inner')
factors = df.merge(popden, how='inner')

# factors = pd.read_csv('allFeatures.csv')
# factors['i_log'] = np.log(factors['income'])
# factors['le_log'] = np.log(factors['lifeExpectancy'])
# factors['pd_log'] = np.log(factors['populationDensity'])

# print(set(tc2.columns).intersection(set(factors.columns)))
covidfactors = tc2.merge(factors, how='inner')
covidfactors = covidfactors.dropna()

# plt.scatter(covidfactors['log_me'], covidfactors['log_deaths_per_cap'])
# plt.show()

X = covidfactors[['log_tourists_per_cap', 'log_income', 'log_le']].values
y = covidfactors['log_deaths_per_cap'].values

# plt.scatter(covidfactors['log_tourists_per_cap'], covidfactors['log_deaths_per_cap'])
# plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

# print(regressor.coef_)

mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
# print(mae, mse)

F, pval = f_regression(X, y.ravel())
# print(F, pval)

mreg = sm.OLS(y, X).fit()
# print(mreg.summary())
