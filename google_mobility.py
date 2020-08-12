import pandas as pd
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.feature_selection import f_regression

from domestictourism import covidfactors

g_mob = pd.read_csv('Global_Mobility_Report.csv', low_memory=False)
g_mob.columns = ['code', 'country', 's1', 's2', 'date', 'rec', 'groc', 'parks', 'transit', 'work', 'home']
g_mob = g_mob[pd.isnull(g_mob['s1'])].reset_index()
g_mob = g_mob.drop(labels=['index', 's1', 's2'], axis=1)
g_mob['date'] = pd.to_datetime(g_mob['date'])
start_date = '2020-03-01'
end_date = '2020-04-30'
mask = (g_mob['date'] >= start_date) & (g_mob['date'] <= end_date)
g_mob = g_mob.loc[mask].reset_index()
g_mob = g_mob.drop(labels=['index', 'code', 'date'], axis=1)
# print(g_mob.head())

agg_gmob = g_mob.groupby('country').mean()
lowest = agg_gmob.min()
agg_gmob[['rec', 'groc', 'parks', 'transit', 'work', 'home']] += abs(lowest) + 1
agg_gmob = agg_gmob.reset_index()
agg_gmob['log_rec'] = np.log(agg_gmob['rec'])
agg_gmob['log_groc'] = np.log(agg_gmob['groc'])
agg_gmob['log_parks'] = np.log(agg_gmob['parks'])
agg_gmob['log_transit'] = np.log(agg_gmob['transit'])
agg_gmob['log_work'] = np.log(agg_gmob['work'])
agg_gmob['log_home'] = np.log(agg_gmob['home'])
# print(agg_gmob.head())

covidfactors = covidfactors[['country', 'deaths_per_cap', 'log_deaths_per_cap', 'log_income', 'log_le', 'log_me']]
# print(covidfactors.head())

covid_mob = covidfactors.merge(agg_gmob, how='inner')
# print(covid_mob.head())

X = covid_mob[['log_rec', 'log_groc', 'log_parks', 'log_transit', 'log_work', 'log_home']].values
y = covid_mob['log_deaths_per_cap'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=8)
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

plt.scatter(covid_mob['log_transit'], covid_mob['log_deaths_per_cap'])
plt.show()
