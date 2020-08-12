import pandas as pd
import numpy as np
from internationaltourism import covidpop
import statsmodels.api as sm
from sklearn.feature_selection import f_regression
from sklearn.linear_model import LinearRegression
from scipy import stats
from matplotlib import pyplot as plt

amob = pd.read_csv('applemobilitytrends.csv')

# averaging mobility data
col = amob.loc[:, '2020-03-01':'2020-04-30']
amob['avg_mob'] = col.mean(axis=1)
amob = amob[['region', 'transportation_type', 'avg_mob']]

# expand transportation_type to be columns
amob = amob.pivot_table(values='avg_mob', index='region', columns='transportation_type').reset_index()
amob.rename(columns={'region': 'country'}, inplace=True)

amobcovid = covidpop.merge(amob, how='inner')
amobcovid = amobcovid[['country', 'deaths_per_cap', 'log_deaths_per_cap', 'driving', 'transit', 'walking']]
amobcovid = amobcovid.dropna().reset_index()
# print(amobcovid.head())

X = amobcovid[['driving', 'transit', 'walking']]
y = amobcovid['deaths_per_cap']

mreg = sm.OLS(y, X).fit()
print(mreg.summary())

F, pval = f_regression(X, y.ravel())
# print(F, pval)

amobcovid['d_t'] = amobcovid['driving'] / amobcovid['transit']
amobcovid['log_dt'] = np.log(amobcovid['d_t'])
amobcovid['d_w'] = amobcovid['driving'] / amobcovid['walking']
amobcovid['log_dw'] = np.log(amobcovid['d_w'])
amobcovid['t_w'] = amobcovid['transit'] / amobcovid['walking']
amobcovid['log_tw'] = np.log(amobcovid['t_w'])

slope, intercept, r_value, p_value, std_err = stats.linregress(amobcovid['log_tw'],
                                                               amobcovid['log_deaths_per_cap'])
print(slope, r_value, p_value)

X = amobcovid['log_tw'].values.reshape(-1,1)
y = amobcovid['log_deaths_per_cap'].values.reshape(-1,1)
regressor = LinearRegression()
regressor.fit(X, y)

y_pred = regressor.predict(X)

# plt.scatter(X, y)
# plt.plot(X, y_pred, 'r')
# plt.xlabel('Log ratio of driving/transit')
# plt.ylabel('Log-deaths per 100,000 people')
# plt.show()
