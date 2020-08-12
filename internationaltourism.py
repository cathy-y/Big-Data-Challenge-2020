import pandas as pd
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

totaltourism = pd.read_csv('allcountries_totaltourism.csv')
tourism2018 = totaltourism[['Country', '2018']]
tourism2018 = tourism2018[tourism2018['2018'] != '..'].reset_index()
tourism2018 = tourism2018.drop(labels=['index'], axis=1)
tourism2018.columns = ['country', 'total_tourists']
tourism2018['total_tourists'] = tourism2018['total_tourists'].str.replace(',', '')
tourism2018['total_tourists'] = pd.to_numeric(tourism2018['total_tourists'])

allcovid = pd.read_csv('covid_countriesaggregated.csv')
maycovid = allcovid[allcovid['Date'] == '2020-05-15']
maycovid = maycovid.drop(labels=['Date'], axis=1).reset_index()
maycovid = maycovid.drop(labels=['index'], axis=1)
maycovid.columns = ['country', 'confirmed', 'recovered', 'deaths']

allpopulation = pd.read_csv('population_total.csv')
pop2020 = allpopulation[['country', '2020']]
pop2020.columns = ['country', 'population']

covidpop = maycovid.merge(pop2020, how='inner')

covidpop['deaths_per_cap'] = (covidpop['deaths'] / covidpop['population']) * 100000
# covidpop['deaths_per_cap'] = covidpop['deaths_per_cap'].replace([0], 0.0001)
covidpop = covidpop[covidpop['deaths_per_cap'] != 0]
covidpop['log_deaths_per_cap'] = np.log(covidpop['deaths_per_cap'])

tourismcovid = tourism2018.merge(covidpop, how='inner')
tourismcovid['tourists_per_cap'] = (tourismcovid['total_tourists'] / tourismcovid['population']) * 100000
tourismcovid['log_tourists_per_cap'] = np.log(tourismcovid['tourists_per_cap'])

# print(tourismcovid)

# plt.scatter(x=tourismcovid['log_tourists_per_cap'], y=tourismcovid['log_deaths_per_cap'])
# plt.show()

X = tourismcovid['log_tourists_per_cap'].values.reshape(-1,1)
y = tourismcovid['log_deaths_per_cap'].values.reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
# print(regressor.coef_, regressor.intercept_)

# plt.scatter(X, y)
# plt.plot(X, y_pred, 'r')
# plt.xlabel('Log-inbound tourists per 100,000 people')
# plt.ylabel('Log-deaths per 100,000 people')
# plt.show()

slope, intercept, r_value, p_value, std_err = stats.linregress(tourismcovid['log_tourists_per_cap'],
                                                               tourismcovid['log_deaths_per_cap'])
# r_value decent, p_value very low
# print(slope, intercept, r_value, p_value)

# figure out what these values represent
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
# print(mae, mse)
