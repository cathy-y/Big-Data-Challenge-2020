import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import statsmodels.api as sm
import seaborn as sns
from scipy.stats import f_oneway
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.feature_selection import f_regression

from tourism_etal import covidfactors
from importexport_tourism import tourismcovid


domestic = pd.read_csv('domestic.csv')
domestic = domestic[domestic['domestic'] != '..'].reset_index()
domestic = domestic.drop(labels=['index'], axis=1)
domestic['domestic'] = domestic['domestic'].str.replace(',', '')
domestic['domestic'] = pd.to_numeric(domestic['domestic'])
domestic.columns = ['country', 'domestic']

allpopulation = pd.read_csv('population_total.csv')
pop2020 = allpopulation[['country', '2020']]
pop2020.columns = ['country', 'population']

dompop = domestic.merge(pop2020, how='inner')
covidfactors = covidfactors.merge(dompop, how='inner')
covidfactors = covidfactors.merge(tourismcovid, how='inner')

covidfactors['domestic_per_cap'] = (covidfactors['domestic'] / covidfactors['population']) * 100000
covidfactors['log_domestic_per_cap'] = np.log(covidfactors['domestic_per_cap'])

# plt.scatter(covidfactors['log_domestic_per_cap'], covidfactors['log_deaths_per_cap'])
# plt.show()

X = covidfactors['log_domestic_per_cap'].values.reshape(-1,1)
y = covidfactors['log_deaths_per_cap'].values.reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3)
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
# print(regressor.coef_)

# plt.scatter(X, y)
# plt.plot(X, y_pred, 'r')
# plt.xlabel('Log-domestic trips per 100,000 people')
# plt.ylabel('Log-deaths per 100,000 people')
# plt.show()

slope, intercept, r_value, p_value, std_err = stats.linregress(covidfactors['log_domestic_per_cap'],
                                                               covidfactors['log_deaths_per_cap'])
# decent r_value, low p_value
# print(slope, r_value, p_value)

# figure out what these values represent
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
# print(mae, mse)

covidfactors['total_international'] = covidfactors['tourists_per_cap'] + covidfactors['outbound_per_cap']
covidfactors['log_international'] = np.log(covidfactors['total_international'])

# X = covidfactors[['log_tourists_per_cap', 'log_income', 'log_le', 'log_me', 'log_domestic_per_cap']].values
X = covidfactors[['log_domestic_per_cap', 'log_income', 'log_le']].values
y = covidfactors['log_deaths_per_cap'].values

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

X = covidfactors['log_international'].values.reshape(-1,1)
y = covidfactors['log_deaths_per_cap'].values.reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3)
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
# print(regressor.coef_)

slope, intercept, r_value, p_value, std_err = stats.linregress(covidfactors['log_international'],
                                                               covidfactors['log_deaths_per_cap'])
# decent r_value, low p_value
# print(r_value, p_value)

# figure out what these values represent
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
# print(mae, mse)

X = covidfactors[['tourists_per_cap', 'domestic_per_cap']].values

model = KMeans(n_clusters=4, random_state=3)
model.fit(X)
labels = model.predict(X)
# print(labels)

# num_clusters = list(range(1, 9))
# inertias = []
#
# for k in num_clusters:
#     model = KMeans(n_clusters=k)
#     model.fit(X)
#     inertias.append(model.inertia_)
#
# plt.plot(num_clusters, inertias, '-o')
#
# plt.xlabel('number of clusters (k)')
# plt.ylabel('inertia')
#
# plt.show()

covidfactors['labels1'] = labels

# sns.barplot(data=covidfactors, x='labels1',y='deaths_per_cap')
# plt.xlabel('Cluster')
# plt.ylabel('Deaths per 100,000 people')
# plt.show()

label_0 = covidfactors[covidfactors['labels1'] == 0].deaths_per_cap
label_1 = covidfactors[covidfactors['labels1'] == 1].deaths_per_cap
label_2 = covidfactors[covidfactors['labels1'] == 2].deaths_per_cap
label_3 = covidfactors[covidfactors['labels1'] == 3].deaths_per_cap

fstat, pval = f_oneway(label_0, label_1, label_2, label_3)
# print(pval)

# sns.scatterplot(data=covidfactors, x='tourists_per_cap', y='domestic_per_cap', hue='labels', legend='full')
# plt.legend(title='Cluster')
# plt.xlabel('Inbound tourists per 100,000 people')
# plt.ylabel('Domestic trips per 100,000 people')
# axes = plt.gca()
# axes.set_ylim([0, 1500])
# plt.show()
#
# sns.scatterplot(data=covidfactors, x='log_tourists_per_cap', y='log_deaths_per_cap', hue='labels', legend='full')
# plt.show()
#
# sns.scatterplot(data=covidfactors, x='log_domestic_per_cap', y='log_deaths_per_cap', hue='labels', legend='full')
# plt.show