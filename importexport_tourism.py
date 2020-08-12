import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway
from sklearn.cluster import KMeans
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.feature_selection import f_regression

from internationaltourism import tourismcovid
from tourism_etal import covidfactors

employees = pd.read_csv('tourism_employees.csv')
employees = employees[employees['employees'] != '..'].reset_index()
employees = employees.drop(labels=['index'], axis=1)
employees['employees'] = employees['employees'].str.replace(',', '')
employees['employees'] = pd.to_numeric(employees['employees'])

tourismcovid = tourismcovid.merge(employees, how='inner')
tourismcovid['employees_per_cap'] = (tourismcovid['employees'] / tourismcovid['population']) * 100000
tourismcovid['log_employees_per_cap'] = np.log(tourismcovid['employees_per_cap'])

hotels = pd.read_csv('hotels.csv')
hotels = hotels[hotels['hotels'] != '..'].reset_index()
hotels = hotels.drop(labels=['index'], axis=1)
hotels['hotels'] = hotels['hotels'].str.replace(',', '')
hotels['hotels'] = pd.to_numeric(hotels['hotels'])

tourismcovid = tourismcovid.merge(hotels, how='inner')
tourismcovid['hotels_per_cap'] = (tourismcovid['hotels'] / tourismcovid['population']) * 100000
tourismcovid['log_hotels_per_cap'] = np.log(tourismcovid['hotels_per_cap'])

outbound = pd.read_csv('outbound.csv')
outbound = outbound[outbound['outbound'] != '..'].reset_index()
outbound = outbound.drop(labels=['index'], axis=1)
outbound['outbound'] = outbound['outbound'].str.replace(',', '')
outbound['outbound'] = pd.to_numeric(outbound['outbound'])

tourismcovid = tourismcovid.merge(outbound, how='inner')
tourismcovid['outbound_per_cap'] = (tourismcovid['outbound'] / tourismcovid['population']) * 100000
tourismcovid['log_outbound_per_cap'] = np.log(tourismcovid['outbound_per_cap'])

balance = pd.read_csv('balance.csv')
balance = balance[balance['balance'] != '..'].reset_index()
balance = balance.drop(labels=['index'], axis=1)
balance['balance'] = balance['balance'].str.replace(',', '')
balance['balance'] = pd.to_numeric(balance['balance'])

tourismcovid = tourismcovid.merge(balance, how='inner')

tourismcovid = tourismcovid.merge(covidfactors, how='inner')
# print(tourismcovid.columns)

X = tourismcovid[['tourists_per_cap', 'outbound_per_cap']].values

model = KMeans(n_clusters=2, random_state=3)
model.fit(X)
labels = model.predict(X)
# print(labels)

num_clusters = list(range(1, 9))
inertias = []

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

tourismcovid['labels'] = labels


# sns.barplot(data=tourismcovid, x='labels',y='deaths_per_cap')
# plt.xlabel('Cluster')
# plt.ylabel('Deaths per 100,000 people')
# plt.show()

label_0 = tourismcovid[tourismcovid['labels'] == 0].deaths_per_cap
label_1 = tourismcovid[tourismcovid['labels'] == 1].deaths_per_cap
# label_2 = tourismcovid[tourismcovid['labels'] == 2].deaths_per_cap
# label_3 = tourismcovid[tourismcovid['labels'] == 3].deaths_per_cap

fstat, pval = f_oneway(label_0, label_1)
# print(pval)

# v = np.concatenate([label_0, label_1, label_2, label_3])
# labels = ['0'] * len(label_0) + ['1'] * len(label_1) + ['2'] * len(label_2) + ['3'] * len(label_3)
# tukey_results = pairwise_tukeyhsd(v, labels, 0.05)
# print(tukey_results)

# sns.scatterplot(data=tourismcovid, x='tourists_per_cap', y='outbound_per_cap', hue='labels', legend='full')
# plt.legend(title='Cluster')
# plt.xlabel('Inbound tourists per 100,000 people')
# plt.ylabel('Outbound tourists per 100,000 people')
# plt.show()

X = tourismcovid[['labels', 'log_income', 'log_le']].values
y = tourismcovid['log_deaths_per_cap'].values

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
