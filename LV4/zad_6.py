import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, max_error

df = pd.read_csv('cars_processed.csv')
print(df.info())

# one hot coding
df_encoded = pd.get_dummies(df, columns=['fuel', 'seller_type', 'transmission'])

#definiranje ulaznih i izlaznih varijabli
X = df_encoded.drop(['selling_price', 'name'], axis=1)
y = df_encoded['selling_price']

#podjela na train i test skup
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=300)

#skaliranje ulaznih varijabli
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#izrada i treniranje modela
linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)

#predikcija na train i test skupu
y_pred_train = linear_model.predict(X_train_scaled)
y_pred_test = linear_model.predict(X_test_scaled)

#evaluacija modela
print("R2 test:", r2_score(y_test, y_pred_test))
print("RMSE test:", np.sqrt(mean_squared_error(y_test, y_pred_test)))
print("Max error test:", max_error(y_test, y_pred_test))
print("MAE test:", mean_absolute_error(y_test, y_pred_test))

fig = plt.figure(figsize=[13, 10])
ax = sns.regplot(x=y_pred_test, y=y_test, line_kws={'color': 'green'})
ax.set(xlabel='Predikcija', ylabel='Stvarna vrijednost', title='Rezultati na testnim podacima')
plt.show()
