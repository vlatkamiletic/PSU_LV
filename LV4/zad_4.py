import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#ucitavanje očišćenih podataka
df = pd.read_csv('cars_processed.csv')
print(df.info())

#različiti grafički prikazi podataka
sns.pairplot(df, hue='fuel')

sns.relplot(data=df, x='km_driven', y='selling_price', hue='fuel')
df = df.drop(['name', 'mileage'], axis=1)

#identifikacija objektnih i numeričkih kolona
obj_cols = df.select_dtypes(include='object').columns.tolist()
num_cols = df.select_dtypes(include=np.number).columns.tolist()

#grafički prikaz kategorijskih podataka
fig = plt.figure(figsize=[15, 8])
for col in range(len(obj_cols)):
    plt.subplot(2, 2, col + 1)
    sns.countplot(x=obj_cols[col], data=df)

df.boxplot(by='fuel', column=['selling_price'], grid=False)

df['selling_price'].hist(grid=False)

# Korelaciona matrica
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, linewidths=2, cmap='coolwarm')

plt.show()

print("Zadatak 1:")
print("Broj mjerenja (automobila) u datasetu:", df.shape[0])

print("Zadatak 2:")
print(df.info())

print("Zadatak 3:")
najskuplji_automobil = df.loc[df['selling_price'].idxmax()]
najjeftiniji_automobil = df.loc[df['selling_price'].idxmin()]
print("Najskuplji automobil:\n", najskuplji_automobil)
print("\nNajjeftiniji automobil:\n", najjeftiniji_automobil)

print("Zadatak 4:")
broj_automobila_2012 = df['year'].value_counts().get(2012, 0)
print("Broj automobila proizvedenih 2012. godine:", broj_automobila_2012)

print("Zadatak 5:")
najvise_km_automobil = df.loc[df['km_driven'].idxmax()]
najmanje_km_automobil = df.loc[df['km_driven'].idxmin()]
print("Automobil s najviše kilometara:\n", najvise_km_automobil)
print("\nAutomobil s najmanje kilometara:\n", najmanje_km_automobil)

print("Zadatak 6:")
najcesce_sjedala = df['seats'].mode()[0]
print("Najčešći broj sjedala:", najcesce_sjedala)

print("Zadatak 7:")
prosjek_km_dizel = df[df['fuel'] == 'Diesel']['km_driven'].mean()
prosjek_km_benzin = df[df['fuel'] == 'Petrol']['km_driven'].mean()
print("Prosječna kilometraža za automobile s dizel motorom:", prosjek_km_dizel)
print("Prosječna kilometraža za automobile s benzinskim motorom:", prosjek_km_benzin)
