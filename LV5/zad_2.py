import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

df = pd.read_csv('occupancy_processed.csv')
X = df[['S3_Temp', 'S5_CO2']].values
y = df['Room_Occupancy_Count'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#algoritam K najbližih susjeda
k = 5  #broj susjeda
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train_scaled, y_train)


#Matrica zabune
y_pred = knn.predict(X_test_scaled)
conf_matrix = confusion_matrix(y_test, y_pred)
print("Matrica zabune:")
print(conf_matrix)

accuracy = accuracy_score(y_test, y_pred)
print("Točnost klasifikacije:", accuracy)

precision_class_0 = precision_score(y_test, y_pred, pos_label=0)
precision_class_1 = precision_score(y_test, y_pred, pos_label=1)
recall_class_0 = recall_score(y_test, y_pred, pos_label=0)
recall_class_1 = recall_score(y_test, y_pred, pos_label=1)

print("Preciznost klase 'Slobodna':", precision_class_0)
print("Preciznost klase 'Zauzeta':", precision_class_1)
print("Odziv klase 'Slobodna':", recall_class_0)
print("Odziv klase 'Zauzeta':", recall_class_1)