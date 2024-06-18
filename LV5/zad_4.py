import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score


df = pd.read_csv('occupancy_processed.csv')


feature_names = ['S3_Temp', 'S5_CO2']
target_name = 'Room_Occupancy_Count'


X = df[feature_names].values
y = df[target_name].values

# Podijeliti podatke na skup za učenje i skup za testiranje
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Skalirati ulazne veličine
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Izgraditi model logističke regresije
logreg = LogisticRegression()
logreg.fit(X_train_scaled, y_train)

# Evaluirati klasifikator na testnom skupu podataka
# Matrica zabune
y_pred = logreg.predict(X_test_scaled)
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