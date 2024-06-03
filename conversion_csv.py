import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# On initialise l entrainement
df = pd.read_csv('train.csv')
y = df["isSold"]
X = df[["integrationType", "device_PC", "device_Phone", "device_Tablet"]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=27)


df_test = pd.read_csv('test.csv')

# A modifier suivant le dataframe
df_dum = pd.get_dummies(df_test, columns=['device'], drop_first=True, dtype=float)
X_eval = df_dum[["integrationType", "device_PC", "device_Phone", "device_Tablet"]]

# A modifier suivant le modele utilise
lr = LogisticRegression(solver='liblinear').fit(X_train, y_train)
lr_pred = lr.predict(X_eval)
bool_array = np.where(lr_pred == 1, True, False)

# Creation du csv
results = pd.DataFrame(bool_array, columns=["isSold"])
df_final = pd.concat([df_test["auctionId"], results] ,axis=1)
df_final.to_csv('test_submit.csv', index=False)