import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

url = "url for drive"
data = pd.read_csv(url)

X = data.iloc[:, :-1].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Dataset shape after preprocessing:", X_scaled.shape)
