import pandas as pd
from sklearn.model_selection import train_test_split
import yaml

df = pd.read_csv("data/raw/row.csv")
X = df.drop(columns=["calories(gm)"])
y = df["calories(gm)"]

with open("params.yaml") as f:
    params = yaml.safe_load(f)
    
test_size = params["train"]["test_size"]
random_state = params["train"]["random_state"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

X_train.to_csv("data/X_train.csv", index=False)
X_test.to_csv("data/X_test.csv", index=False)
y_train.to_csv("data/y_train.csv", index=False)
y_test.to_csv("data/y_test.csv", index=False)
