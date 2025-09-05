import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import joblib, json

X_test = pd.read_csv("data/X_test.csv")
y_test = pd.read_csv("data/y_test.csv")

model = joblib.load("model/model.pkl")
preds = model.predict(X_test)

# Metrics
r2 = r2_score(y_test, preds)
mse = mean_squared_error(y_test, preds)

# Save to metrics.json
metrics = {
    "r2_score": round(r2, 4),
    "mse": round(mse, 4)
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("R² Score:", r2)
print("MSE:", mse)
