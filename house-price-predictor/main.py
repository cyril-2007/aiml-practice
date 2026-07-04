import pandas as pd
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("house_data.csv")

# Input and output
X = data[["Area"]]
y = data["Price"]

# Train model
model = LinearRegression()
model.fit(X, y)

# User input
area = float(input("Enter house area in sq.ft: "))

# Predict price
predicted_price = model.predict(
    pd.DataFrame([[area]], columns=["Area"])
)

print("Predicted House Price:", predicted_price[0], "lakhs")