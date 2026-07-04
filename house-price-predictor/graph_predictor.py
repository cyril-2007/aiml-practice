import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("house_data.csv")

# Prepare data
X = data[["Area"]]
y = data["Price"]

# Train model
model = LinearRegression()
model.fit(X, y)

# User input
area = float(input("Enter house area in sq.ft: "))

# Predict price
prediction = model.predict(
    pd.DataFrame([[area]], columns=["Area"])
)

print("Predicted House Price:", prediction[0], "lakhs")

# Plot actual data points
plt.scatter(X, y)

# Plot regression line
plt.plot(X, model.predict(X))

# Plot predicted point
plt.scatter([area], [prediction[0]], marker="x", s=100)

plt.title("House Price Predictor")
plt.xlabel("Area (sq.ft)")
plt.ylabel("Price (Lakhs)")

plt.show()