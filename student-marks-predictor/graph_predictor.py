import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("student_data.csv")

# Prepare data
X = data[["Hours"]]
y = data["Marks"]

# Train model
model = LinearRegression()
model.fit(X, y)

# User input
hours = float(input("Enter study hours: "))

# Predict
prediction = model.predict(pd.DataFrame([[hours]], columns=["Hours"]))

print("Predicted Marks:", prediction[0])

# Plot original points
plt.scatter(X, y)

# Plot regression line
plt.plot(X, model.predict(X))

# Predicted point
plt.scatter([hours], [prediction[0]], marker="x", s=100)

plt.title("Student Marks Predictor")
plt.xlabel("Study Hours")
plt.ylabel("Marks")

plt.show()