import pandas as pd
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("student_data.csv")

# Input and output
X = data[["Hours"]]
y = data["Marks"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Take input from user
hours = float(input("Enter study hours: "))

# Predict marks
predicted_marks = model.predict(pd.DataFrame([[hours]], columns=["Hours"]))

print("Predicted Marks:", predicted_marks[0])