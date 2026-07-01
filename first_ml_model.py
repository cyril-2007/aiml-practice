from sklearn.linear_model import LinearRegression
import numpy as np

hours = np.array([[2], [4], [6], [8]])
marks = np.array([40, 50, 60, 80])

model = LinearRegression()
model.fit(hours, marks)

prediction = model.predict([[5]])

print("Predicted Marks:", prediction[0])