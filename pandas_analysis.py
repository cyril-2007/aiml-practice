import pandas as pd

data = {
    "Name": ["Cyril", "John", "Mary"],
    "Age": [18, 19, 20],
    "Marks": [90, 85, 95]
}

df = pd.DataFrame(data)

print(df)

print("Average Marks:", df["Marks"].mean())
print("Highest Marks:", df["Marks"].max())
print("Lowest Marks:", df["Marks"].min())