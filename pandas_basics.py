import pandas as pd

data = {
    "Name": ["Cyril", "John", "Mary"],
    "Age": [18, 19, 20],
    "Marks": [90, 85, 95]
}

df = pd.DataFrame(data)

print(df)