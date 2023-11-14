import pandas as pd

# Your dictionary
data = {'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 35],
        'City': ['New York', 'San Francisco', 'Los Angeles']}

print(list(data.keys()))
for key in data.keys():
        print(key)
        print(type(key))

# Convert the dictionary to a Pandas DataFrame
df = pd.DataFrame(data)

# Display the DataFrame
print(df)

a = ['test'+str(i) for i in range(1, 10)]

print(a)