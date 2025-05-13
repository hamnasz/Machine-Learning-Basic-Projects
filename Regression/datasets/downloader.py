import os
import requests

# Create the datasets directory if it doesn't exist
os.makedirs("datasets", exist_ok=True)

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
response = requests.get(url)

if response.status_code == 200:
    with open("datasets/data.csv", "wb") as f:
        f.write(response.content)
    print("Dataset downloaded successfully!")
else:
    print("Failed to download dataset.")