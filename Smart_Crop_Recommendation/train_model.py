import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "Crop_recommendation.csv")

if not os.path.exists(csv_path):
    raise FileNotFoundError(
        f"Dataset not found: {csv_path}\n"
        "Please check the CSV filename and folder."
    )

data = pd.read_csv(csv_path)

print("Dataset loaded successfully!")
print(data.head())