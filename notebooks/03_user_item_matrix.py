import pandas as pd

# Load dataset
df = pd.read_csv("data/interactions.csv")

# Create User-Item matrix
user_item_matrix = df.pivot_table(
    index="user_id",
    columns="track",
    values="rating"
)

print("User-Item Matrix:\n")
print(user_item_matrix.fillna(0))
