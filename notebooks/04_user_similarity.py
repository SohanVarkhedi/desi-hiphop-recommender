import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv("data/interactions.csv")

# Create user-item matrix
user_item_matrix = df.pivot_table(
    index="user_id",
    columns="track",
    values="rating"
).fillna(0)

# Compute cosine similarity
user_similarity = cosine_similarity(user_item_matrix)

# Convert to DataFrame for readability
similarity_df = pd.DataFrame(
    user_similarity,
    index=user_item_matrix.index,
    columns=user_item_matrix.index
)

print("User Similarity Matrix:\n")
print(similarity_df)
