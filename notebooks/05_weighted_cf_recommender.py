import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

TARGET_USER = 1

# Load data
df = pd.read_csv("data/interactions.csv")

# Create user-item matrix
user_item_matrix = df.pivot_table(
    index="user_id",
    columns="track",
    values="rating"
).fillna(0)

# Compute similarity
similarity = cosine_similarity(user_item_matrix)
similarity_df = pd.DataFrame(
    similarity,
    index=user_item_matrix.index,
    columns=user_item_matrix.index
)

# Get similarity scores for target user
user_sim_scores = similarity_df.loc[TARGET_USER]

# Weighted rating calculation
weighted_scores = user_item_matrix.T.dot(user_sim_scores)
sum_of_sim = user_sim_scores.sum()

predicted_ratings = weighted_scores / sum_of_sim

# Remove songs already rated by user
already_rated = user_item_matrix.loc[TARGET_USER]
predicted_ratings = predicted_ratings[already_rated == 0]

# Top recommendations
recommendations = predicted_ratings.sort_values(ascending=False)

print("Recommended tracks for user", TARGET_USER)
print(recommendations.head(5))
