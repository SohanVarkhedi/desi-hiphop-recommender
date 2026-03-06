import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder

TARGET_USER = 1

# ---------- Load datasets ----------
interactions = pd.read_csv("data/interactions.csv")
tracks = pd.read_csv("data/tracks_metadata.csv")

# ---------- USER-ITEM MATRIX ----------
user_item_matrix = interactions.pivot_table(
    index="user_id",
    columns="track",
    values="rating"
).fillna(0)

# ---------- USER SIMILARITY ----------
user_similarity = cosine_similarity(user_item_matrix)
user_similarity_df = pd.DataFrame(
    user_similarity,
    index=user_item_matrix.index,
    columns=user_item_matrix.index
)

# ---------- CF SCORE ----------
user_sim_scores = user_similarity_df.loc[TARGET_USER]

weighted_scores = user_item_matrix.T.dot(user_sim_scores)
cf_scores = weighted_scores / user_sim_scores.sum()

# ---------- CONTENT SCORE ----------
encoder = OneHotEncoder()
style_encoded = encoder.fit_transform(tracks[["style"]]).toarray()

track_similarity = cosine_similarity(style_encoded)
track_similarity_df = pd.DataFrame(
    track_similarity,
    index=tracks["track"],
    columns=tracks["track"]
)

user_tracks = interactions[interactions.user_id == TARGET_USER]["track"]

content_scores = track_similarity_df[user_tracks].mean(axis=1)

# ---------- HYBRID ----------
hybrid_scores = cf_scores.add(content_scores, fill_value=0)

already_rated = user_item_matrix.loc[TARGET_USER]
hybrid_scores = hybrid_scores[already_rated == 0]

recommendations = hybrid_scores.sort_values(ascending=False)

print("\nHybrid Recommendations for User", TARGET_USER)
print(recommendations.head(5))