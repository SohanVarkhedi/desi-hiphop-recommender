import pandas as pd

# Load dataset
df = pd.read_csv("data/interactions.csv")

TARGET_USER = 1

# Songs liked by target user
user_songs = df[df["user_id"] == TARGET_USER]["track"].tolist()

print("Songs liked by user:", user_songs)

# Find users with similar taste
similar_users = df[
    (df["track"].isin(user_songs)) &
    (df["user_id"] != TARGET_USER)
]["user_id"].unique()

print("Similar users:", similar_users)

# Songs liked by similar users but not target user
recommendations = df[
    (df["user_id"].isin(similar_users)) &
    (~df["track"].isin(user_songs))
]

print("\nRecommended songs:")
print(recommendations[["track", "artist"]].drop_duplicates())
