import pandas as pd

# Load dataset
df = pd.read_csv("data/interactions.csv")

print("Dataset preview:")
print(df.head())

# Compute average rating for each track
track_stats = (
    df.groupby(["track", "artist"])
      .agg(avg_rating=("rating", "mean"),
           num_ratings=("rating", "count"))
      .reset_index()
)

# Sort by rating, then by number of ratings
top_tracks = track_stats.sort_values(
    by=["avg_rating", "num_ratings"],
    ascending=False
)

print("\nTop Recommended Tracks (Baseline):")
print(top_tracks.head(5))
