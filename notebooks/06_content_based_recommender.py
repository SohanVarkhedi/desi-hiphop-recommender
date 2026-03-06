import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder

# Load metadata
tracks = pd.read_csv("data/tracks_metadata.csv")

# Encode style feature
encoder = OneHotEncoder()
style_encoded = encoder.fit_transform(tracks[["style"]]).toarray()

# Compute similarity between tracks
similarity = cosine_similarity(style_encoded)

similarity_df = pd.DataFrame(
    similarity,
    index=tracks["track"],
    columns=tracks["track"]
)

TARGET_TRACK = "Namastute"

print("Tracks similar to:", TARGET_TRACK)
print(similarity_df[TARGET_TRACK].sort_values(ascending=False)[1:6])