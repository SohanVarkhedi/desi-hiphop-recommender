import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

st.title("DHH Recommender 🎧")

TARGET_USER = st.number_input("Enter User ID", min_value=1, step=1)

df = pd.read_csv("data/interactions.csv")

# User-item matrix
user_item_matrix = df.pivot_table(
    index="user_id",
    columns="track",
    values="rating"
).fillna(0)

similarity = cosine_similarity(user_item_matrix)
similarity_df = pd.DataFrame(
    similarity,
    index=user_item_matrix.index,
    columns=user_item_matrix.index
)

if TARGET_USER in user_item_matrix.index:

    user_sim_scores = similarity_df.loc[TARGET_USER]

    weighted_scores = user_item_matrix.T.dot(user_sim_scores)
    sum_of_sim = user_sim_scores.sum()

    predicted_ratings = weighted_scores / sum_of_sim

    already_rated = user_item_matrix.loc[TARGET_USER]
    predicted_ratings = predicted_ratings[already_rated == 0]

    st.subheader("Recommended Tracks")
    st.write(predicted_ratings.sort_values(ascending=False).head(5))

else:
    st.warning("User not found")
