import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import process

class MovieRecommender:
    def __init__(self):
        # Load the data
        self.ratings = pd.read_csv("ml-25m/ratings.csv")
        self.movies = pd.read_csv("ml-25m/movies.csv")
        self.matrix, self.user_mapper, self.movie_mapper, self.user_inv_mapper, self.movie_inv_mapper = self.create_matrix(self.ratings)

    def create_matrix(self, df):
        unique_users = df['userId'].nunique()
        unique_movies = df['movieId'].nunique()

        user_mapper = dict(zip(np.unique(df["userId"]), list(range(unique_users))))
        movie_mapper = dict(zip(np.unique(df["movieId"]), list(range(unique_movies))))
        
        user_inv_mapper = dict(zip(list(range(unique_users)), np.unique(df["userId"])))
        movie_inv_mapper = dict(zip(list(range(unique_movies)), np.unique(df["movieId"])))
        
        user_index = [user_mapper[i] for i in df['userId']]
        item_index = [movie_mapper[i] for i in df['movieId']]

        matrix = csr_matrix((df["rating"], (user_index, item_index)), shape=(unique_users, unique_movies))
        
        return matrix, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper

    def movie_finder(self, title):
        all_titles = self.movies['title'].tolist()
        closest_match = process.extractOne(title, all_titles)
        matched_title = closest_match[0]  # Get the best match title
        matched_id = self.movies[self.movies['title'] == matched_title]['movieId'].iloc[0]  # Get the corresponding movie ID
        return matched_id, matched_title

    def find_similar_movies(self, movie_id, k=10, metric='cosine'):
        matrix = self.matrix.T
        movie_ind = self.movie_mapper[movie_id]
        movie_vec = matrix[movie_ind]
        if isinstance(movie_vec, (np.ndarray)):
            movie_vec = movie_vec.reshape(1,-1)
        kNN = NearestNeighbors(n_neighbors=k+1, algorithm="brute", metric=metric)
        kNN.fit(matrix)
        neighbour = kNN.kneighbors(movie_vec, return_distance=False)
        neighbour_ids = [self.movie_inv_mapper[n] for n in neighbour[0] if n != movie_ind]
        return neighbour_ids

# Example usage
if __name__ == "__main__":
    recommender = MovieRecommender()
    movie_id, movie_title = recommender.movie_finder("Toy Story")
    recommendations = recommender.find_similar_movies(movie_id)
    print(f"Because you watched {movie_title}:")
    for id in recommendations:
        print(recommender.movies[recommender.movies['movieId'] == id]['title'].iloc[0])
