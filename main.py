import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import manhattan_distances
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

class SpotifyRecommendation:
    def __init__(self, dataset_path):
        # Load the dataset
        self.spotify_df = pd.read_csv(dataset_path)
        self._prepare_data()

    def _prepare_data(self):
        # Drop unnecessary columns
        self.spotify_df.drop(columns=['id', 'name', 'release_date', 'year', 'artists'], inplace=True)

        # Visualize correlation matrix using heatmap
        corr_matrix = self.spotify_df.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
        plt.show()

        # Normalize the data using MinMaxScaler
        datatypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        normalization_data = self.spotify_df.select_dtypes(include=datatypes)
        scaler = MinMaxScaler()
        self.normalized_data = pd.DataFrame(scaler.fit_transform(normalization_data), columns=normalization_data.columns)

        # Use K-means clustering to assign different clusters
        kmeans = KMeans(n_clusters=10)
        self.spotify_df['features'] = kmeans.fit_predict(self.normalized_data)

    def recommend_songs(self, song_name, amount=1):
       
        song = self.spotify_df[self.spotify_df.name.str.lower() == song_name.lower()].head(1).values[0]
        rec = self.spotify_df[self.spotify_df.name.str.lower() != song_name.lower()]

        distances = manhattan_distances(rec.drop(columns=['name', 'artists', 'features']), [song[2:]])

        rec['distance'] = distances.flatten()
        rec = rec.sort_values('distance')

       
        columns = ['artists', 'name']
        return rec[columns][:amount]

def load_and_recommend():
    
    window = tk.Tk()
    window.title("Spotify Song Recommendation")
    window.geometry("800x600")

  
    dataset_path = filedialog.askopenfilename()

    recommendation_model = SpotifyRecommendation(dataset_path)


    song_name = input("Enter the song name for recommendations: ")


    recommendations = recommendation_model.recommend_songs(song_name, 10)
    print(recommendations)

    window.mainloop()

load_and_recommend()
