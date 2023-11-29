import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import manhattan_distances
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from scipy.cluster.vq import vq
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

class SpotifyRecommendation:
    def __init__(self, dataset_path):
        self.spotify_df = pd.read_csv(dataset_path)
        self._prepare_data()

    def _prepare_data(self):
        # Drop unnecessary columns
        self.spotify_df.drop(columns=['id', 'name', 'release_date', 'year', 'artists'], inplace=True)

      
        corr_matrix = self.spotify_df.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
        plt.show()

        
        datatypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        normalization_data = self.spotify_df.select_dtypes(include=datatypes)
        scaler = MinMaxScaler()
        self.normalized_data = pd.DataFrame(scaler.fit_transform(normalization_data), columns=normalization_data.columns)


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

def load_image():
    file_path = filedialog.askopenfilename()
    return Image.open(file_path)

def display_image(image):
    img = ImageTk.PhotoImage(image)
    panel = tk.Label(window, image=img)
    panel.image = img
    panel.pack()

window = tk.Tk()
window.title("Spotify Song Recommendation System")
window.geometry("800x600")


load_data_button = tk.Button(window, text="Load Spotify Dataset", command=lambda: recommendation_model._prepare_data())
load_data_button.pack()


recommend_button = tk.Button(window, text="Recommend Songs", command=lambda: display_image(recommendation_model.recommend_songs("Mixe", 10)))
recommend_button.pack()


window.mainloop()
