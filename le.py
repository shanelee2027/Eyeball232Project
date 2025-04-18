from PIL import Image, ImageOps
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from scipy.sparse import lil_matrix, csr_matrix
from sklearn.manifold import SpectralEmbedding
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('data/filtered_df2.csv') #loading data
image_files = list(df['Image'])
batch_features = []

for i in range(0, len(image_files), 32): #batch size is 32; need to load it 
    print(i)
    batch_files = image_files[i:i+32]
    for file in batch_files:
        img = Image.open("data/eyeball_img/" + file).convert('RGB')
        img = ImageOps.grayscale(img) 
        # Split into 3 channels
        #r, g, b = img.split()
        img_array = np.asarray(img, dtype=np.float32) / 255.0
        batch_features.append(img_array.flatten())
        #print(len(batch_features))

batch_features = np.array(batch_features)

assert batch_features.shape == (2589, 524288)

embedding = SpectralEmbedding(n_components=3) #spectral embedding
X_transformed = embedding.fit_transform(batch_features) #only dooing the first 20 features
X_transformed.shape

assert X_transformed.shape == (2589, 3)

np.savez("le_results.npz", X=X_transformed[:, 0], Y=X_transformed[:, 1] , Z=X_transformed[:, 2])