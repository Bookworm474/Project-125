import os, ssl
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LogisticRegression
from PIL import Image

#get x and y data from labels file
x = np.load('image.npz')["arr_0"]
y = pd.read_csv("labels.csv")["labels"]

#split data for training and testing
x_train, x_test, y_train, y_test = tts(x, y,
    random_state=9,
    train_size=7500,
    test_size=2500
)

#scale x data
x_train_scaled = x_train / 255.0
y_train_scaled = y_train / 255.0

#create model
clf = LogisticRegression(solver="saga", multi_class="multinomial").fit(x_train_scaled, y_train_scaled)

#function to predict alphabet from provided image
def get_prediction(image):
    im_pil = Image.open(image)
    image_bw = im_pil.convert("L")
    image_bw_resized = image_bw.resize((28,28), Image.ANTIALIAS)
    pixel_filter = 20
    min_pixel = np.percentile(image_bw_resized, pixel_filter)
    image_bw_resized_inverted_scaled = np.clip(image_bw_resized-min_pixel, 0, 255)
    max_pixel = np.max(image_bw_resized)
    image_bw_resized_inverted_scaled = np.asarray(image_bw_resized_inverted_scaled)/max_pixel
    
    #predict value of image
    sample = np.array(image_bw_resized_inverted_scaled).reshape(1,784)
    prediction = clf.predict(sample)
    #return prediction
    return prediction[0]