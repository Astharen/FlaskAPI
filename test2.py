from flask import Flask, jsonify, request
from flask_restful import Api, Resource
import requests
import base64
import numpy as np
import io
import joblib
from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers
import cv2
import matplotlib.pyplot as plt


IMG_SHAPE = (28, 28, 1)

def save_img_as_csv(file):
    image = Image.open(file).convert("L")
    # convert image to numpy array
    data = np.asarray(image)/255
    print(data.shape)
    plt.imshow(data)
    plt.show()
    # np.savetxt(file.split('.')[0] + ".csv", data, delimiter=",")



file = "example7.png"
# print(file.split('.')[0] + ".csv")
save_img_as_csv(file)

