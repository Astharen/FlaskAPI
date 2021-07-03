from flask import Flask, jsonify, request, render_template
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
# to retrieve and send back data

app = Flask(__name__) # Crea el servidor
api = Api(app)



# create our model
IMG_SHAPE = (28, 28, 1)

# decode the image coming from the request
def decode_request(req):
    encoded = req["image"]
    decoded = base64.b64decode(encoded)
    return decoded

def load_model1():
    new_model = keras.models.load_model('model.h5')
    return new_model

model = load_model1()

def preprocess(decoded):
    #resize and convert in RGB in case image is in RGBA
    # pil_image = Image.open(io.BytesIO(decoded)).convert("L") 
    pil_image = cv2.imdecode(np.frombuffer(decoded, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    # pil_image = Image.open(io.BytesIO(pil_image)).convert("L") 
    # print(pil_image.shape)
    supp_pil = Image.fromarray(pil_image).convert("L") 
    image = np.asarray(supp_pil).reshape((1, IMG_SHAPE[0], IMG_SHAPE[1], 1))
    # image = np.expand_dims(pil_image, -1)
    # batch = np.expand_dims(image, axis=0)
    
    return image/255


@app.route('/')
def home():
    return render_template('app.html')
  

@app.route("/predict", methods=["POST"])
def predict():
    print("[+] request received")
    # get the data from the request and put ir under the right format
    req = request.get_json(force=True)
    image = decode_request(req)
    batch = preprocess(image)
    print(batch.shape)
    # actual prediction of the model
    predictions = model.predict(batch)[0]
    print(predictions)
    pred_list = [np.argmax(predictions), predictions[np.argmax(predictions)]]
    # get the label of the predicted class
    # top_label = [(i[1],str(i[2])) for i in decode_predictions(prediction)[0][0]]
    # create the response as a dict
    response = {"prediction": [str(pred_list[0]), str(pred_list[1])]}
    print("[+] results {}".format(response))
    
    return jsonify(response) # return it as json


if __name__ == "__main__":
	app.run(debug=True, port=4000) # Inicia el api