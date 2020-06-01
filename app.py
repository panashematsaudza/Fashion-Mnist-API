#import project dependencies 
import os 
import requests
import numpy as np 
import tensorflow as tf 
import imageio
from flask import Flask,request,jsonify

#stage 2 : Load the pretained model
with open("fashion_mnist_model.json","r") as f:
    model_json = f.read()

model = tf.keras.models.model_from_json(model_json)

##
model.load_weights("fashion_mnist_model.h5")

#stage 3 :create flask api

app = Flask(__name__)

# defining the classification function 

@app.route('/',methods=['GET'])
def root():
    welcome ="Whats popping welcome to the fashion mnist api"
    return jsonify(welcome)

@app.route("/api/v1/<string:img_name>",methods =["POST"])
def classify_image(img_name):
    upload_directory = "uploads/"
    image = imageio.imread(upload_directory + img_name)

    classes = ['T-shirt/top' , 'Trouser' , 'Pullover','Dress' ,'Coat' , 'Sandal' ,'Shirt' ,'Sneaker' ,'Bag' ,'Ankle boot']

    prediction = model.predict([image.reshape(1,28*28)])

    return jsonify({"object_detected":classes[np.argmax(prediction[0])]})

#start flask server 

app.run(port=1637 ,debug=False)