# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 11:32:32 2021

@author: Jaya Sri Bandi
"""
import os
from PIL import Image
from numpy import asarray
import numpy as np
#import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
from flask import Flask,request,render_template
import traceback
from werkzeug.utils import secure_filename
from tensorflow.keras import backend as K
UPLOAD_FOLDER = 'static/uploads/'
global model
#print(tf.__version__)
try:
    
    app = Flask(__name__,template_folder='template')
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    K.clear_session()
    model = load_model('model/ecg_model.h5')
    #graph = tf.get_default_graph()
    def process_image(filename):
        img = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        img=img[1300:1600:,120:1900]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.bitwise_not(img)
        th2 = cv2.adaptiveThreshold(img,255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,15,-2)
        horizontal = th2
        rows,cols = horizontal.shape
        
        horizontalsize = cols //30
        horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontalsize,1))
        horizontal = cv2.erode(horizontal, horizontalStructure, (-1, -1))
        horizontal_inv = cv2.bitwise_not(horizontal)
        masked_img = cv2.bitwise_and(img, img, mask=horizontal_inv)
        masked_img_inv = cv2.bitwise_not(masked_img)
        cv2.imwrite(app.config['UPLOAD_FOLDER']+"final.jpg",masked_img_inv )
        return 
    def convert_to_array(filename):
        image=Image.open(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        data = asarray(image)
        X = np.zeros((1,300,1780))
        X[0]=data
        X = (X - X.mean())/(X.std())
        return X
    def change(x): 
        answer = np.zeros((np.shape(x)[0]))
        for i in range(np.shape(x)[0]):
            max_value = max(x[i, :])
            max_index = list(x[i, :]).index(max_value)
            answer[i] = max_index
        return answer
    @app.route("/")
    def form():
        return render_template("form.html")
    @app.route('/predict',methods=['POST','GET'])
    def predict():
        print(request.files)
        file=request.files['img_file']
        filename=secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        process_image(filename)
        processed_img=convert_to_array('final.jpg')
        #with graph.as_default():
        predictions = model.predict(processed_img)
        predictions=change(predictions)
        
        if predictions[0]==0:
            return render_template('form.html', prediction='Heart Health Analysis: '+'Abnormal!')
        elif predictions[0]==1:
            return render_template('form.html', prediction='Heart Health Analysis: '+'Normal!')
    if __name__ == "__main__":
        app.run()
except:
    traceback.print_exc()
    