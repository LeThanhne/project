
from flask import Flask, flash , render_template, request , url_for , redirect
import numpy as np
import time
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import tensorflow as tf

model = keras.models.load_model('saved_model/model_1.h5') ### LOAD MODEL LÊN CODE

classes = np.loadtxt('saved_model/labels.csv',delimiter=',',dtype= str) # LOAD TEXT CHO PHẦN ĐƯA RA DỰ ĐOÁN

app = Flask(__name__)

app.secret_key = "projectX"

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html') ## để show cái file HTML mình đã làm ##

@app.route('/', methods=['POST'])
def predict():
    allow = ['png','jpg','jpeg','gif'] 

    if 'imagefile' not in request.files:
        flash('no file part')
        return redirect(request.url)

    imagefile = request.files['imagefile'] 

    if imagefile.filename == '':    ### KHÔNG UPLOAD FILE NHƯNG BẤM PREDICT 
        flash('Please upload a image to predict !')
        return redirect(request.url)

    elif imagefile.filename.rsplit(".",1)[1].lower() not in allow:
        flash('Allowed types are png, jpg, jpeg, gif !') ### NẾU SAI ĐỊNH DẠNG ẢNH
        return redirect(request.url)                       ### UPLOAD FILE SAI ĐỊNH DẠNG ( KHÔNG PHẢI ẢNH )
        
    image_path = "./static/images/" + imagefile.filename ## SAVE ảnh từ người dùng POST lên vào thư mục static/images ##

    image_path1 = "/images/" + imagefile.filename ## TẠO ĐƯỜNG DẪN ĐỂ DISPLAY HÌNH SAU KHI PREDICT

    imagefile.save(image_path)

    image = tf.keras.preprocessing.image.load_img(image_path,target_size = (256,256)) # LOAD HÌNH VÀO BIẾN 
    img_tensor = tf.keras.preprocessing.image.img_to_array(image) # CHUYỂN HÌNH SANG ARRAY
    img_tensor = np.expand_dims(img_tensor, axis=0)

    pre = model.predict(img_tensor) ## DỰ ĐOÁN

    time.sleep(0.2)

    return render_template('index2.html',imgname = image_path1, prediction = "This is {}".format(classes[pre.argmax()]))

if __name__ == '__main__':
    app.run(port=3000,debug=True) 