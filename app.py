import os
import numpy as np
import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
# Keras
from tensorflow.keras.models import load_model
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
#from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from matplotlib.pyplot import pause
from googlesearch import search
# Define a flask app
app = Flask(__name__)
# Model saved with Keras model.save()
MODEL_PATH = 'models\model_inception.h5'
# Load your trained model
model = load_model(MODEL_PATH)
#model._make_predict_function()    
# 
print()
print('Invoking model...')
print()
print('Model loaded Successful. Start serving......')
def model_predict(img_path, model):
    #update by ViPS
    # Image directory
    directory ="C:\PlantDisease_Deployment-main"
    # Change the current directory 
    # to specified directory 
    os.chdir(directory)
    img = cv2.imread(img_path)
    #img = cv2.imread("0a4b3cde-c83a-4c83-b037-010369738152___RS_Late.B 6985_flipLR.JPG")
    segmentor = SelfiSegmentation()
    img_Out = segmentor.removeBG(img, (225,225,225), threshold=0.99)
    #cv2.imshow('img',img_Out)
    #cv2.imwrite("new"+img_path,img_Out)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #os.chdir(directory)
    #img_new = cv2.imread("new"+img_path)
    new_arr = cv2.resize(img_Out,(224,224))
    new_arr = np.array(new_arr/255)
    new_arr = new_arr.reshape(-1, 224,224, 3)
    preds = model.predict(new_arr)
    os.remove(img_path)
    #cv2.imshow('img',img_Out)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return preds
@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')
@app.route('/guid', methods=['GET'])
def guid():
    # Guid page
    return render_template('guid.html')
@app.route('/team', methods=['GET'])
def team():
    # Team page
    return render_template('team.html')
@app.route('/contact', methods=['GET'])
def contact():
    # Contact page
    return render_template('contact.html')
@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(f.filename )  #secure_filename(f.filename)
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)

        # Process your result for human
        pred_class = preds.argmax()              # Simple argmax
 
        CATEGORIES = ["Bacterial Spot \n There are no cures for systemically infected plants and these plants should be discarded. " ,"Early Blight  \n hjkjhgf  ","Late Blight \n hjkjhgf","Leaf Mold \n hjkjhgf","Septoria Leaf Spot \n hjkjhgf","Spider Mites Two Spotted Spirer Mite \n hjkjhgf","Target Spot \n hjkjhgf","Yellow Leaf Curl Virus \n hjkjhgf","Mosaic Virus \n hjkjhgf","Healthy \n hjkjhgf"]
        INFO = ["SAAFDH", "WETREYT", "WRETDGA","WEFUI", "KJHG", "XCVB", "JGHF","JKKR", "FGH", "CVVV"]
        cattegories_1 = CATEGORIES[pred_class]
        info_1 = INFO[pred_class]
        CATEGORIES[pred_class]
        """
        for i in search(CATEGORIES[pred_class],tld="com", num=5, stop=5, pause=2):
        print (i)
        """
        return cattegories_1 +  info_1 
    
    return render_template("index.html")
    
    #return None
   
if __name__ == '__main__':
    app.run(debug=True)

