from flask import Flask, render_template, request, send_from_directory, url_for
from tensorflow.keras.preprocessing import image #type: ignore
from tensorflow.keras.models import load_model #type: ignore
import numpy as np
import os

app = Flask(__name__)

# Load the pre-trained model
model = load_model(r'C:\Users\admin dell\Desktop\Completed projects\Hymenoptera (Ant vs Bees)\website\hymen.h5', compile=False)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

# Renaming the function for the '/submit' route to avoid conflict
@app.route('/submit', methods=['POST'])
def submit_form():
    if request.method == 'POST':
        # Save the uploaded file
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        filepath = os.path.join(basepath, 'uploads', f.filename)
        f.save(filepath)

        # Load and preprocess the image
        img = image.load_img(filepath, target_size=(128,128))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize the image

        # Predict the class
        op = ['Ant', 'Bee']  # Adjust the labels to your categories (Ant, Bee)
        predictions = model.predict(img_array)
        prediction_class = (predictions > 0.5).astype('int8')
        prediction_class_index = int(prediction_class[0][0])
        prediction_labels = op[prediction_class_index]
        image_url = url_for('static', filename='uploads/' + f.filename)
     
        
        
        return render_template('result.html', text = prediction_labels, image = f.filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(os.path.join(app.root_path, 'uploads'), filename)

if __name__ == "__main__":
    if not os.path.exists('uploads'):
        os.makedirs('uploads')

    app.run(debug=True)
