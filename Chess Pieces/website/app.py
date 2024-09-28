from flask import Flask, render_template, request, send_from_directory, url_for
from tensorflow.keras.preprocessing import image #type: ignore
from tensorflow.keras.models import load_model #type: ignore
import numpy as np
import os

app = Flask(__name__)

# Load the pre-trained model
model = load_model(r'C:\Users\admin dell\Desktop\Completed projects\Chess Pieces\chess-1.keras', compile=False)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/submit', methods=['POST'])
def submit_form():
    if request.method == 'POST':
        # Save the uploaded file
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        filepath = os.path.join(basepath, 'uploads', f.filename)
        f.save(filepath)

        img = image.load_img(filepath, target_size = (224,224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array/255
        op = ['Black bishop', 'Black king', 'Black knight', 'Black pawn', 'Black queen', 'Black rook', 'White bishop', 'White king', 'White knight', 'White pawn', 'White queen', 'White rook']
        prediction = model.predict(img_array)
        prediction_class = np.argmax(prediction, axis=1)
        prediction_class_index = prediction_class[0]
        prediction_class_label = op[prediction_class_index]
        image_url = url_for('static', filename='uploads/' + f.filename)
     
        
        
        return render_template('result.html', text = prediction_class_label, image = f.filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(os.path.join(app.root_path, 'uploads'), filename)

if __name__ == "__main__":
    if not os.path.exists('uploads'):
        os.makedirs('uploads')

    app.run(debug=True)
