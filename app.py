from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image

app = Flask(__name__)

# Load your model
model = load_model('models/model.h5')

# Ensure the upload folder exists
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        # If user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            # Process the file and predict
            img = Image.open(file_path)
            img = img.resize((100, 100))  # Resize to the input size of the model

            # Convert image to grayscale if the model expects grayscale images
            if model.input_shape[-1] == 1:
                img = img.convert('L')  # Convert to grayscale ('L' mode)
                img_array = np.array(img)
                img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension
            else:
                img_array = np.array(img)  # RGB mode
            
            img_array = np.expand_dims(img_array, axis=0)  # Create a batch
            img_array = img_array / 255.0  # Normalize the image

            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions[0])
            accuracy = np.max(predictions[0])

            return render_template('result.html', filename=filename, predicted_class=predicted_class, accuracy=accuracy)
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename='uploads/' + filename))

if __name__ == '__main__':
    app.run(debug=True)
