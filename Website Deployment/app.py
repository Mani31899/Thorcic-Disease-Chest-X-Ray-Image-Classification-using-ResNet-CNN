from flask import Flask, request, render_template, redirect, url_for
import os
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import uuid

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = tf.keras.models.load_model('C:/Users/manik/OneDrive/Desktop\Final year project/my_model')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename != '':
            # Save the file
            filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Process the file
            img = Image.open(filepath).convert('RGB')
            img = img.resize((96, 96))  # Adjust according to your model's expected input size
            img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
            prediction = model.predict(img_array)
            predicted_class = np.argmax(prediction, axis=1)
            result = "Disease" if predicted_class[0] == 1 else "No Disease"
            
            # Cleanup if needed
            # os.remove(filepath)  # Uncomment if you want to remove the file after prediction

            return render_template('result.html', result=result, user_image=filename)
        else:
            result = "No file selected"
            return render_template('index.html', result=result)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)


