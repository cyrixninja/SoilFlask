import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from PIL import Image, ImageOps

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()

# You can also use pretrained model from Keras
# Check https://keras.io/applications/

model ='model.h5'
print('Model loaded. Check http://127.0.0.1:5000/')


def classifier(img, file):
    np.set_printoptions(suppress=True)
    model = keras.models.load_model(file)
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = img
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array
    prediction = model.predict(data)
    return prediction



@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        image = Image.open(file_path).convert('RGB')
        # Make prediction
        prediction = classifier(image, model)

        # x = x.reshape([64, 64]);
        pest_class = ['Black Soil',
        'Cinder Soil',
        'Laterite Soil',
        'Peat Soil',
        'Yellow Soil'
        ]
        a = prediction[0]
        ind=np.argmax(a)
        print('Suggestion:', pest_class[ind])
        result1=pest_class[ind]
        if result1=="Black Soil":
            result="It is Black Soil.The black soil moisture very well hence it’s excellent for growing cotton. This is also popularly known as black cotton soil. However, there are many other crops that can be grown in these soils are; Rice and sugarcane, wheat, Jowar, linseed, sunflower, cereal crops, citrus fruits, vegetables, tobacco, groundnut, any oilseed crops, and millets."
        elif result1=="Cinder Soil":
            result="It is Cinder Soil.The crops that can be grown are roses,succulents,cactus,adenium,snake plant & orchids"
        elif result1=="Laterite Soil":
            result="It is Larerite Soil.These soil are not very fertile and are used in cotton growing, rice cultivation, wheat cultivation, pulses growing, cultivation of tea, growing coffee, growing rubber, growing coconut, and growing cashews. Most of the time this soil is used to make bricks due to the presence of large amounts of iron."
        elif result1=="Peat Soil":
            result="It is Peat Soil.Crops that can be grown are potatoes, sugar beet, celery, onions, carrots, lettuce and market garden crops"
        elif result1=="Yellow Soil":
            result="It is Yellow Soil/Red Soil.Crops that can be grown are  Rice, wheat, sugarcane, maize/corn, groundnut, ragi (finger millet) and potato, oilseeds, pulses, millets, and fruits such as mango, orange, citrus, and vegetables can be grown under ideal irrigation."
        return result
    return None


if __name__ == '__main__':
    # app.run(port=5002, debug=True)

    # Serve the app with gevent
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()
    app.run()