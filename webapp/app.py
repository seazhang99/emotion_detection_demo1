
from flask import Flask, jsonify, redirect, render_template, request
import numpy as np
import cv2
import io
import json
from keras.models import model_from_json, load_model
import tensorflow as tf

emotions = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Sad', 'Surprised', 'Neutral']
height = 48
width = 48
global model, graph, face_cascade


def init():
  the_model = load_model('resources/fer_model.hdf5')
  the_graph = tf.get_default_graph()
  the_face_cascade = cv2.CascadeClassifier("resources/haarcascade_frontalface_default.xml")
  return the_model, the_graph, the_face_cascade

def detect_faces(gray_image):
  faces = face_cascade.detectMultiScale(gray_image, 1.3, 5)
  if faces is None or not len(faces) > 0:
    return None, None
  print(faces)
  the_faces = []
  images = []
  for index, (x, y, w, h) in enumerate(faces):
    face = faces[index]
    the_faces.append(faces[index])
    image = gray_image[face[1]:(face[1] + face[2]), face[0]:(face[0] + face[3])]
    image = cv2.resize(image, (width, height), interpolation = cv2.INTER_CUBIC) / 255.
    image = np.reshape(image, ([-1, width, width, 1]))
    images.append(image)
  return the_faces, images

def detect_emotions(faces, images):
  if faces is None:
    return None
  results = []
  for image in images:
    with graph.as_default():
      results.append(model.predict(image))
  data = []
  for index, (x, y, w, h) in enumerate(faces):
    data_hash = {}
    data_hash['x'] = str(x)
    data_hash['y'] = str(y)
    data_hash['w'] = str(w)
    data_hash['h'] = str(h)
    result = results[index][0]
    for index, emotion in enumerate(emotions):
      data_hash[emotion] = str(result[index])
    data.append(data_hash)
  return data

app = Flask(__name__)


@app.route('/')
@app.route('/demo1')
def demo1():
    return render_template('detect_from_image.html')

@app.route('/demo2')
def demo2():
    return render_template('detect_from_webcam.html')

@app.route('/demo3')
def demo3():
    return render_template('detect_from_video.html')

@app.route('/inference_sync', methods=['POST'])
def inference_sync():
  if request.method == 'POST' and 'photo' in request.files:
    photo = request.files['photo']
    in_memory_file = io.BytesIO()
    photo.save(in_memory_file)
    data = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(data, 1)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces, images = detect_faces(gray_image)
    data = detect_emotions(faces, images)
    try:
       print(json.dumps(data))
       if data is None:
        raise AttributeError('No face detected')
       return json.dumps(data)
    except AttributeError:
       return json.dumps({'Result': 'No Detection'})

if __name__ == '__main__':
  #json_file = open('fer_model_json.json', 'r')
  #model = model_from_json(json_file.read())
  #json_file.close()
  #model.load_weights('fer_model_weights.h5')
  model, graph, face_cascade = init()
  app.run(debug = True, host='0.0.0.0', ssl_context=('/working/keys/server.crt', '/working/keys/server.key'))
