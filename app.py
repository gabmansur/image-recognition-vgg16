from flask import Flask, render_template, request
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input, decode_predictions, VGG16

# Import VGG16 model
model = VGG16()

# Function to preprocess image
def img_preprocess(filepath):
    # load an image from file
    image_init = load_img((filepath), target_size=(224, 224))
    # convert the image pixels to a numpy array
    image = img_to_array(image_init)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    # predict the probability across all output classes
    yhat = model.predict(image)
    # convert the probabilities to class labels
    label = decode_predictions(yhat)
    # retrieve the most likely result, e.g. highest probability
    return label

app = Flask(__name__)

@app.route('/')
def upload_f():
   return render_template('upload.html')

@app.route('/prediction', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      f.save("img.jpg")
      label = img_preprocess("img.jpg")
      pred_1 = label[0][0][1]
      pred_1 = pred_1.replace("_", " ")
      pred_2 = label[0][1][1]
      pred_2 = pred_2.replace("_", " ")
      pred_3 = label[0][2][1]
      pred_3 = pred_3.replace("_", " ")
      return render_template('prediction.html', pred_1=pred_1, pred_2=pred_2, pred_3=pred_3)

# if __name__ == '__main__':
app.run(debug = True, port=8050)