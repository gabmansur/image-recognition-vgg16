# Image Recognition using VGG-16 Web App (Flask)
_A small web application using Flask where I use the pre-trained VGG16 weights to predict the class/labels of an uploaded image._

## Why?
I love creating interactive environments where people can have a more personal experience. In this project, my idea was to let the user upload his own image and display the top three most probable classes. 

## Walk-through

After importing the necessary packages, I import the VGG-16 model. [Here](https://machinelearningmastery.com/use-pre-trained-vgg-model-classify-objects-photographs/) you can read more about the VGG models and ImageNet. I will not get into the details of its structure, but it's important to know that the model inputs an image of size 224x224 and outputs a softmax layer of size (1000,1). That means there are 1000 possible labels this model can predict - ice cream, hat, and beer, are some of them. By the way, this code is based on the [Machine Learning Mastery](https://machinelearningmastery.com/use-pre-trained-vgg-model-classify-objects-photographs/) site. This is one of my favourite sites btw for tutorials.


<img src="https://media.geeksforgeeks.org/wp-content/uploads/20200219152207/new41.jpg" width="800">

VGG-16 Luckily, Keras provides an Applications interface for loading and using pre-trained models, so I only have to load it.
```
# Import VGG16 model
model = VGG16()
```

Next, I make a function to load and pre-process the image the user will upload. Let's walk through it.

```
def img_preprocess(filepath):
    #1 load an image from file
    image_init = load_img((filepath), target_size=(224, 224))
    #2 convert the image pixels to a numpy array
    image = img_to_array(image_init)
    #3 reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    #4 prepare the image for the VGG model
    image = preprocess_input(image)
    #5 predict the probability across all output classes
    yhat = model.predict(image)
    #6 convert the probabilities to class labels
    label = decode_predictions(yhat)
    return label
```

1. The image is loaded already in the targeted size - remember the VGG-16 input size is 224x224.

2. We need to convert the pixels in the images to a numpy array. Since its a RGB (color) image, with 3 channels, we will end up with an array of shape 224x224x3.

3. The network expects an input of shape (samples, rows, columns, channels). Since we are only working with a single image, we need to reshape the array to (1,224,224,3).

4. We do the preprocessing of the image. I honestly don't really understanding the details of it. I found on stackoverflow that "The preprocess_input function is meant to adequate your image to the format the model requires." Okay, I can live with that.

5. Now we can load the image into our model and make predictions. Yay! Remember that the softmax layer has a size of (1000,1)? The values correspond to the probabilities between 0 and 1 of our uploaded image of being each of the labels. Check the monstruosity below:

<img src="/images/yhat.PNG" width="500">

By the way, for this example, I am using Suzie as a model. Because look how cute she is. So basically our friend yhat is telling us Suzie has a 5.09449860e-09 probability of looking like a tree or something like that. 

<img src="/images/susu.jpeg" width="400">

6. Now let's decode those predictions and make some sense out of it. This Keras functions gets the top 5 highest probabilities and give us the corresponding labels.

![](/images/label.PNG)

Great! It guessed right (there's another Suzie picture where it guesses she's a Siamese Cat. Curious to say the least). It's interesting how it can even distinguish dog breeds, right? The function finally returns the label.

### The Flask app

Now we get to the fun. Let's make all this into an app. The [Flask's Documentation website](https://flask.palletsprojects.com/en/1.1.x/patterns/fileuploads/?highlight=upload) has great resources and tutorials, simple and easy to understand.

So, pretty simple below. Once you get used to Flask, you will see this code is very cookbookish. I'm simply requesting a file, which I save as "img.jpg"; and making it run through that preprocessing function we made (not to be confused with the Keras preprocess_input). Then I pick the top three predictions and throw it in my "prediction.html".
```
app = Flask(__name__)

@app.route('/')
def upload_f():
   return render_template('upload.html')

@app.route('/prediction', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      f.save("img.jpg")
      pred_1 = label[0][0][1]
      pred_1 = pred_1.replace("_", " ")
      pred_2 = label[0][1][1]
      pred_2 = pred_2.replace("_", " ")
      pred_3 = label[0][2][1]
      pred_3 = pred_3.replace("_", " ")
      return render_template('prediction.html', pred_1=pred_1, pred_2=pred_2, pred_3=pred_3)

# if __name__ == '__main__':
app.run(debug = True, port=8050)
```

## Next steps

There's a lot that can be developed from this. Some ideas are:

* Test other CNN architectures and compare, like ResNet50. Would be nice to show the predictions of each one side by side and see what each one predicts!
* Implement some failproofing around. For instance, limit the extension of the uploaded file to jpg, jpeg, png, and such.
* Develop the visual of the WebApp by working on the HTML and CSS. This is something I've been really into lately. I find [this website](https://www.w3schools.com/css/default.asp) has very good resources. 

## Suggestions?

I'd love to hear your feedback. Feel free to reach out to correct me, add something, and exchange ideas!
