# Res50
## Training Process:
The model is trainied and written with tensorflow.  
To train this model,we just need to run the file cnn.py. Note that there are 4 ways to train the model in all in this file. The only difference between them is the training set applied to the model.When you are ready to train the model with a certain data augumentation method, be sure to annotate the other codes which correspond to other data augumentation methods. You can easily tell the four different methods at the bottom of the codes. The weight and tensorboard will be saved automatically.

## Testing Process:
To test this model,we can simply use the get_accuracy function of the neural network instance that we trained before. After we put the test images set and their labels into that function, the function will return the accuracy of the model's prediction.
To test the specific class of a picture, you can use the predict_image function, the parameters are image path and the model path.Notice that the image size should be exactly 32*32*3.
To use the trained model,you can use `tf.keras.models.load_model(model_path)`
