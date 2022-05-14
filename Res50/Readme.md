# Res50
## Training Process:
The model is trainied and written with tensorflow.  
To train this model,we just need to run the file cnn.py. Note that there are 4 ways to train the model in all in this file. The only difference between them is the training set applied to the model.When you are ready to train the model with a certain data augumentation method, be sure to annotate the other codes which correspond to other data augumentation methods. You can easily tell the four different methods at the bottom of the codes. The weight and tensorboard will be saved automatically.

## Testing Process:
To test this model,we can simply use the get_accuracy function of the neural network instance that we trained before. After we put the test images and their labels into that function, the function will return the accuracy of the model's prediction.

## Save and load:
To save this model, we can use the save_model function of the Neural Network class.This function will save four npy files under the same directory as the python file.
The four files indicate the parameter matrix of the first layer, the parameter matrix of the second layer, the bias of the first layer and the bias of the second layer.

To load this model, we first need to initialize the NeuralNetwork class to create a two layer network.Then,we can first use the load_np function of the class to get trained parameters.This function will return the parameter matrix of the first layer, the parameter matrix of the second layer, the bias of the first layer and the bias of the second layer.Note that in order to run this function correctly, we need to put the four npy files under the same directory as the python file.And the four npy files should be named as "w1.npy", "w2.npy","b1.npy","b2.npy" corresponded to the parameters it stored.After that, we can use the load model function to assign the parameters to the model.In that way, the network will get trained parameters.
