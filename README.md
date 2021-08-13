# Using-Genetic-Algorithm-for-Optimizing-Convolution-Neural-Networks

The objective of this project is to use Genetic Algorithm to determine the most optimum hyper-parameters of a CNN model that can be used for image classification. The proposed method have been tested on a small dataset of 400 images.

**Proposed CNN Model and Hyper-Parameters**

The CNN model learns features of images through a series of convolutional layers. After extracting image features, the data is downsampled using the pooling method. The downsampled data is then processed by the Dropout layer, which reduces the risk of statistical noise and overfitting in the training data. The processed data is flattened before being passed onto the inputs of the fully connected (FC) layers of the model. Based on the input features, neurons are activated and deactivated (based on the weight of each neuron) after being processed by an activation function. The output of the last dense layer is processed by the Softmax function, which outputs a vector of probabilities for the input data. Based on the predicted output of the model, an optimization function is used to reduce losses and errors in the model and to improve the accuracy of the model. 

![image](https://user-images.githubusercontent.com/63419186/129333253-dd1af85b-f8bb-486b-836c-d6e9907c8588.png)


**The hyper-parameters of a CNN model are: ** 
1)	Number of Epochs
2)	Number of Convolutional Layers
3)	Number of Filters per Convolutional Layer
4)	Kernel Size per Convolutional Layer
5)	Activation Function
6)	Pooling Layer
7)	Number of Neurons per Feed Forward Hidden Layer (Dense Layer)
8)	Optimization Function
i)	SGD (Stochastic Gradient Descent)
ii)	 RMSProp (Root Mean Square Propagation)
iii)	Adadelta (Adaptive Delta)
iv)	Adam (Adaptive Moment Estimation)

Factors such as number of running times of the CNN model and number of epochs have been predefined so as to limit the CNN model from automatically enhancing its performance (more emphasis is placed on the hyper-parameters) through the learning process.

**Representation of Hyperparameters**

![image](https://user-images.githubusercontent.com/63419186/129332699-1f5506c0-4a43-4286-823b-c7683839f796.png)


**Proposed Genetic Algorithm**

The proposed Genetic Algorithm consists of creating an initial population with 10 chromosomes. The chromosome is encoded with the hyper-parameters (which represent the genes), as detailed in above Table. The fitness of each chromosome is then calculated based on the accuracy obtained from the CNN model (with the hyper-parameters as defined by the chromosome). The two fittest chromosomes, that is the hyper-parameters demonstrating the better accuracy, are selected as the parents and are also passed onto the next generation. The same two parents are selected for crossover and random mutation; the new offspring chromosomes are added to the new generation population. The fitness of the new generation population is generated through the CNN algorithm. The process is repeated a certain number of times until no improvement is seen in the fitness value of the new population.

**Decoding the best chromosome found by the program**
![image](https://user-images.githubusercontent.com/63419186/129333137-1142615e-ed80-4c95-b915-1f90262ddfc9.png)


