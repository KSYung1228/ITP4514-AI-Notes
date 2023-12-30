# Lab11 - Deep Learning: Artificial Neural Networks

## AI, Machine Learning and Deep Learning

### Deep Learning
 - The deep learning architecture is a deep neural network. "Deep" in "deep learning" refers to the number of layers of the neural network. 

### Artificial Neural Networks
 - A neural network can be simply expressed as an information processing system designed to imitate the human brain structure and functions based on its source, features, and explanations.
 - Artificial neural network (neural network): Formed by artificial neurons connected to each other, the neural network extracts and simplifies the human brain's microstructure and functions.
 - It is an important approach to simulate human intelligence and reflect several basic features of human brain functions, such as concurrent information processing, learning, association, model classification, and memory. 
 - Artificial Neural Networks (ANNs) are inspired by the human brain:
   - Massively parallel, distributed system, made up of simple processing units (neurons).
   - Synaptic connection strengths among neurons are used to store the acquired knowledge.
   - Knowledge is acquired by the network from its environment through a learning process.

### Properties of ANNs
 - Learning from examples – labeled or unlabeled.
 - Adaptivity – changing the connection strengths to learn things.
 - Non-linearity – the non-linear activation functions are essential.
 - Fault tolerance – if one of the neurons or connections is damaged, the whole network still works quite well.
 - Thus, they might be better alternatives than classical solutions for problems characterized by:
   - high dimensionality, noisy, imprecise or imperfect data; and
   - a lack of a clearly stated mathematical solution or algorithm.

### Training a NN Model
 - To build a neural network model, you need to consider
   - Task for NN (Regression? Classification?)
   - Input and output dimension
   - number of hidden layers
   - number of neutrons for each layer
   - Activation functions (Sigmoid, Tanh, ReLU, Softmax)
   - Cost Function (RSS / Cross-entropy)
   - Normalization
   - Optimizer
   - Common problems of ANN

### Packages for ANN
 - You don’t need to create an ANN from scratch, instead, we will use DL framework to complete the tasks: 
   - TensorFlow (tf) – a free and open-source software library for machine learning.
   - Keras – high-level API built on tensorflow that could help programmers build DL model easier.
 - Generally, Keras could solve common DL problems and tf provides more customizable options. 
 - DL frameworks also need to work with libraries like panda, numpy for data manipulation.

## Activation Functions
 - Activation functions are important for the neural network model to learn and understand complex non-linear functions.
 - They allow introduction of non-linear features to the network.
 - Without activation functions, output signals are only simple linear functions.
 - The complexity of linear functions is limited, and the capability of learning complex function mappings from data is low.

## Normalizer
 - Regularization is an important and effective technology to reduce generalization errors in machine learning.
 - It is especially useful for deep learning models that tend to be over-fit due to a large number of parameters.
 - Researchers have proposed many effective technologies to prevent over-fitting, including:
   - Adding constraints to parameters, such as 𝐿1 and 𝐿2 norms
   - Expanding the training set, such as adding noise and transforming data
   - Dropout
   - Early stopping

## Penality Parameters
 - Many regularization methods restrict the learning capability of models by adding a penalty parameter Ω(𝜃) to the objective function 𝐽. Assume that the target function after regularization is 𝐽 ̃.

$$ 𝐽 ̃(𝜃;𝑋,𝑦)=𝐽(𝜃;𝑋,𝑦)+𝛼Ω(𝜃) $$

 - Where 𝛼𝜖[0,∞) is a hyperparameter that weights the relative contribution of the norm penalty term Ω and the standard objective function 𝐽(𝑋;𝜃). If 𝛼 is set to 0, no regularization is performed. The penalty in regularization increases with 𝛼.

## L1 Regularization
 - Add 𝐿_1 norm constraint to model parameters, that is,
$$𝐽 ̃(𝑤;𝑋,𝑦)=𝐽(𝑤;𝑋,𝑦)+𝛼‖𝑤‖_1,$$

 - If a gradient method is used to resolve the value, the parameter gradient is
$$𝛻𝐽 ̃(𝑤)=∝𝑠𝑖𝑔𝑛(𝑤)+𝛻𝐽(𝑤).$$

## L2 Regularziation
 - Add norm penalty term 𝐿_2 to prevent overfitting.
$$𝐽 ̃(𝑤;𝑋,𝑦)=𝐽(𝑤;𝑋,𝑦)+1/2 𝛼‖𝑤‖_2^2,$$

 - A parameter optimization method can be inferred using an optimization technology (such as a gradient method):
$$𝑤=(1−𝜀𝛼)𝜔−𝜀𝛻𝐽(𝑤),$$

 - where 𝜀 is the learning rate. Compared with a common gradient optimization formula, this formula multiplies the parameter by a reduction factor.

## L1 vs L2
 - The major differences between 𝐿_2 and 𝐿_1:
   - According to the preceding analysis, 𝐿_1 can generate a more sparse model than 𝐿_2. When the value of parameter 𝑤 is small, 𝐿_1 regularization can directly reduce the parameter value to 0, which can be used for feature selection.

   - From the perspective of probability, many norm constraints are equivalent to adding prior probability distribution to parameters. In 𝐿_2 regularization, the parameter value complies with the Gaussian distribution rule. In 𝐿_1 regularization, the parameter value complies with the Laplace distribution rule.

## Dropout
 - Dropout is a common and simple regularization method, which has been widely used since 2014.

 - ropout randomly discards some inputs during the training process. In this case, the parameters corresponding to the discarded inputs are not updated. 

## Early Stopping
 - A test on data of the validation set can be inserted during the training. When the data loss of the verification set increases, perform early stopping.

## Optimizer
 - Optimizers tie together the loss function and model parameters by updating the model in response to the output of the loss function.
 - In simpler terms, optimizers shape and mold your model into its most accurate possible form by futzing with the weights.
 - The loss function is the guide to the terrain, telling the optimizer when it’s moving in the right or wrong direction.
 - Purposes of the algorithm optimization include but are not limited to:
   - Accelerating algorithm convergence.
   - Preventing or jumping out of local extreme values.
   - Simplifying manual parameter setting, especially the learning rate (LR).
 - Common optimizers: common GD optimizer, momentum optimizer, Nesterov, AdaGrad, AdaDelta, RMSProp, Adam, AdaMax, and Nadam

## Keras Neural Network Models
 - The focus of the Keras library is a model.
 - The simplest model is defined in the Sequential class which is a linear stack of Layers.
 - A more useful way is to create a Sequential model and add your layers in the order of the computation you wish to perform.
 - A. Model Inputs
   - The first layer in your model must specify the shape of the input.
   - This is the number of input attributes and is defined by the input_dim argument. This argument expects an integer.
 - B. Model Layers
   - Layers of different type are a few properties in common, specifically their method of weight initialization and activation functions.

   - 1\. Weight Initialization
     - The type of initialization used for a layer is specified in the kernel_initializer and bias_initializer arguments.
     - Some common types of layer initialization include:
       - random_uniform: Weights are initialized to small uniformly random values between 0 and 0.05.
       - random_normal: Weights are initialized to small Gaussian random values (zero mean and standard deviation of 0.05).
       - zeros: All weights are set to zero values.
     - You can see a full list of initialization techniques supported on the Usage of initializations page.
   - 2\. Activation Function
     - Keras supports a range of standard neuron activation function, such as: softmax, relu, tanh and sigmoid.
     - You typically specify the type of activation function used by a layer in the activation argument, which takes a string value.
     - You can see a full list of activation functions supported by Keras on the Usage of activations page.
     - Interestingly, you can also create an Activation object and add it directly to your model after your layer to apply that activation to the output of the Layer.
   - 3\. Layer Types
     - There are a large number of core Layer types for standard neural networks.
     - Some common and useful layer types you can choose from are:
     - Dense: Fully connected layer and the most common type of layer used on multi-layer perceptron models.
     - Dropout: Apply dropout to the model, setting a fraction of inputs to zero in an effort to reduce over fitting.
     - Merge: Combine the inputs from multiple models into a single model.
 - C. Model Compilation
   - Once you have defined your model, it needs to be compiled.
   - This creates the efficient structures used by the underlying backend (Theano or TensorFlow) in order to efficiently execute your model during training.
   - You compile your model using the compile() function and it accepts three important attributes:
   - Model optimizer.
   - Loss function.
   - Metrics.
   - 1\. Model Optimizers
     - The optimizer is the search technique used to update weights in your model.
     - You can create an optimizer object and pass it to the compile function via the optimizer argument.
     - This allows you to configure the optimization procedure with it’s own arguments, such as learning rate.
   - 1\. Model Optimizers (cont.)
     - You can also use the default parameters of the optimizer by specifying the name of the optimizer to the optimizer argument. For example:
     - model.compile(optimizer='sgd’)

     - Some popular gradient descent optimizers you might like to choose from include:
     - SGD: stochastic gradient descent, with support for momentum.
     - Adam: Adaptive Moment Estimation (Adam) that also uses adaptive learning rates.
   - 2\. Model Loss Functions
     - The loss function, also called the objective function is the evaluation of the model used by the optimizer to navigate the weight space.
     - You can specify the name of the loss function to use to the compile function by the loss argument. Some common examples include:
     - mse: for mean squared error.
     - binary_crossentropy: for binary logarithmic loss (logloss).
     - categorical_crossentropy: for multi-class logarithmic loss (logloss).
   - 3\. Model Metrics
     - Metrics are evaluated by the model during training.
     - Common metric is accuracy.
     - See Metrics for more choices.
 - D. Model Training
   - The model is trained on NumPy arrays using the fit() function, for example
   - model.fit(X, y, epochs=, batch_size=)
   - Training both specifies the number of epochs to train on and the batch size.
   - Epochs (epochs) is the number of times that the model is exposed to the training dataset.
   - Batch Size (batch_size) is the number of training instances shown to the model before a weight update is performed.
 - E. Model Prediction
   - Once you have trained your model, you can use it to make predictions on test data or new data.
   - There are a number of different output types you can calculate from your trained model, each calculated using a different function call on your model object.
   - For example:
     - model.evaluate(): To calculate the loss values for input data.
     - model.predict(): To generate network output for input data.
 - E. Model Prediction (cont.)
   - model.predict_classes(): To generate class outputs for input data.
   - model.predict_proba(): To generate class probabilities for input data.
   - For example, on a classification problem you will use the predict_classes() function to make predictions for test data or new data instances.

## Basic Neural Network
 - Steps of Deep Learning Project:
   - Preprocess and load data – We need to process it before feeding to the neural network. In this step, we will also visualize data which will help us to gain insight into the data.
   - Define model – Now we need a neural network model. This means we need to specify the number of hidden layers in the neural network and their size, the input and output size.
   - Loss and optimizer – Now we need to define the loss function according to our task. We also need to specify the optimizer to use with learning rate and other hyperparameters of the optimizer.
   - Fit model – This is the training step of the neural network. Here we need to define the number of epochs for which we need to train the neural network.
 - Data processing:
   - We will use simple data of mobile price range classifier.
   - The dataset consists of 20 features and we need to predict the price range in which phone lies.
   - These ranges are divided into 4 classes.
   - The features of our dataset include:
    'battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc', 'four_g',
    'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc', 'px_height', 'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time', 'three_g', 'touch_screen', 'wifi'
 - Building Neural Network:
   - Keras is a simple tool for constructing a neural network.
   - It is a high-level framework based on tensorflow, theano or cntk backends.
   - In our dataset, the input is of 20 values and output is of 4 values.
   - So the input and output layer is of 20 and 4 dimensions respectively.
 - Building Neural Network (cont.)
   - In our neural network, we are using two hidden layers of 16 and 12 dimension.
   - Sequential specifies to Keras that we are creating model sequentially and the output of each layer we add is input to the next layer we specify.
   - model.add is used to add a layer to our neural network. We need to specify as an argument what type of layer we want. The Dense is used to specify the fully connected layer.
   - the output layer takes different activation functions and for the case of multiclass classification, it is softmax.
   - ‘relu’ (Rectified Linear Unit): Returns element-wise max(x, 0).
   - Now we specify the loss function and the optimizer.
   - It is done using compile function in Keras.
   - model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
   - Categorical_crossentropy specifies that we have multiple classes. The optimizer is Adam.
   - Metrics is used to specify the way we want to judge the performance of our neural network. Here we have specified it to accuracy.

## Training Our Model
 - Training step is simple in Keras.
 - model.fit is used to train it.
 - history = model.fit(X_train, y_train, epochs=100, batch_size=64)
 - Here we need to specify the input data -> X_train, labels -> y_train, number of epochs (iterations), and batch size.
 - It returns the history of model training.
 - History consists of model accuracy and losses after each epoch.

## Multi-Layer Perceptron Overview
 - A perceptron is a simple binary classification algorithm.
 - It helps to divide a set of input signals into two parts—“yes” and “no”.
 - A multilayer perceptron (MLP) is a perceptron that teams up with additional perceptrons, stacked in several layers, to solve complex problems.
 - The diagram on the next slide shows an MLP with three layers.
 - Each perceptron in the first layer on the left (the input layer), sends outputs to all the perceptrons in the second layer (the hidden layer), and all perceptrons in the second layer send outputs to the final layer on the right (the output layer).

## Multi-Layer Perceptron on Digit Classification
 - MNIST Dataset
   - Every MNIST data point has two parts: an image of a handwritten digit (2828) and a corresponding label (0-9).
   - We’ll call the images “X” and the labels “y”.
   - The training set and test set contain 6000 and 1000 data points respectively.



