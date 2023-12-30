# Lab12 - Deep Learning: CNN & RNN

## Convolutional Neural Network
 - A convolutional neural network (CNN) is a feedforward neural network.
 - CNN excels at image processing.
 - It includes a convolutional layer, a pooling layer, and a fully connected layer.
 - CNN was proposed by Hubel and Wiesel in the 1960s in their studies in cats' cortex neurons.
 - Now, CNN has become one of the research hotspots in many scientific fields, especially in the pattern classification field.
 - The network is widely used because it can avoid complex pre-processing of images and directly input original images.

### Common Usage
 - Tissue/anatomy/lesion/ segmentation
 - Disorder classification
 - Lesion/tumour detection and classification
 - Survival/disease activity/ development prediction
 - Image construction/enhancement
 - Segmentation of anatomical structures and organs
 - Detection of abnormalities and diseases
 - Detection, segmentation and classification of nucleus

### Main Concepts of CNN
 - Local receptive field: It is generally considered that human perception of the outside world is from local to global.
 - Spatial correlations among local pixels of an image are closer than those among distant pixels.
 - Therefore, each neuron does not need to know the global image.
 - It only needs to know the local image.
 - The local information is combined at a higher level to generate global information.
 - Parameter sharing: One or more filters/kernels may be used to scan input images.
 - Parameters carried by the filters are weights. In a layer scanned by filters, each filter uses the same parameters during weighted computation.
 - Weight sharing means that when each filter scans an entire image, parameters of the filter are fixed.

### CNN
 - Image/Vision classification and object detection 
   - An image has 2D(matrix) or 3D(tensor) structure (i.e., RGB) 
   - Information is contained in a pixel, an element of a matrix (2D image) or a tensor (2D images for RGB or 2D images captured with 2 cameras). 
   - Nearby pixels (values) are highly correlated.
   - Patterns in an image can be identified by the correlations between nearby pixels.
   - Nearby pixels must be processed as a chunk.
   - Identifying patterns in an image is “translation invariant” and “size invariant”, i.e., we can identify same patterns wherever it is located and whatever its size is. 
   - Sometimes, it should also be rotation invariant.

### Convolution Sublayer
 - The basic architecture of a CNN is multi-channel convolution consisting of multiple single convolutions. The output of the previous layer (or the original image of the first layer) is used as the input of the current layer. It is then convolved with the filter in the layer and serves as the output of this layer. The convolution kernel of each layer is the weight to be learned. Similar to FCN, after the convolution is complete, the result should be biased and activated through activation functions before being input to the next layer.
 - Convolution Sublayer
   - contains a set of filters whose parameters need to be learned
   - Each filter responses to a certain pattern within a receptive fields on the input. 
   - Filter examples: three filters of size 5x5 response to different patterns (diamond, T and diagonal, respectively) 
![](/Lab12/Picture1.png)

### Activation Sublayer
 - Activation Sublayer
   - Output of convolution sublayer is then passed through an activation function.
   - ReLU or leaky ReLU are typically used

### Pooling Layer
 - Use between two successive Cons layers
 - Down sample the sublayer input
 - Summarised the data
 - Two types:
   - Max-polling (takes maximum value)
   - Average-polling (take average of dd values)

### Fully Connected Layer
 - The fully connected layer is essentially a classifier. The features extracted on the convolutional layer and pooling layer are straightened and placed at the fully connected layer to output and classify results.
 - Generally, the Softmax function is used as the activation function of the final fully connected output layer to combine all local features into global features and calculate the score of each type.

### CNN Training
 - CNN model (convolution layer) 
 - Same as NN
   - optimise weight metrics
   - apply chain rule to compute gradient w.r.t. the weight metrics
 - Differences from NN
   - 3D arrays of neurons
   - partial connection & weight sharing in conv. sublayer
   - passing gradient through pooling sublayer
 - Improving performance of CNN 
   - Apply dropout to avoid co-adaptation between channels 
   - Data normalization: adjust mean (brightness) and variance (contrast) of image to make them fall within predefined ranges 
   - Batch normalization: normalize data for each batch at each layer
   - Data augmentation: increase data set by resizing and/or rotating the original image -> size/rotation invariance 

## Recurrent Neural Network
 - The recurrent neural network (RNN) is a neural network that captures dynamic information in sequential data through periodical connections of hidden layer nodes.
 - It can classify sequential data.
 - Unlike other forward neural networks, the RNN can keep a context state and even store, learn, and express related information in context windows of any length.
 - Different from traditional neural networks, it is not limited to the space boundary, but also supports time sequences.
 - In other words, there is a side between the hidden layer of the current moment and the hidden layer of the next moment.
 - The RNN is widely used in scenarios related to sequences, such as videos consisting of image frames, audio consisting of clips, and sentences consisting of words. 
 - Recurrent Neural Networks are a very important variant of neural networks heavily used in Natural Language Processing (NLP).
 - In a general neural network, an input is processed through a number of layers and an output is produced.
 - RNNs are called recurrent because they perform the same task for every element of a sequence, with the output being depended on the previous computations.
 - Features
 - Recurrences means output fed back to input
 - Necessarily, the input is a time-series data
 - Main applications are speech recognition, language modelling (machine translation, sentence completion), where data is given as time series

## Applications of RNNs
 - In addition to NLP, RNNs are also heavily used in:
   - machine translation,
   - sentiment analysis,
   - weather forecasting, and
   - stock price prediction.

## Types of RNNs
 - There are three major types of RNNs:
   - Simple RNN
   - Long / Short Term Memory (LSTM)
   - Gated Recurrent Unit (GRU)

## The pros and cons of a simple RNN
 - Pros
   - possibility of processing input of any length
   - Model size not increasing with size of input
   - Computation takes into account historical information
   - Weights are shared across time
 - Cons
   - Computation being slow
   - Difficulty of accessing information from a long time ago
   - Cannot consider any future input for the current state

## Structure of RNNs
 - Xt is the input of the input sequence at time t.
 - St is the memory unit of the sequence at time t and caches previous information.
 - St = tanh(UXt+WSt-1)
 - Ot is the output of the hidden layer of the sequence at time t.
 - Ot = tanh(VSt)
 - Ot after through multiple hidden layers, it can get the final output of the sequence at time t.

## Backpropagation Through Time (BPTT)
 - BPTT:
   - Traditional backpropagation is the extension on the time sequence.
   - There are two sources of errors in the sequence at time of memory unit: first is from the hidden layer output error at t time sequence; the second is the error from the memory cell at the next time sequence t + 1.
   - The longer the time sequence, the more likely the loss of the last time sequence to the gradient of w in the first time sequence causes the vanishing gradient or exploding gradient problem.
   - The total gradient of weight w is the accumulation of the gradient of the weight at all time sequence.
 - Three steps of BPTT:
   - Computing the output value of each neuron through forward propagation.
   - Computing the error value of each neuron through backpropagation δj.
   - Computing the gradient of each weight.
 - Updating weights using the SGD algorithm.

## Time Sequence
 - Problem with RNN
 - Due to recurrence nature, RNN training requires back propagation through time
 - Despite that the standard RNN structure solves the problem of information memory, the information attenuates during long-term memory.
 - Information needs to be saved long time in many tasks. For example, a hint at the beginning of a speculative fiction may not be answered until the end.
 - The RNN may not be able to save information for long due to the limited memory unit capacity.
 - We expect that memory units can remember key information.
 - If T get large, the gradient may vanish or explode -> the training rule should be care fully tuned

## Vanishing Gradient
 - The partial derivative is close to zero
 - The output is not related to far previous inputs
 - Long dependency is vanishing 

## Solutions
 - Change activation functions
   - Can alleviated but not eliminate the problem
   - May affect the learning performance
 - LSTM

## RNN: LSTM
Long/Short term memory (LSTM)
 - A variant of RNN to solve the vanishing gradient and to make system memory longer.
 - Good for classify, process and make predictions based on time series data.
 - A LTSM cell is composed of 
   - input gate
   - output gate
   - forget gate
 - The gate decides whether the input could be output or not (forget)

