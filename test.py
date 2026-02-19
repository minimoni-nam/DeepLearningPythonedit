"""
    Testing code for different neural network configurations.
    Adapted for Python 3.5.2

    Usage in shell:
        python3.5 test.py

    Network (network.py and network2.py) parameters:
        2nd param is epochs count
        3rd param is batch size
        4th param is learning rate (eta)

    Author:
        Michał Dobrzański, 2016
        dobrzanski.michal.daniel@gmail.com
"""
#Entrené en test en lugar de network porque tenía entendido que la segunda solo define la red
#creo que sí se puede directamente con network, pero por el tiempo de entrega probaré después  
# ----------------------
# - read the input data:
#import network carga el rchivo donde está definida la red
#import mnist loader normaliza los datos y separa el entrenamiento de la validación y de la prueba
'''
import network
import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)
'''
# ---------------------
# - network.py example:
#import network

'''
net = network.Network([784, 20, 10])
net.SGD(training_data,
epochs=20, mini_batch_size=10, eta=3.0, test_data=test_data)
'''

# ----------------------
# - network2.py example:
#import network2

'''
net = network2.Network([784, 20, 10], cost=network2.CrossEntropyCost)
#net.large_weight_initializer()
net.SGD(training_data, 20, 10, 0.1, lmbda = 5.0,evaluation_data=validation_data,
    monitor_evaluation_accuracy=True)
'''

# chapter 3 - Overfitting example - too many epochs of learning applied on small (1k samples) amount od data.
# Overfitting is treating noise as a signal.
'''
net = network2.Network([784, 20, 10], cost=network2.CrossEntropyCost)
net.large_weight_initializer()
net.SGD(training_data[:1000], 400, 10, 0.5, evaluation_data=test_data,
    monitor_evaluation_accuracy=True,
    monitor_training_cost=True)
'''

# chapter 3 - Regularization (weight decay) example 1 (only 1000 of training data and 30 hidden neurons)
'''
net = network2.Network([784, 20, 10], cost=network2.CrossEntropyCost)
net.large_weight_initializer()
net.SGD(training_data[:1000], 400, 10, 0.5,
    evaluation_data=test_data,
    lmbda = 0.1, # this is a regularization parameter
    monitor_evaluation_cost=True,
    monitor_evaluation_accuracy=True,
    monitor_training_cost=True,
    monitor_training_accuracy=True)
'''

# chapter 3 - Early stopping implemented
'''
net = network2.Network([784, 20, 10], cost=network2.CrossEntropyCost)
net.SGD(training_data[:1000], 20, 10, 0.5,
    lmbda=5.0,
    evaluation_data=validation_data,
    monitor_evaluation_accuracy=True,
    monitor_training_cost=True,
    early_stopping_n=10)
'''

# chapter 4 - The vanishing gradient problem - deep networks are hard to train with simple SGD algorithm
# this network learns much slower than a shallow one.
'''
net = network2.Network([784, 20, 20, 20, 20, 10], cost=network2.CrossEntropyCost)
net.SGD(training_data, 20, 10, 0.1,
    lmbda=5.0,
    evaluation_data=validation_data,
    monitor_evaluation_accuracy=True)
'''


# ----------------------
# imortnetwork es para determinar dónde está definida la red, como la arquitectura y el back propagation
import network
#import mnist_loader normaliza los datos y separa el entreamiento de la validación y la prueba
import mnist_loader
 

# load_data_wrapper nos devuelve los tres conjuntos que tenemos a la izquierda:
# training data es de donde la red aprende
# validation data  ajusta y test data evalúa
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
# mini-batch size:
mini_batch_size = 10

# chapter 6 - shallow architecture using just a single hidden layer, containing 100 hidden neurons.
'''
net = Network([
    FullyConnectedLayer(n_in=784, n_out=100),
    SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
net.SGD(training_data, 60, mini_batch_size, 0.1, validation_data, test_data)
'''

# chapter 6 - 5x5 local receptive fields, 20 feature maps, max-pooling layer 2x2
'''
net = Network([
    ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                  filter_shape=(20, 1, 5, 5),
                  poolsize=(2, 2)),
    FullyConnectedLayer(n_in=20*12*12, n_out=100),
    SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
net.SGD(training_data, 60, mini_batch_size, 0.1, validation_data, test_data)
'''

# chapter 6 - inserting a second convolutional-pooling layer to the previous example => better accuracy
'''
net = network.Network([784, 20, 10])
'''

# chapter 6 -  rectified linear units and some l2 regularization (lmbda=0.1) => even better accuracy
import network
import mnist_loader

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net = network.Network([784, 20, 10])

net.SGD(
    training_data,
    epochs=20,
    mini_batch_size=10,
    eta=3.0,
    test_data=test_data
)
