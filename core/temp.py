for W, b, activation in zip(W_init, b_init, activations):
    layers.append(Layer(W, b, activation))


for layer in layers:
    params += layer.params


def output(self, x):
    '''
    Compute the MLP's output given an input

    :parameters:
        - x : theano.tensor.var.TensorVariable
            Theano symbolic variable for network input

    :returns:
        - output : theano.tensor.var.TensorVariable
            x passed through the MLP
    '''
    # Recursively compute output
    for layer in self.layers:
        x = layer.output(x)
    return x


def squared_error(self, x, y):
    '''
    Compute the squared euclidean error of the network output against the "true" output y

    :parameters:
        - x : theano.tensor.var.TensorVariable
            Theano symbolic variable for network input
        - y : theano.tensor.var.TensorVariable
            Theano symbolic variable for desired network output

    :returns:
        - error : theano.tensor.var.TensorVariable
            The squared Euclidian distance between the network output and y
    '''
    return T.sum((self.output(x) - y) ** 2)