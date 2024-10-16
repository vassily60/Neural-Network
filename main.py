import numpy as np
import pandas as pd
import os
import sys
import argparse

def sigmoid(x):
    return 1/(1 + np.exp(-x))

# Initializing Weights and Biases
def init_weights_biases(num_input_nodes, num_hidden_nodes, num_output_nodes):
    parameter_dictionary = {}
    hidden_biases = np.zeros((num_hidden_nodes, 1))
    output_biases = np.zeros((num_output_nodes, 1))
    hidden_weights = np.random.randn(num_hidden_nodes, num_input_nodes)
    output_weights = np.random.randn(num_output_nodes, num_hidden_nodes)
    parameter_dictionary["hidden_biases"] = hidden_biases
    parameter_dictionary["output_biases"] = output_biases
    parameter_dictionary["hidden_weights"] = hidden_weights
    parameter_dictionary["output_weights"] = output_weights
    return parameter_dictionary

# Reading Data from File
def read_file_to_array(file_name):
    df = pd.read_csv(file_name, sep='\t')

    features = df.iloc[:, :-1].values.astype(float)
    labels = df.iloc[:, -1].values.astype(float)
    
    features = features.T
    header_array = df.columns.values.reshape(-1,1)
    
    return features, labels.reshape(1,len(labels)), header_array

# Forward Propagation
def forward_propagate(feature_array, weights_biases_dict):
    hidden_layer_values = np.dot(weights_biases_dict["hidden_weights"], feature_array) + weights_biases_dict["hidden_biases"]
    hidden_layer_outputs = sigmoid(hidden_layer_values)

    # print(hidden_layer_outputs)

    output_layer_values = np.dot(weights_biases_dict["output_weights"], hidden_layer_outputs) + weights_biases_dict["output_biases"]
    output_layer_outputs = sigmoid(output_layer_values)

    # print(output_layer_outputs)

    return {"hidden_layer_outputs": hidden_layer_outputs,
            "output_layer_outputs": output_layer_outputs}
    

# Calculating Loss
def find_loss(output_layer_outputs, labels):
    # The number of examples is the number of columns in labels
    num_examples = labels.shape[1]
    loss = (-1 / num_examples) * np.sum(np.multiply(labels, np.log(output_layer_outputs)) +
    np.multiply(1-labels, np.log(1-output_layer_outputs)))
    return loss

# Backpropagation
def backprop(feature_array, labels, output_vals, weights_biases_dict, verbose=False):
    if verbose:
        print()
    # We get the number of examples by looking at how many total
    # labels there are. (Each example has a label.)
    num_examples = labels.shape[1]

    # These are the outputs that were calculated by each
    # of our two layers of nodes that calculate outputs.
    hidden_layer_outputs = output_vals["hidden_layer_outputs"]
    output_layer_outputs = output_vals["output_layer_outputs"]
    # These are the weights of the arrows coming into our output
    # node from each of the hidden nodes. We need these to know
    # how much blame to place on each hidden node.
    output_weights = weights_biases_dict["output_weights"]
    # This is how wrong we were on each of our examples, and in
    # what direction. If we have four training examples, there
    # will be four of these.
    # This calculation works because we are using binary cross-entropy,
    # which produces a fairly simply calculation here.
    raw_error = output_layer_outputs - labels
    if verbose:
        print("raw_error", raw_error)

    # This is where we calculate our gradient for each of the
    # weights on arrows coming into our output.
    output_weights_gradient = np.dot(raw_error, hidden_layer_outputs.T)/num_examples
    if verbose:
        print("output_weights_gradient", output_weights_gradient)

    # This is our gradient on the bias. It is simply the
    # mean of our errors.
    output_bias_gradient = np.sum(raw_error, axis=1, keepdims=True)/num_examples
    if verbose:
        print("output_bias_gradient", output_bias_gradient)

    # We now calculate the amount of error to propagate back to our hidden nodes.
    # First, we find the dot product of our output weights and the error
    # on each of four training examples. This allows us to figure out how much,
    # for each of our training examples, each hidden node contributed to our
    # getting things wrong.
    blame_array = np.dot(output_weights.T, raw_error)
    if verbose:
        print("blame_array", blame_array)

    # hidden_layer_outputs is the actual values output by our hidden layer for
    # each of the four training examples. We square each of these values.
    hidden_outputs_squared = np.power(hidden_layer_outputs, 2)
    if verbose:
        print("hidden_layer_outputs", hidden_layer_outputs)
        print("hidden_outputs_squared", hidden_outputs_squared)

    # We now multiply our blame array by 1 minus the squares of the hidden layer's
    # outputs.
    propagated_error = np.multiply(blame_array, 1-hidden_outputs_squared)
    if verbose:
        print("propagated_error", propagated_error)

    # Finally, we compute the magnitude and direction in which we
    # should adjust our weights and biases for the hidden node.
    hidden_weights_gradient = np.dot(propagated_error, feature_array.T)/num_examples
    hidden_bias_gradient = np.sum(propagated_error, axis=1,
    keepdims=True)/num_examples
    if verbose:
        print("hidden_weights_gradient", hidden_weights_gradient)
        print("hidden_bias_gradient", hidden_bias_gradient)

    # A dictionary that stores all of the gradients
    # These are values that track which direction and by
    # how much each of our weights and biases should move
    gradients = {"hidden_weights_gradient": hidden_weights_gradient,
    "hidden_bias_gradient": hidden_bias_gradient,
    "output_weights_gradient": output_weights_gradient,
    "output_bias_gradient": output_bias_gradient}
    return gradients

# Updating Weights and Biases
def update_weights_biases(parameter_dictionary, gradients, learning_rate):

    new_hidden_weights = parameter_dictionary["hidden_weights"] - learning_rate*gradients["hidden_weights_gradient"]
    new_hidden_biases = parameter_dictionary["hidden_biases"] - learning_rate*gradients["hidden_bias_gradient"]
    new_output_biases = parameter_dictionary["output_biases"] - learning_rate*gradients["output_bias_gradient"]
    new_output_weights = parameter_dictionary["output_weights"] - learning_rate*gradients["output_weights_gradient"]
    
    return {"hidden_biases": new_hidden_biases, 
            "output_biases":new_output_biases,
            "hidden_weights": new_hidden_weights,
            "output_weights": new_output_weights}


# Training the Network
def train_network(file_name, num_inputs, num_hiddens, num_outputs, epochs, learning_rate):
    weights_biases_dict = init_weights_biases(num_inputs,num_hiddens,num_outputs)
    for i in range(epochs):
        
        features, labels, headers = read_file_to_array(file_name)
        
        output_vals = forward_propagate(features, weights_biases_dict)
        if i % 100 == 0: 
            loss = find_loss(output_vals["output_layer_outputs"],labels)
            print(f"loss: {loss}")

        gradient_dic = backprop(features, labels, output_vals, weights_biases_dict)

        weights_biases_dict = update_weights_biases(weights_biases_dict,gradient_dic, learning_rate)
    return weights_biases_dict
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='File')
    parser.add_argument('-i', help='input file', required=True)
    args = parser.parse_args()

    # Check if the file exists
    if not (os.path.isfile(args.i)):
        print("error", args.i, "does not exist, exiting.", file = sys.stderr)
        exit(-1)
    file = args.i

    num_input = 2
    num_hidden = 2
    num_output = 1
    epochs = 1000
    learning_rate = 0.5
    #learning rate 0.5 is really good but  not 0.6
    print(train_network(args.i, num_input,num_hidden,num_output,epochs,learning_rate))



    # weights_biases_dict = init_weights_biases(2,2,1)
    # print(weights_biases_dict)
    # features, labels, headers = read_file_to_array(file)
    
    # output_vals = forward_propagate(features, weights_biases_dict)

    # loss = find_loss(output_vals["output_layer_outputs"],labels)

    # gradient_dic = backprop(features, labels, output_vals, weights_biases_dict)

    # print(update_weights_biases(weights_biases_dict,gradient_dic, 0.3))
    
