###############################################################
#
#                      ann.py
#
#      Assignment: A6 - Artificial Neural Networks
#      Authors:  Keren Zhou    (kzhou)
#      Date: 2020-12-14
#
#      Summary:
#      Implementation for the artificial neural network to 
#      classify a plant as Iris Setosa, Iris Versicolor or 
#      Iris Virginica by its sepal_length, sepal_width, 
#      petal_length and petal_width in cm.
#
###############################################################

import math
import random
import copy

class Example:
    """
    Class of formatted examples.
    Each formatted example has a sepal length, a sepal width, a petal length 
    and a petal width as inputs, and has 3 degrees of certainty for Iris 
    Setosa, Iris Versicolour, and Iris Virginica.
    Typically, exact one degree of certainty is 1 and the other 2 are 0.
    """
    def __init__(self, sepal_length, sepal_width, petal_length, petal_width, 
                 iris_type):
        """
        Init a formatted example by the given raw data.

        Parameters:
        sepal_length: Sepal length in cm.
        sepal_width:  Sepal width in cm.
        petal_length: Petal length in cm.
        petal_width: Petal width in cm.
        iris_type: Iris type (Iris Setosa, Iris Versicolour or Iris Virginica).
        
        Returns: Initialized example object.
        """
        # Init inputs.
        self.sepal_length = sepal_length
        self.sepal_width = sepal_width
        self.petal_length = petal_length
        self.petal_width = petal_width

        # Init outputs.
        self.setosa = 0 # Degree of certainty for a class of Iris.
        self.versicolour = 0
        self.virginica = 0
        if iris_type == "Iris-setosa":
            self.setosa = 1
        elif iris_type == "Iris-versicolor":
            self.versicolour = 1
        elif iris_type == "Iris-virginica":
            self.virginica = 1
        
        return


    def __str__(self):
        return "[" + str(self.sepal_length) + ", " \
                   + str(self.sepal_width) + ", " \
                   + str(self.petal_length) + ", " \
                   + str(self.petal_width) + ", " \
                   + str(self.setosa) + ", " \
                   + str(self.versicolour) + ", " \
                   + str(self.virginica) + "]"


class ANN:
    """
    Class of artificial neuron network (ANN) for classifying a plant as Iris 
    Setosa, Iris Versicolour, or Iris Virginica by its sepal length, sepal 
    width, petal length and petal width in cm.
    """
    def __init__(self):
        """
        Init an ANN by the given ANN weight file.
        If the ANN weight file is not provided, randomize the weights.

        Parameters:
        None
        
        Returns: Initialized ANN object.
        """
        # Topology structure.
        self.STRUCTURE = [4, 4, 3] # STRUCTURE[i] is #neuron in layer i.
                # Layer 0 is input layer. It has 4 inputs.
                # Layer 2 is output layer. It has 3 outputs. Each output is the 
                # degree of certainty for a specific Iris type.
                # Layer 1 is the hidden layer. It has 6 neuron.
                # Each neuron on layer i has a directed connection to each 
                # layer on layer i+1.
        self.INPUT_LAYER = 0
        self.OUTPUT_LAYER = 2
        self.SEPAL_LENGTH_INPUT = 0
        self.SEPAL_WIDTH_INPUT = 1
        self.PETAL_LENGTH_INPUT = 2
        self.PETAL_WIDTH_INPUT = 3
        self.SETOSA_OUTPUT = 0
        self.VERSICOLOUR_OUTPUT = 1
        self.VIRGINICA_OUTPUT = 2
        
        self.BOUND = 3 # Scale inputs betwen -bound and +bound.
        self.LEARNING_RATE = 0.1

        # Init bias.
        self.BIAS_VALUE = 1
        # bias_weights[i][j] is the bias weight to the neuron j on layer i.
        self.bias_weights = [] 
        for width in self.STRUCTURE: # #neurons in this layer.
            layer_bias_weights = [] # Bias weights in this layer.
            for i in range(0, width):
                layer_bias_weights.append(random.uniform(-0.1, 0.1))
            self.bias_weights.append(layer_bias_weights)
        
        # Init neurons.
        # For neuron j on layer i, its potential is potentials[i][j], its 
        # output is outputs[i][j], its delta is delta[i][j].
        self.potentials = []
        for width in self.STRUCTURE: # #neurons in this layer.
            layer_potentials = [] # Potentials in this layer.
            for i in range(0, width):
                layer_potentials.append(0)
            self.potentials.append(layer_potentials)

        # Outputs and deltas share the same structure as potentials.
        self.outputs = copy.deepcopy(self.potentials)
        self.deltas = copy.deepcopy(self.potentials)

        # weights[i][j][k] is the weight from neuron j on layer i to newron k 
        # on layer i+1.
        self.weights = []
        for parent_layer in range(0, len(self.STRUCTURE) - 1):
            layer_weights = [] # Weights of neurons in this layer to the next 
                    # layer.
            for parent_neuron in range(0, self.STRUCTURE[parent_layer]): 
                neuron_weights = [] # Weights of this neuron to other neurons 
                        # in the next layer.
                for child_neuron in range(0, self.STRUCTURE[parent_layer + 1]):
                    neuron_weights.append(random.uniform(-0.3, 0.3))
                layer_weights.append(neuron_weights)
            self.weights.append(layer_weights)
        
        return


    def read_raw_examples(self, raw_example_path):
        """
        Read raw examples from the given file and create a list of examples.

        Parameters:
        raw_example_path: Path of file containing raw examples.
        
        Returns: Nothing.
        """
        examples = []
        data_file = open(raw_example_path, "r")
        for line in data_file:
            raw_strings = line.rstrip().split(",")
            if raw_strings != ['']:
                sepal_length = float(raw_strings[0])
                sepal_width = float(raw_strings[1])
                petal_length = float(raw_strings[2])
                petal_width = float(raw_strings[3])
                iris_type = raw_strings[4]
                example = Example(sepal_length, sepal_width, petal_length, 
                                  petal_width, iris_type)
                examples.append(example)

        return examples


    def decorrelate(self, value, mean, var):
        """
        Decorrelate the given value by its mean and variance (SD^2).
        y = (x - mean) / var

        Parameters:
        value: Value to decorrelate.
        mean: Mean of the values.
        var: Variance of the values.

        Returns: Decorrelated value.
        """
        return (value - mean) / var

    
    def decorrelate_all(self, examples):
        """
        Decorrelate the given list of examples by its mean and variance (SD^2).
        y = (x - mean) / var

        Parameters:
        examples: List of examples to decorrelate.

        Returns: Decorrelated examples.
        """
        # Analyze examples.
        # Average.
        total_examples = len(examples)
        total_sepal_length = 0
        total_sepal_width = 0
        total_petal_length = 0
        total_petal_width = 0
        # Variance and standard deviation.
        total_squared_sepal_length = 0
        total_squared_sepal_width = 0
        total_squared_petal_length = 0
        total_squared_petal_width = 0
        for example in examples:
            # Average.
            total_sepal_length += example.sepal_length
            total_sepal_width += example.sepal_width
            total_petal_length += example.petal_length
            total_petal_width += example.petal_width
            # Variance and standard deviation.
            total_squared_sepal_length += example.sepal_length \
                                          * example.sepal_length
            total_squared_sepal_width += example.sepal_width \
                                         * example.sepal_width
            total_squared_petal_length += example.petal_length \
                                          * example.petal_length
            total_squared_petal_width += example.petal_width \
                                         * example.petal_width
        # Average.
        self.average_sepal_length = total_sepal_length / total_examples
        self.average_sepal_width = total_sepal_width / total_examples
        self.average_petal_length = total_petal_length / total_examples
        self.average_petal_width = total_petal_width / total_examples
        # Variance.
        average_squared_sepal_length = total_squared_sepal_length \
                                       / total_examples
        average_squared_sepal_width = total_squared_sepal_width \
                                      / total_examples
        average_squared_petal_length = total_squared_petal_length \
                                       / total_examples
        average_squared_petal_width = total_squared_petal_width \
                                      / total_examples
        self.var_sepal_length = average_squared_sepal_length \
                                - self.average_sepal_length \
                                * self.average_sepal_length
        self.var_sepal_width = average_squared_sepal_width \
                               - self.average_sepal_width \
                               * self.average_sepal_width
        self.var_petal_length = average_squared_petal_length \
                                - self.average_petal_length \
                                * self.average_petal_length
        self.var_petal_width = average_squared_petal_width \
                               - self.average_petal_width \
                               * self.average_petal_width

        # De-correlate examples by variance.
        for example in examples:
            example.sepal_length \
                    = self.decorrelate(example.sepal_length, 
                                       self.average_sepal_length, 
                                       self.var_sepal_length)
            example.sepal_width \
                    = self.decorrelate(example.sepal_width, 
                                       self.average_sepal_width, 
                                       self.var_sepal_width)
            example.petal_length \
                = self.decorrelate(example.petal_length, 
                                   self.average_petal_length, 
                                   self.var_petal_length)
            example.petal_width \
                = self.decorrelate(example.petal_width, 
                                   self.average_petal_width, 
                                   self.var_petal_width)

        return examples


    def scale(self, value, scale, bound):
        """
        Linearly scale the given value to the range between -bound and +bound.

        Parameters:
        value: Value to decorrelate.
        scale: Max or -min for the value.
        bound: Upper and lower bound to scale to.

        Returns: Scaled value.
        """
        return value / scale * bound


    def scale_all(self, examples, bound):
        """
        Linearly scale the given list of examples making each input between 
        -bound and +bound.

        Parameters:
        examples: List of examples to scale.

        Returns: Scaled examples.
        """
        # Find out max and min.
        max_sepal_length = examples[0].sepal_length
        max_sepal_width = examples[0].sepal_width
        max_petal_length = examples[0].petal_length
        max_petal_width = examples[0].petal_width
        min_sepal_length = examples[0].sepal_length
        min_sepal_width = examples[0].sepal_width
        min_petal_length = examples[0].petal_length
        min_petal_width = examples[0].petal_width
        for example in examples:
            max_sepal_length = max(max_sepal_length, example.sepal_length)
            max_sepal_width = max(max_sepal_width, example.sepal_width)
            max_petal_length = max(max_petal_length, example.petal_length)
            max_petal_width = max(max_petal_width, example.petal_width)
            min_sepal_length = min(min_sepal_length, example.sepal_length)
            min_sepal_width = min(min_sepal_width, example.sepal_width)
            min_petal_length = min(min_petal_length, example.petal_length)
            min_petal_width = min(min_petal_width, example.petal_width)

        # Compute scale facter based on |max| and |min|.
        if min_sepal_length < 0:
            min_sepal_length = -min_sepal_length
        self.scale_sepal_length = max(max_sepal_length, min_sepal_length)
        if min_sepal_width < 0:
            min_sepal_width = -min_sepal_width
        self.scale_sepal_width = max(max_sepal_width, min_sepal_width)
        if min_petal_length < 0:
            min_petal_length = -min_petal_length
        self.scale_petal_length = max(max_petal_length, min_petal_length)
        if min_petal_width < 0:
            min_petal_width = -min_petal_width
        self.scale_petal_width = max(max_petal_width, min_petal_width)

        # Scale examples.
        for example in examples:
            # example.sepal_length \
            #     = example.sepal_length / self.scale_sepal_length * bound
            example.sepal_length = self.scale(example.sepal_length, 
                                              self.scale_sepal_length, 
                                              self.BOUND)
            # example.sepal_width \
            #     = example.sepal_width / self.scale_sepal_width * bound
            example.sepal_width = self.scale(example.sepal_width, 
                                              self.scale_sepal_width, 
                                              self.BOUND)
            # example.petal_length \
            #     = example.petal_length / self.scale_petal_length * bound
            example.petal_length = self.scale(example.petal_length, 
                                              self.scale_petal_length, 
                                              self.BOUND)
            # example.petal_width \
            #     = example.petal_width / self.scale_petal_width * bound
            example.petal_width = self.scale(example.petal_width, 
                                              self.scale_petal_width, 
                                              self.BOUND)

        return examples


    def split_sets(self, examples, training_set, validation_set, test_set):
        """
        Randomly split the given examples into a training set (50%), a 
        validation set (25%) and a test set (25%).

        Parameters:
        examples: Examples to split.
        training_set: Training set.
        validation_set: Validation set.
        test_set: Test set.

        Returns: Nothing.
        """
        random.shuffle(examples)
        total = len(examples)
        training_set_end = int(total / 2) # End example index for training set.
        validation_set_end = int(total * 3 / 4) # End example index for 
                # validation set.
        for i in range(0, training_set_end):
            training_set.append(examples[i])
        for i in range(training_set_end, validation_set_end):
            validation_set.append(examples[i])
        for i in range(validation_set_end, total):
            test_set.append(examples[i])
        
        return


    def prepare(self, raw_example_path, training_set, validation_set, test_set):
        """
        Read examples from the given file, format them and divide them into a 
        training set, a validation set and a test set.

        Parameters:
        raw_example_path: Path of raw example file.
        training_set: List of examples for training.
        validation_set: List of examples for validation.
        test_set: List of examples for test.
        
        Returns: Nothing.
        """
        # Read raw examples.
        examples = self.read_raw_examples(raw_example_path)

        # De-correlate examples.
        self.decorrelate_all(examples)

        # Scale examples.
        self.scale_all(examples, self.BOUND)
        
        # Separate examples by Iris type.
        setosa_examples = []
        versicolour_examples = []
        virginica_examples = []
        for example in examples:
            if example.setosa == 1:
                setosa_examples.append(example)
            elif example.versicolour == 1:
                versicolour_examples.append(example)
            elif example.virginica == 1:
                virginica_examples.append(example)

        # Divide examples into a training set (50%), a validation set (25%) and 
        # a test set(25%).
        self.split_sets(setosa_examples, training_set, validation_set, test_set)
        self.split_sets(versicolour_examples, training_set, validation_set, test_set)
        self.split_sets(virginica_examples, training_set, validation_set, test_set)
        # Shuffle the order of examples.
        random.shuffle(training_set)
        random.shuffle(validation_set)
        random.shuffle(test_set)

        return

    def sigmoid(self, potential):
        """
        Sigmoid activation function.
    
        Parameters:
        potential: Neuron potential
            
        Returns: Neuron output.
        """
        return 1 / (1 + math.exp(-potential))
    
    
    def sigmoid_derivative(self, potential):
        """
        Derivative of the Sigmoid activation function.
    
        Parameters:
        potential: Potential of the neuron.
            
        Returns: Derivative of neuron output.
        """
        output = sigmoid(potential)
        return output * (1 - output)


    def forward_propagation(self, sepal_length, sepal_width, petal_length, 
                            petal_width):
        """
        Forward propagation.
        Update potentials and outputs of each neuron by the given inputs.

        Parameters:
        sepal_length: Sepal length.
        sepal_width: Sepal length.
        petal_length: Petal length.
        petal_width: Petal width.

        Returns: Iris type with the highest degree of certainty.
        """
        # Init outputs of input neurons.
        self.outputs[self.INPUT_LAYER][self.SEPAL_LENGTH_INPUT] = sepal_length
        self.outputs[self.INPUT_LAYER][self.SEPAL_WIDTH_INPUT] = sepal_width
        self.outputs[self.INPUT_LAYER][self.PETAL_LENGTH_INPUT] = petal_length
        self.outputs[self.INPUT_LAYER][self.PETAL_WIDTH_INPUT] = petal_width

        # Update potentials and outputs of other neurons layer by layer.
        for layer in range(1, len(self.STRUCTURE)):
            for child in range(0, self.STRUCTURE[layer]):
                self.potentials[layer][child] = 0
                # w * B
                self.potentials[layer][child] \
                        += self.bias_weights[layer][child] * self.BIAS_VALUE
                # Sum of w * o
                for parent in range(0, self.STRUCTURE[layer - 1]):
                    self.potentials[layer][child] \
                            += self.weights[layer - 1][parent][child] \
                            * self.outputs[layer - 1][parent] 
                # Update output.
                self.outputs[layer][child] \
                        = self.sigmoid(self.potentials[layer][child])

        return 


    def backward_propagation(self, setosa, versicolour, 
                              virginica):
        """
        Backward propagation.
        Update weights of neurons and the bias by the given target outputs.

        Parameters:
        setosa: Degree of certainty for Iris Setosa.
        versicolour:  Degree of certainty for Iris Versicolour.
        virginica:  Degree of certainty for Iris Virginica.

        Returns: Nothing.
        """
        # Derivative of sigmoid(x) = sigmoid(x) * (1 - sigmoid(x)).
        # Thus, derivative of outputs[i][j] = outputs[i][j] 
        # * (1 - outputs[i][j])

        # Compute error of output layer.
        # Iris Setosa
        target = setosa
        output = self.outputs[self.OUTPUT_LAYER][self.SETOSA_OUTPUT]
        self.deltas[self.OUTPUT_LAYER][self.SETOSA_OUTPUT] \
                = output * (1 - output) * (target - output)
        # Iris Versicolour
        target = versicolour
        output = self.outputs[self.OUTPUT_LAYER][self.VERSICOLOUR_OUTPUT]
        self.deltas[self.OUTPUT_LAYER][self.VERSICOLOUR_OUTPUT] \
                = output * (1 - output) * (target - output)
        # Iris Virginica
        target = virginica
        output = self.outputs[self.OUTPUT_LAYER][self.VIRGINICA_OUTPUT]
        self.deltas[self.OUTPUT_LAYER][self.VIRGINICA_OUTPUT] \
                = output * (1 - output) * (target - output)

        # Compute error of hidden layers.
        for layer in range(self.OUTPUT_LAYER - 1, self.INPUT_LAYER, -1):
            for parent in range(0, self.STRUCTURE[layer]):
                self.deltas[layer][parent] = 0
                for child in range(0, self.STRUCTURE[layer + 1]):
                    self.deltas[layer][parent] \
                        += self.weights[layer][parent][child] \
                        * self.deltas[layer + 1][child]
                self.deltas[layer][parent] \
                        *= self.outputs[layer][parent] \
                        * (1 - self.outputs[layer][parent])

        # Update weights.
        for layer in range(self.OUTPUT_LAYER, self.INPUT_LAYER, -1):
            for child in range(0, self.STRUCTURE[layer]):
                child_delta = self.deltas[layer][child]
                # Update bias weight.
                self.bias_weights[layer][child] \
                        += self.LEARNING_RATE * self.BIAS_VALUE * child_delta
                # Update neuron weight.
                for parent in range(0, self.STRUCTURE[layer - 1]):
                    self.weights[layer - 1][parent][child] \
                            += self.LEARNING_RATE \
                            * self.outputs[layer - 1][parent] * child_delta

        return


    def validate(self, examples):
        """
        Classify the given examples by the current ANN and compute the mean 
        square error of outputs.

        Parameters:
        examples: List of examples to classify.
        
        Returns: Sum of MSE(Iris Setosa), MSE(Iris Versicolour) and 
                 MSE(Iris Virginica).
        """
        # Mean square error for each output.
        mse_setosa = 0 
        mse_versicolour = 0
        mse_virginica = 0
        for e in examples:
            # Get outputs.
            self.forward_propagation(e.sepal_length, e.sepal_width, 
                                     e.petal_length, e.petal_width)

            # Compute error for Iris Setosa.
            error = e.setosa \
                    - self.outputs[self.OUTPUT_LAYER][self.SETOSA_OUTPUT]
            mse_setosa += error * error

            # Compute error for Iris Versicolour.
            error = e.versicolour \
                    - self.outputs[self.OUTPUT_LAYER][self.VERSICOLOUR_OUTPUT]
            
            # Compute error for Iris Virginica
            error = e.virginica \
                    - self.outputs[self.OUTPUT_LAYER][self.VIRGINICA_OUTPUT]
        
        total = len(examples)
        mse_setosa /= total
        mse_versicolour /= total
        mse_virginica /= total

        print("MSE(Iris Setosa) = " + str(mse_setosa))
        print("MSE(Iris Versicolour) = " + str(mse_setosa))
        print("MSE(Iris Virginica) = " + str(mse_setosa))

        return mse_setosa + mse_versicolour + mse_virginica


    def get_result(self):
        """
        Get the final output of the current ANN.
        
        Parameters:
        None.
        
        Returns: Iris type that has the highest degree of certainty.
                 self.SETOSA_OUTPUT (0) if Iris Setosa;
                 self.VERSICOLOUR_OUTPUT (1) if Iris Versicolour;
                 self.VIRGINICA_OUTPUT (2) if Iris Virginica.
        """
        setosa = self.outputs[self.OUTPUT_LAYER][self.SETOSA_OUTPUT]
        versicolour = self.outputs[self.OUTPUT_LAYER][self.VERSICOLOUR_OUTPUT]
        virginica = self.outputs[self.OUTPUT_LAYER][self.VIRGINICA_OUTPUT]
        max_output = max(setosa, versicolour, virginica)
        if setosa == max_output:
            return self.SETOSA_OUTPUT
        elif versicolour == max_output:
            return self.VERSICOLOUR_OUTPUT
        elif virginica == max_output:
            return self.VIRGINICA_OUTPUT
        # Error.
        return -1


    def test(self, examples):
        """
        Classify the given examples by the current ANN and compute accuracy.

        Parameters:
        examples: List of examples to classify.
        
        Returns: Nothing.
        """
        # Confusion matrix.
        # confusion[i][j] = rate that type i is classified as type j.
        confusion = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

        # Test with each example.
        for e in examples:
            # Get outputs.
            self.forward_propagation(e.sepal_length, e.sepal_width, 
                                     e.petal_length, e.petal_width)
            
            # Compute accuracy.
            if e.setosa == 1:
                confusion[self.SETOSA_OUTPUT][self.get_result()] += 1
            elif e.versicolour == 1:
                confusion[self.VERSICOLOUR_OUTPUT][self.get_result()] += 1
            elif e.virginica == 1:
                confusion[self.VIRGINICA_OUTPUT][self.get_result()] += 1
        
        accuracy = 0
        accuracy += confusion[self.SETOSA_OUTPUT][self.SETOSA_OUTPUT]
        accuracy += confusion[self.VERSICOLOUR_OUTPUT][self.VERSICOLOUR_OUTPUT]
        accuracy += confusion[self.VIRGINICA_OUTPUT][self.VIRGINICA_OUTPUT]
        accuracy /= len(examples)

        # Convert counts to percentage.
        for row in range(0, len(confusion)):
            # Compute subtotal of one actual Iris type.
            subtotal = 0 
            for col in range(0, len(confusion[row])):
                subtotal += confusion[row][col]
            # Convert counts to percentage.
            for col in range(0, len(confusion[row])):
                confusion[row][col] /= subtotal
                confusion[row][col] *= 100
                confusion[row][col] = int(confusion[row][col])

        print("counfusion matrix (%):")
        for row in confusion:
            print(row)
        print("Row 1: Actual Iris-setosa")
        print("Row 2: Actual Iris-versicolor")
        print("Row 3: Actual Iris-virginica")
        print("Col 1: Predicted Iris-setosa")
        print("Col 2: Predicted Iris-versicolor")
        print("Col 3: Predicted Iris-virginica")

        return accuracy


    def train(self, raw_example_path):
        """
        Train the ANN with the given raw example data. 
        After training, create an ANN weight file.

        Parameters:
        raw_example_path: Path of raw example file.
        
        Returns: Nothing.
        """

        # Prepare examples.
        training_set = []
        validation_set = []
        test_set = []
        self.prepare(raw_example_path, training_set, validation_set, test_set)

        # Train ANN.
        print("######## Training Phase ########")
        prev_mse = 0
        for round in range(0, 10000):
            print("round", round)
            # Train with training set for a round.
            for e in training_set:
                self.forward_propagation(e.sepal_length, e.sepal_width, 
                                         e.petal_length, e.petal_width)
                self.backward_propagation(e.setosa, e.versicolour, e.virginica)
            curr_mse = self.validate(validation_set)
            print("MSE(sum) = " + str(curr_mse))
            # accuracy = self.test(validation_set)
            # print("accuracy(all) = " + str(accuracy))
            print()
            if abs(curr_mse - prev_mse) / curr_mse < 0.001:
                break
            prev_mse = curr_mse
        print("######## Training End ########")
        print()


        # Test ANN.
        print("######## Test Phase ########")
        accuracy = self.test(test_set)
        print()
        print("accuracy(all) = " + str(accuracy))
        print()
        print("######## Test End ########")
        print()

        return


    def classify(self, sepal_length, sepal_width, petal_length, petal_width):
        """
        Classify a plant as Iris Setosa, Iris Versicolour, or Iris Virginica 
        by the given sepal length, sepal width, petal length and petal width 
        in cm.

        Parameters:
        sepal_length: Sepal length in cm.
        sepal_width: Sepal length in cm.
        petal_length: Petal length in cm.
        petal_width: Petal width in cm.
        
        Returns: Iris-setosa, Iris-versicolor or Iris-virginica.
        """
        # De-correlate the input.
        sepal_length = self.decorrelate(sepal_length, 
                                        self.average_sepal_length, 
                                        self.var_sepal_length)
        sepal_width = self.decorrelate(sepal_width, 
                                        self.average_sepal_width, 
                                        self.var_sepal_width)
        petal_length = self.decorrelate(petal_length, 
                                        self.average_petal_length, 
                                        self.var_petal_length)
        petal_width = self.decorrelate(petal_width, 
                                        self.average_petal_width, 
                                        self.var_petal_width)

        # Scale the input.
        sepal_length = self.scale(sepal_length, 
                                  self.scale_sepal_length, 
                                  self.BOUND)
        sepal_width = self.scale(sepal_width, 
                                 self.scale_sepal_width, 
                                 self.BOUND)
        petal_length = self.scale(petal_length, 
                                 self.scale_petal_length, 
                                 self.BOUND)
        petal_width = self.scale(petal_width, 
                                 self.scale_petal_width, 
                                 self.BOUND)

        self.forward_propagation(sepal_length, sepal_width, petal_length, 
                                 petal_width)

        result = self.get_result()

        if result == self.SETOSA_OUTPUT:
            return "Iris-setosa"
        elif result == self.VERSICOLOUR_OUTPUT:
            return "Iris-versicolor"
        elif result == self.VIRGINICA_OUTPUT:
            return "Iris-virginica"
        return "Unknown"
