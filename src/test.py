###############################################################
#
#                      test.py
#
#      Assignment: A6 - Artificial Neural Networks
#      Authors:  Keren Zhou    (kzhou)
#      Date: 2020-12-14
#
#      Summary:
#      Test driver and playground.
#
###############################################################

import ann

import math
import random

#############################################################
# # Test sigmod function.
# for x in range(-10, 10):
#     print("x = " + str(x) + ", sigmoid(x) = ", str(ann.sigmoid(x)))
#############################################################

#############################################################
# # Test forward propagation.
# my_ann = ann.ANN()
# my_ann.bias_weights = [[1, 2, 3, 4], [5, 6, 7], [8, 9, 10]]
# my_ann.weights[0][0] = [1, 3, 5]
# my_ann.weights[0][1] = [2, 4, 6]
# my_ann.weights[0][2] = [7, 8, 9]
# my_ann.weights[0][3] = [1, 2, 3]
# my_ann.weights[1][0] = [1, 2, 3]
# my_ann.weights[1][1] = [4, 5, 6]
# my_ann.weights[1][2] = [7, 8, 9]
# my_ann.forward_propagation(1, 2, 3, 4)

# # Test backward propagation.
# my_ann.outputs = [[1, 2, 3, 4], [5, 6, 7], [8, 9, 10]]
# my_ann.backward_propagation(1, 0, 0)

# # Test get_result()
# my_ann.outputs[2] = [1, 2, 3]
# print("output layer:", my_ann.outputs[2])
# print("result", my_ann.get_result())


# Test all.
my_ann = ann.ANN()

# Train ANN.
my_ann.train("raw_example.txt")

# Interactive classification.
print("######## Classification Phase ########")
while True:
    print("Please input an example to classify.")
    print("Input a negative value to stop, such as -1.")

    sepal_length = float(input("Input sepal length in cm: "))
    if sepal_length < 0:
        break

    sepal_width = float(input("Input sepal width in cm: "))
    if sepal_width < 0:
        break

    petal_length = float(input("Input petal length in cm: "))
    if petal_length < 0:
        break

    petal_width = float(input("Input petal width in cm: "))
    if petal_width < 0:
        break

    result = my_ann.classify(sepal_length, sepal_width, petal_length, 
                             petal_width)
    print("predicted Iris type:", result)
    print()
print("######## Classification End ########")
#############################################################
