###############################################################
#
#                      main.py
#
#      Assignment: A6 - Artificial Neural Networks
#      Authors:  Keren Zhou    (kzhou)
#      Date: 2020-12-14
#
#      Summary:
#      Main driver for the artificial neural network to 
#      classify a plant as Iris Setosa, Iris Versicolor or 
#      Iris Virginica by its sepal_length, sepal_width, 
#      petal_length and petal_width in cm.
#
###############################################################

import ann

import sys

# main
my_ann = ann.ANN()

# Train ANN.
if len(sys.argv) == 1:
    my_ann.train("ANN - Iris data.txt")
else:
    my_ann.train(sys.argv[1])

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
