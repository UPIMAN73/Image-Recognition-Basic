# Local Class Imports
from Classes.nnimage import NNImage
from Classes.cnnitem import CNNItem
from Classes.traininfo import TrainInfo

# System Imports
from os import listdir
from os.path import isfile, join
from math import sqrt, factorial
from random import random, randint
from datetime import datetime
import json


# Matrix and Array based calculations and declarations
def zeroArray(arry):
    res = []
    for i in range(0, len(arry)):
        res.append(0)
    return res

# Matrix Sum
def matrixSum(matrix):
  res = 0
  for i in matrix:
    res += i
  return res


# Final Layer for Neural Networks
def finalLayer(pmatrix, pools):
    ansArray = []

    # iterate through the pools
    for i in pools:
        ansA = []

        # go through the pool matrix from pool layer
        for j in range(0, len(pmatrix)):
          ans = 0

          # get sum from pool to later compare the values and then classify
          for k in range(0, len(pmatrix[j])):
               ans += pmatrix[j][k] * i[j][k]
        
          # Append answer values to anwer array
          ansA.append(ans)
        
        # Take the matrix sum and append it to answer arrays of pools
        ansArray.append(matrixSum(ansA))

    # return the index of the network items
    return ansArray.index(max(ansArray))


# TODO neural network setup
img_dir = "./Images"
image_names = [f for f in listdir(img_dir) if ( isfile(join(img_dir, f)) and (".png" in join(img_dir, f)) )]
images_array = []
image_matrix = []


# NN Arrays for calculating and applying NN
item_array = []
conv_array = []
pool_array = []

# Train Stats
train_stats = []
for i in image_names:
  train_stats.append(TrainInfo(i.split("_")[0].capitalize()))

# Filter Setup
filter = [1, 1, 1, 1]
num_of_states = 2    # Highly important if you want to setup the correct process for your NN
filter_array = []
desired_filter_length = pow(len(filter), num_of_states) #factorial(len(filter)) * num_of_states
ranfilter = []

# setting up preset arrays

while len(filter_array) != desired_filter_length:
  ranfilter = []
  for i in range(0, len(filter)):
      if randint(0, 1) == 1:
        ranfilter.append(1)
      else:
        ranfilter.append(-1)
  if ranfilter not in filter_array:
    filter_array.append(ranfilter)
print("Done setting up filter arrays")
# print(filter_array)

# Analyzing matrix size
amsize = int(sqrt(len(filter)))

# TODO Must add in a filter collection and have them sorted based on their correction scores

# Accuracy lists
correct = []
correct_array = []
correct_perc_array = []

for filter in filter_array:
  # file name for loop 
  correct = []
  for fname in image_names:
    # First Encode and setup image for NN
    img = NNImage(join(img_dir, fname))
    image_matrix.append(img.encodedMatrix)
    images_array.append(img)

    # Then Setup and calculate for NN
    # Convolution Layer
    convolution = CNNItem(fname.split("_")[0].capitalize(), img.encodedMatrix, filter, ams=amsize)
    conv_array.append(convolution.cmatrix)

    # Pooling Layer
    convolution.pmatrix = convolution.poolLayer(convolution.cmatrix, 3, -3)
    pool_array.append(convolution.pmatrix)

    # Item Layer
    item_array.append(convolution)

    # Final Layer
    guess_name = item_array[finalLayer(convolution.pmatrix, pool_array)].name

    # Correction appending and printouts
    if (guess_name == convolution.name):
      #  print("Correct Guess: " + guess_name)
       correct.append(convolution)
       for i in train_stats:
         if i.name == guess_name:
           i.filters.append(filter)
    # else:
    #   print("Incorrect Guess: " + guess_name + "    is actually  " + convolution.name)

  # Printout of accuracy of the entire NN simulation
  # print("\n\n\n\n")
  # print("# of Images Guessed Correctly: " + str(len(correct)))
  correct_percentage = len(correct) / (1.0 * len(image_names)) * 100
  correct_perc_array.append(correct_percentage)
  correct_array.append(len(correct))
  print("% Value of Images Correct: " + str(correct_percentage) + "%")

# Only do this if you want to generate random filters
# # generate random filter
# filter = []
# for i in range(0, 4):
#   if randint(0, 6) >= 3:
#     filter.append(random() * -1)
#   else:
#     filter.append(random())


# Writing filter values to an outfile so that we can reload them at some point
correct_array = sorted(correct_array)
print(correct_perc_array)
max_correct_indexes = [i for i, x in enumerate(correct_array) if x == max(correct_array)]
max_filters = []
for i in sorted(max_correct_indexes, reverse=True):
  print("Highest Filter Correction Value: " + str(filter_array[i]))
  print("Highest Correction Percentage Value: " + str(correct_perc_array[i]))
  print("Highest Correction Value: " + str(correct_array[i]))
  max_filters.append(filter_array[i])



outFile = open("filters.txt", "w")

for i in max_correct_indexes:
  filter = filter_array[i]
  print("Filter Value: " + str(filter))
  outFile.write(str(filter) + "\n")

for i in range(0, len(filter_array)):
  if i in max_correct_indexes:
    continue
  else:
    filter = filter_array[i]
    outFile.write(str(filter) + "\n")

outFile.close()

# write out training stats
json_file = open("training_stats.json", "w")
obj_setup = {"date" : str(datetime.now()), "objects" : []}


for i in train_stats:
  obj_setup["objects"].append({"name" : i.name, "filters" : i.filters})

json_file.write(str(str(json.dumps(obj_setup)).encode("ascii").decode("ascii")))
json_file.close()