# Local Class Imports
from Classes.nnimage import NNImage
from Classes.cnnitem import CNNItem

# System Imports
from os import listdir
from os.path import isfile, join

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
image_names = [f for f in listdir("./Images") if ( isfile(join("./Images", f)) and (".png" in join("./Images", f)) )]
images_array = []
image_matrix = []

# NN Arrays for calculating and applying NN
item_array = []
conv_array = []
pool_array = []


# Filter Array
# Filter Array must be defined
filter = [1, 1, 1, 1]
#filter_array = []

# TODO Must add in a filter collection and have them sorted based on their correction scores

# Accuracy lists
correct = []
#correct_array = []

for fname in image_names:
  # First Encode and setup image for NN
  img = NNImage(join("./Images", fname))
  image_matrix.append(img.encodedMatrix)
  images_array.append(img)

  # Then Setup and calculate for NN
  # Convolution Layer
  convolution = CNNItem(fname.split("_")[0].capitalize(), img.encodedMatrix, filter)
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
    print("Correct Guess: " + guess_name)
    correct.append(convolution)
  else:
    print("Incorrect Guess: " + guess_name + "    is actually  " + convolution.name)


# Printout of accuracy of the entire NN simulation
print("\n\n\n\n")
print("# of Images Guessed Correctly: " + str(len(correct)))
correct_percentage = len(correct) / (1.0 * len(image_names)) * 100

#correct_array.append(correct_percentage)

print("% Value of Images Correct: " + str(correct_percentage) + "%")




# Writing filter values to an outfile so that we can reload them at some point

#max_correct_indexes = [i for i, j in enumerate(a) if j == m]
#print("Highest Filter Correction Value: " + max(correct_array))

# outFile = open("filters.txt", "w")

# for i in max_correct_indexes:
#   filter = filter_array[i]
#   print("Filter Value: " + str(filter))
#   outFile.write(filter + "\n")


# outFile.close()

