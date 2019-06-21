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
print(image_names)

# Filter Array
filter = []

for fname in image_names:
  # First Encode and setup image for NN
  img = NNImage(fname)
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
  print(item_array[finalLayer(convolution.pmatrix, pool_array)].name)