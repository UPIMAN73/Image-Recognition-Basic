from Classes.nnimage import NNImage
from Classes.cnnitem import CNNItem

fname = "./Images/backslash_5x5.png"

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


# Image Encoding Test
img1 = NNImage("./Images/backslash_5x5.png")
img1_matrix = img1.encodedMatrix
print(img1_matrix)


# TODO neural network setup


