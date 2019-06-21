# possibly import cnn

fname = "backslash_5x5.png"
imfile = open(fname, "rb")

def zeroArray(arry):
    res = []
    for i in range(0, len(arry)):
        res.append(0)
    return res

# Encoding values for color black
def blackScalePerc(r, g, b):
    res = None
    rgb_c = [r/255, g/255, b/255]
    res = (rgb_c[0] + rgb_c[1] + rgb_c[2])/3
    return res

# Fully encoded values for black scaling
def blackScale(score):
    return (-2 * score) + 1

def loadImage(img):
    imgMatrix = []
    imgString = img.read()
    print(imgString)
    imgMatrix = imgString.split("\n")
    print("\n\n\n")
    print(imgMatrix)
    return imgMatrix


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



img1_matrix = loadImage(imfile)

# TODO neural network setup


