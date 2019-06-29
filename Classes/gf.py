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