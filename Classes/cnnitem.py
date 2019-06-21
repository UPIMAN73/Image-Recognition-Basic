class CNNItem:
    def __init__(self, name, imgMatrix, filter):
        self.name = name
        self.imgMatrix = imgMatrix
        self.cmatrix = self.convolutionLayer(imgMatrix, filter)
        self.pmatrix = None # This must be redefined for specific values
    

    # convolutional layers
    def convolutionLayer(self, matrix, filter):
        #construct convolution matrix
        cmatrix = []

        #2d layers for any nxn matrix which n > 1
        if (len(matrix) > 1):
            if (len(matrix[0]) > 1 and len(matrix[1]) > 1):

                # actual 2x2 matrix setup for scanning matrices
                for i in range(0, len(matrix) - 1):
                    for j in range(1, len(matrix[i])):

                        # layers calculations based on filters
                        msamples = [matrix[i][j - 1], matrix[i][j], matrix[i + 1][j - 1], matrix[i + 1][j]]
                        msolution = 0
                        for k in range(0, len(msamples)):
                            msolution += msamples[k] * filter[k]
                        m.append(msolution)
        
        # return the convoluted matrix
        return cmatrix


    # Pool layer is the layer that sets up the convoluted matrix to finalized processes
    def poolLayer(self, cmatrix, v1, v2):
        # two matrix values because comparing two values
        enc1 = []
        enc2 = []

        # First comparison values
        for i in range(0, len(cmatrix)):
            if cmatrix[i] >= v1:
                enc1.append(1)
            else:
                enc1.append(-1)
        
        # Second comparison values
        for i in range(0, len(cmatrix)):
            if cmatrix[i] <= v2:
                enc2.append(1)
            else:
                enc2.append(-1)
        
        # Pool matrix
        return [enc1, enc2]