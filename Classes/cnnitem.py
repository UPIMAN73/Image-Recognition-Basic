class CNNItem:
    def __init__(self, name, imgMatrix, filter, ams=2):
        # Name of object
        self.name = name
        
        # Check to make sure that the image matrix is setup properly
        if (len(imgMatrix) > 0 and len(imgMatrix[0]) > 0):
            self.imgMatrix = imgMatrix
        else:
            self.imgMatrix = []
        
        # Check to make sure that the analysis matrix is smaller or equal to image matrix
        if (ams <= len(imgMatrix) and ams <= len(imgMatrix[0]) and ams > 0):
            self.cmatrix = self.convolutionLayer(imgMatrix, filter, ams)
        else:
            self.cmatrix = None
            print("Analysis Matrix Size is too large")
        
        # Pool Matrix
        self.pmatrix = None # This must be redefined for specific values
    

    # convolutional layers
    def convolutionLayer(self, matrix, filter, analyzing_size):
        #construct convolution matrix
        cmatrix = []
        anal_size = analyzing_size

        #2d layers for any nxn matrix which n > 1
        if (len(matrix) > 1):
            if (len(matrix[0]) > 1 and len(matrix[1]) > 1):

                # actual 2x2 matrix setup for scanning matrices
                for i in range(0, len(matrix) - (anal_size - 1)):
                    for j in range((anal_size - 1), len(matrix[i])):

                        # Setting up Layer array based on specs of the analysis matrix size
                        msamples = []
                        for k in range(0, anal_size):
                            for l in range(-anal_size + 1, 1):
                                msamples.append(matrix[i + k][j + l])

                        # layers calculations based on filters
                        msolution = 0
                        for k in range(0, len(msamples)):
                            msolution += msamples[k] * filter[k]
                        cmatrix.append(msolution)
        
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