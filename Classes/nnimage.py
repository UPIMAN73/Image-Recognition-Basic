from PIL import Image
import numpy

class NNImage:
    def __init__(self, name):
        self.name = name
        self.img = Image.open(name)
        self.imgMatrix = numpy.array(self.img, dtype=int)
        self.encodedMatrix = self.blackEncoding() # black encoding matrix
    
    # Encoding RGB values to Percentage values for the color black
    def blackScalePerc(self, r, g, b):
        res = None
        rgb_c = [r/255, g/255, b/255]
        res = (rgb_c[0] + rgb_c[1] + rgb_c[2])/3 # take average of all 3 values
        return res

    # Fully encoded values for black scaling
    def blackScale(self, score):
        return (-2 * score) + 1  # linear equation for defining blackscale

    
    # Black encoding matrix of the image matrix
    # Best used on black and white images
    def blackEncoding(self):
        result = []
        
        # Iterate through image matrix based on width
        for i in self.imgMatrix:
            
            # row matrix is used to gather all of the values used for the encoding process
            row_matrix = []

            # Actual calculation for encoding values
            for j in range(0, len(i)):
                red = i[j][0]
                green = i[j][1]
                blue = i[j][2]
                row_matrix.append(self.blackScale(self.blackScalePerc(red, green, blue)))
            
            # appending the calculated row matrix to the result matrix
            result.append(row_matrix)
        return result
