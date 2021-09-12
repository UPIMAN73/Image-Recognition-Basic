from Classes.nnimage import NNImage
import matplotlib.pyplot as plot
from os.path import isfile, join
from os import listdir
from PIL import Image
import numpy

# Variables for image settings
img_dir = "./Images/letters"
dst_dir = "./Show-Images"
img_names = [f for f in listdir(img_dir) if ( (isfile(join(img_dir, f)) and ((".png" in join(img_dir, f)) or (".jpg" in join(img_dir, f)) or (".jpg" in join(img_dir, f)) ) )) ]
nn_img_arry = []
rgb_img_arry = []
matplot_lib_show = False

# Conversion and Writing to image file
for fname in img_names:    
    img = NNImage(join(img_dir, fname))
    img.encodedMatrix = img.blackEncoding()() # best use black encoding
    nn_img_arry.append(img)
    Image.fromarray(img.imgMatrix, "RGB").save(join(img_dir, join(dst_dir, str(fname.split(".")[0]) + "_black_rgb.jpg")))
    print("Saved " + fname)

# Display Images if matplotlib settings is enabled
if matplot_lib_show:
    for img in rgb_img_arry:
        plot.imshow(img)

