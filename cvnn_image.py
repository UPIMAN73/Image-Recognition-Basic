
fname = "backslash_5x5.png"
imfile = open(fname, "rb")


def loadImage(img):
    imgMatrix = []
    imgString = img.read()
    print(imgString)
    imgMatrix = imgString.split("\n")
    print("\n\n\n")
    print(imgMatrix)

loadImage(imfile)