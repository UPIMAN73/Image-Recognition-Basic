# Local Class Imports
from Classes.nnimage import NNImage
from Classes.cnnitem import CNNItem
from Classes.traininfo import TrainInfo

# System Imports
from os import listdir
from os.path import isfile, join
from math import sqrt, factorial
from random import random, randint
from datetime import datetime
import json

############################
#      Training Class      #
############################

"""
This training class is used to train an image neural network classifyer (CNN - [Convoluted Neural Network])
so that the computer can recognize and classify an object based on an image.
"""
class Train:
    def __init__(self, img_dir, filter, states, threshold, tstats_dir="./Training", tstat_fname="training_stats.json", filter_out="filters.txt", obj_file="object_pool.json", debug=False, negative_filter_encoding=False):
        # Training Stats output names and object files
        self.tstats_dir = tstats_dir
        self.tstat_fname = tstat_fname
        self.filter_out_name = filter_out
        self.obj_file_name = obj_file
        self.debug = debug

        # Image content variables
        self.img_dir = img_dir
        self.img_names = [f for f in listdir(img_dir) if ( (isfile(join(img_dir, f)) and ((".png" in join(img_dir, f)) or (".jpg" in join(img_dir, f)) or (".jpg" in join(img_dir, f)) ) )) ]
        self.img_arry = []
        self.img_matrix = []

        # Encode and setup image for NN
        for fname in self.img_names:    
            img = NNImage(join(self.img_dir, fname))
            img.encodedMatrix = img.blackEncoding()() # best use black encoding
            self.img_matrix.append(img.encodedMatrix)
            self.img_arry.append(img)

        # NN Arrays for calculating and applying NN
        self.item_array = []
        self.conv_array = []
        self.pool_array = []

        # Train Stats
        self.train_stats = []
        for i in self.img_names:
            self.train_stats.append(TrainInfo(i.split("_")[0].capitalize()))
        

        # number variables
        self.num_of_states = states    # Highly important if you want to setup the correct process for your NN
        self.threshold = threshold            # Important for pool layer

        # Filter Setup
        ranfilter = []
        #[1, 1, 1, 1, 1, 1, 1, 1, 1]
        self.filter_array = []

        # Either use factorial or power rule
        self.desired_filter_length = pow(len(filter), self.num_of_states) #factorial(len(filter)) * num_of_states

        # setting up preset arrays
        while len(self.filter_array) != self.desired_filter_length:
            ranfilter = []
            
            # more than 1 state filter
            if negative_filter_encoding:
                for i in range(0, len(filter)):
                    ranum = randint(int(-states/2), int(states/2))
                    if ranum > 0 or ranum < 0:
                        ranfilter.append(ranum)
                    else:
                        ranfilter.append(0)
            
            # 0-state possibility filter
            elif states > 2:
                for i in range(0, len(filter)):
                    ranum = randint(0, states)
                    if ranum > 0:
                        ranfilter.append(ranum)
                    else:
                        ranfilter.append(0)
            
            # 2 state filter
            elif states == 2:
                for i in range(0, len(filter)):
                    ranum = randint(0, 1)
                    if ranum == 1:
                        ranfilter.append(1)
                    else:
                        ranfilter.append(0)
            if ranfilter not in self.filter_array:
                self.filter_array.append(ranfilter)
        print("Done setting up filter arrays")
        print("Length of Filter Array is:   %d" % len(self.filter_array))

        # Analyzing matrix size
        self.amsize = int(sqrt(len(filter)))

        # Accuracy lists
        self.correct = []
        self.correct_array = []
        self.correct_perc_array = []
    
    # Get the train info object by name
    def getTrainInfo(self, name):
        for i in self.train_stats:
            if i.name == name:
                return i
        return None

    # Matrix and Array based calculations and declarations
    def zeroArray(self, arry):
        res = []
        for i in range(0, len(arry)):
            res.append(0 * i)
        return res

    # Matrix Sum
    def matrixSum(self, matrix):
        res = 0
        for i in matrix:
            res += i
        return res


    # Final Layer for Neural Networks
    def finalLayer(self, pmatrix, pools):
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
            ansArray.append(self.matrixSum(ansA))

        # return the index of the network items
        return ansArray.index(max(ansArray))


    """ Neural Network Training Function of the Program """
    # Run the Training Loop for the Neural Network to Learn
    def run(self):
        for filter in self.filter_array:
            # file name for loop 
            correct = []
            for fname in self.img_names:
                img_found = False
                for i in self.img_arry:
                    if img_found == False:
                        # Change this split character for different OS
                        if i.name.split("\\")[1] == fname:
                            img = i
                            img_found = True
                        else:
                            continue
                    else:
                        break

                if img_found == False:
                    break
                
                # Then Setup and calculate for NN
                # Convolution Layer
                convolution = CNNItem(fname.split("_")[0].capitalize(), img.encodedMatrix, filter, ams=self.amsize)
                self.conv_array.append(convolution.cmatrix)

                # Pooling Layer
                convolution.pmatrix = convolution.poolLayer(convolution.cmatrix, self.threshold, -self.threshold)
                self.pool_array.append(convolution.pmatrix)

                # Item Layer
                self.item_array.append(convolution)

                # Final Layer
                guess_name = self.item_array[self.finalLayer(convolution.pmatrix, self.pool_array)].name

                # Correction appending and printouts
                if (guess_name == convolution.name):
                    if self.debug:
                        print("Correct Guess: " + guess_name)
                    correct.append(convolution)
                    for i in self.train_stats:
                        if i.name == guess_name:
                            i.filters.append(filter)
                            
                            # Adding pool to training info
                            tinfo = self.getTrainInfo(convolution.name)
                            if tinfo != None:
                                tinfo.pool.append(convolution.pmatrix)
                                tinfo.convolution.append(convolution.cmatrix)
                else:
                    if self.debug:
                        print("Incorrect Guess: " + guess_name + "    is actually  " + convolution.name)

            # Printout of accuracy of the entire NN simulation
            if self.debug:
                print("\n\n\n\n")
                print("# of Images Guessed Correctly: " + str(len(correct)))
            
            # Calculate percenntage
            self.correct_percentage = len(correct) / (1.0 * len(self.img_names)) * 100
            self.correct_perc_array.append(self.correct_percentage)
            self.correct_array.append(len(correct))
            #print("% Value of Images Correct: " + str(self.correct_percentage) + "%")
    

    # Write out all of the information to a bunch of files within a specific directory
    def writeOut(self):
        print("Writing out Training Results")

        # Writing filter values to an outfile so that we can reload them at some point
        self.correct_array = sorted(self.correct_array)
        
        if self.debug:
            print(self.correct_perc_array)
        
        # max setting up filter arrangement for best filters
        self.max_correct_indexes = [i for i, x in enumerate(self.correct_array) if x == max(self.correct_array)]
        self.max_filters = []
        for i in sorted(self.max_correct_indexes, reverse=True):
            if self.debug:
                print("Highest Filter Correction Value: " + str(self.filter_array[i]))
                print("Highest Correction Percentage Value: " + str(self.correct_perc_array[i]))
                print("Highest Correction Value: " + str(self.correct_array[i]))
            self.max_filters.append(self.filter_array[i])

        # Write out filters to a filter file
        outFile = open(join(self.tstats_dir, self.filter_out_name), "w")

        # Write out best filters
        for i in self.max_correct_indexes:
            filter = self.filter_array[i]
            if self.debug:
                print("Filter Value: " + str(filter))
            outFile.write(str(filter) + "\n")

        # Write out the rest of the filters
        for i in range(0, len(self.filter_array)):
            if i in self.max_correct_indexes:
                continue
            else:
                outFile.write(str(self.filter_array[i]) + "\n")

        # Close file
        outFile.close()

        # write out training stats to a JSON file
        json_file = open(join(self.tstats_dir, self.tstat_fname), "w")
        obj_setup = {"date" : str(datetime.now()), "objects" : []}

        # update object setup for training stats to a dictionary to convert to JSON
        for i in self.train_stats:
            obj_setup["objects"].append({"name" : i.name, "filters" : i.filters, "pool" : i.pool, "convolutions" : i.convolution})

        # write out information to a JSON file and clsoe it
        json_file.write(str(str(json.dumps(obj_setup)).encode("ascii").decode("ascii")))
        json_file.close()