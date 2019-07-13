from Classes.training import Train
from math import sqrt

# filter setup
filter = []
filter_size = 4     # The filter size will be the value of a square term (size^2 = filter_size)
threshold = 2       # Change this value to change the accuracy of the program in pooling layer
states = 2          # Number of states used for accuracy within convolution layer

for i in range(0, filter_size):
    filter.append(1)


# Filter and Training Stats name
filter_out_name = "filter_" + (str(int(math.sqrt(filter_size))) + "X" + str(int(math.sqrt(filter_size)))) + ".txt"
tstat_out_name = "planet_stats.json"


# Train Class Test
train = Train("./Images/letters", filter, states, threshold, negative_filter_encoding=True, 
tstat_fname=tstat_out_name, filter_out=filter_out_name, debug=False)

train.run()
train.writeOut()