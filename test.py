from Classes.training import Train

# filter setup
filter_size = 2 # [size x size] == size^2 array
filter = []
threshold = 2 # Change this value to change the accuracy of the program
states = 2

for i in range(0, filter_size * filter_size):
    filter.append(1)

# Train Class Test
train = Train("./Images/Test", filter, states, threshold, negative_filter_encoding=True, debug=False)
train.run()
train.writeOut()