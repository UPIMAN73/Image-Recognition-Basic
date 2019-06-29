from Classes.training import Train

# filter setup
filter_size = 2 # [size x size] == size^2 array
filter = []
threshold = 1
states = 3

for i in range(0, filter_size * filter_size):
    filter.append(1)

# Train Class Test
train = Train("./Images/Test", filter, states, threshold, negative_filter_encoding=False)
train.run()
print(train.item_array[1].imgMatrix)
train.writeOut()