import json
from addict import Dict #addict allows for cool nested dict handling

# Step 2: Define some constants.
parameters = Dict()     # Parameters are a nested dictionary (addict library)

# The MNIST dataset has 10 classes, representing the digits 0 through 9.
parameters.dataset.num_classes = 10

# The MNIST images are always 28x28 pixels.
parameters.dataset.image_size = 28
parameters.dataset.image_pixels = parameters.dataset.image_size * parameters.dataset.image_size

# Batch size. Must be evenly dividable by dataset sizes.
parameters.run.batch_size = 100
parameters.run.eval_batch_size = 1

# CNN - number of output channels per layer
parameters.graph.num_channels_1 = 4
parameters.graph.num_channels_2 = 8
parameters.graph.num_channels_3 = 12

# Convolutional Layers- filter sizes
parameters.graph.filter_1_units = 5 # 5x5
parameters.graph.filter_2_units = 5 # 5x5
parameters.graph.filter_3_units = 4 # 4x4

# FULLY CONNECTED LAYER
parameters.graph.fully_connected_units = 200

# SOFTMAX OUTPUT LAYER
# parameters.dataset.num_classes = 10 - defined above

# Maximum number of training steps.
parameters.run.max_steps = 20000

# Directory to put the training data.
parameters.run.train_dir = "/home/pantelis/Projects/tensorflow-intro/mnist_cnn"

def sample_function(max_steps):
    print(max_steps)

sample_function(**parameters.run)

with open('data.json', 'w') as fp:
    json.dump(data, fp)