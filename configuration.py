"""Image classification model and training config"""

class ModelConfig(object):
    """Wrapper class for model hyperparameters."""

    def __init__(self):
        """Sets the default model hyperparameters."""

        self.image_height = 32

        self.image_width = 32

        self.num_classes = 10

        self.image_channels = 3 # channels: R,G,B

        self.filter_size = 3 # try 3 

        self.conv1_num_filters = 32 # no. of conv filters in first conv module

        self.conv2_num_filters = 32 # no. of conv filters in second conv module

        self.conv3_num_filters = 128 # no. of conv filters in third conv module (optional)

        self.fc1_num_features = 1024 # dim of first FC layer

        self.fc2_num_features = 512 # dim of second FC layer (optional)

        self.learning_rate = 2e-5


class TrainingConfig(object):
    """Wrapper class for training hyperparameters."""

    def __init__(self):
        """Sets the default training hyperparameters."""
        
        self.epochs = 30 

        self.batch_size = 32

        self.keep_rate = 0.5 # keep rate for dropout

        self.vis_weights_every_epoch = 1





