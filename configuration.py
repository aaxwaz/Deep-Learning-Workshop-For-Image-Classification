"""Image classification model and training config"""

class ModelConfig(object):
    """Wrapper class for model hyperparameters."""

    def __init__(self):
        """Sets the default model hyperparameters."""

        self.image_height = 28

        self.image_width = 28

        self.num_classes = 10

        self.image_channels = 3

        self.filter_size = 5

        self.conv1_num_filters = 32

        self.conv2_num_filters = 64

        self.fc1_num_features = 1024 

        self.learning_rate = 1e-4

        self.optimizer = 'Adam'


class TrainingConfig(object):
    """Wrapper class for training hyperparameters."""

    def __init__(self):
        """Sets the default training hyperparameters."""
        
        self.epochs = 10





