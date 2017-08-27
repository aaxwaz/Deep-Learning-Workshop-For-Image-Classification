import numpy as np
import pickle
import os
import sys
import urllib.request
import tarfile
import zipfile

class dataset():

    def __init__(self, data_path, data_url="https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz", img_size=32, num_channels=3, 
                 num_classes=10, num_files_train=5, images_per_file=10000):
        """
        :param data_path: Director to download the data; To create new one if not exists
        :param data_url: URL for cifar 10 data set on the internet 
        :param img_size: Width and height of each image after resize
        :param num_channels: Number of channels in each image, 3 channels: Red, Green, Blue
        :param num_classes: Number of classes
        :param num_files_train: Number of files for the training-set
        :param images_per_file: Number of images for each batch-file in the training-set
        :param num_images_train: Total number of images in the training-set
        """

        self.data_path = data_path
        self.data_url = data_url 
        self.img_size = img_size 
        self.num_channels = num_channels 
        self.num_classes = num_classes 
        self.num_files_train = num_files_train 
        self.images_per_file = images_per_file 
        self.num_images_train = num_files_train * images_per_file 

    def _get_file_path(self, filename=""):
        """
        Return the full path of a data-file for the data-set.
        If filename=="" then return the directory of the files.
        """

        return os.path.join(self.data_path, "cifar-10-batches-py/", filename)

    def _unpickle(self, filename):
        """
        Unpickle the given file and return the data.
        Note that the appropriate dir-name is prepended the filename.
        """

        # Create full path for the file.
        file_path = self._get_file_path(filename)

        print("Loading data: " + file_path)

        with open(file_path, mode='rb') as file:
            # In Python 3.X it is important to set the encoding,
            # otherwise an exception is raised here.
            data = pickle.load(file, encoding='bytes')

        return data

    def _convert_images(self, raw):
        """
        Convert images from the CIFAR-10 format and
        return a 4-dim array with shape: [image_number, height, width, channel]
        where the pixels are floats between 0.0 and 1.0.
        """

        # Convert the raw images from the data-files to floating-points.
        raw_float = np.array(raw, dtype=float) / 255.0

        # Reshape the array to 4-dimensions.
        images = raw_float.reshape([-1, self.num_channels, self.img_size, self.img_size])

        # Reorder the indices of the array.
        images = images.transpose([0, 2, 3, 1])

        return images


    def _load_data(self, filename):
        """
        Load a pickled data-file from the CIFAR-10 data-set
        and return the converted images (see above) and the class-number
        for each image.
        """

        # Load the pickled data-file.
        data = self._unpickle(filename)

        # Get the raw images.
        raw_images = data[b'data']

        # Get the class-numbers for each image. Convert to numpy-array.
        cls = np.array(data[b'labels'])

        # Convert the images.
        images = self._convert_images(raw_images)

        return images, cls

    ########################################################################
    # Public functions that you may call to download the data-set from
    # the internet and load the data into memory.
    def load_class_names(self):
        """
        Load the names for the classes in the CIFAR-10 data-set.
        Returns a list with the names. Example: names[3] is the name
        associated with class-number 3.
        """

        # Load the class-names from the pickled file.
        raw = self._unpickle(filename="batches.meta")[b'label_names']

        # Convert from binary strings.
        names = [x.decode('utf-8') for x in raw]

        return names

    def load_training_data(self):
        """
        Load all the training-data for the CIFAR-10 data-set.
        The data-set is split into 5 data-files which are merged here.
        Returns the images, class-numbers and one-hot encoded class-labels.
        """

        # Pre-allocate the arrays for the images and class-numbers for efficiency.
        images = np.zeros(shape=[self.num_images_train, self.img_size, self.img_size, self.num_channels], dtype=float)
        cls = np.zeros(shape=[self.num_images_train], dtype=int)

        # Begin-index for the current batch.
        begin = 0

        # For each data-file.
        for i in range(self.num_files_train):
            # Load the images and class-numbers from the data-file.
            images_batch, cls_batch = self._load_data(filename="data_batch_" + str(i + 1))

            # Number of images in this batch.
            num_images = len(images_batch)

            # End-index for the current batch.
            end = begin + num_images

            # Store the images into the array.
            images[begin:end, :] = images_batch

            # Store the class-numbers into the array.
            cls[begin:end] = cls_batch

            # The begin-index for the next batch is the current end-index.
            begin = end

        return images, cls, self.one_hot_encoded(class_numbers=cls, num_classes=self.num_classes)

    def load_test_data(self):
        """
        Load all the test-data for the CIFAR-10 data-set.
        Returns the images, class-numbers and one-hot encoded class-labels.
        """

        images, cls = self._load_data(filename="test_batch")

        return images, cls, self.one_hot_encoded(class_numbers=cls, num_classes=self.num_classes)

    def one_hot_encoded(self, class_numbers, num_classes=None):
        """
        Generate the One-Hot encoded class-labels from an array of integers.
        For example, if class_number=2 and num_classes=4 then
        the one-hot encoded label is the float array: [0. 0. 1. 0.]
        :param class_numbers:
            Array of integers with class-numbers.
            Assume the integers are from zero to num_classes-1 inclusive.
        :param num_classes:
            Number of classes. If None then use max(class_numbers)+1.
        :return:
            2-dim array of shape: [len(class_numbers), num_classes]
        """

        # Find the number of classes if None is provided.
        # Assumes the lowest class-number is zero.
        if num_classes is None:
            num_classes = np.max(class_numbers) + 1

        return np.eye(num_classes, dtype=float)[class_numbers]

    def maybe_download_and_extract(self):
        """
        Download and extract the data if it doesn't already exist.
        Assumes the self.data_url is a tar-ball file.

        :return:
            Nothing.
        """

        # Filename for saving the file downloaded from the internet.
        # Use the filename from the URL and add it to the self.data_path.
        filename = self.data_url.split('/')[-1]
        file_path = os.path.join(self.data_path, filename)

        # Check if the file already exists.
        # If it exists then we assume it has also been extracted,
        # otherwise we need to download and extract it now.
        if not os.path.exists(file_path):
            # Check if the download directory exists, otherwise create it.
            if not os.path.exists(self.data_path):
                os.makedirs(self.data_path)

            # Download the file from the internet.
            file_path, _ = urllib.request.urlretrieve(url=self.data_url,
                                                      filename=file_path,
                                                      reporthook=self._print_download_progress)

            print()
            print("Download finished. Extracting files.")

            if file_path.endswith(".zip"):
                # Unpack the zip-file.
                zipfile.ZipFile(file=file_path, mode="r").extractall(self.data_path)
            elif file_path.endswith((".tar.gz", ".tgz")):
                # Unpack the tar-ball.
                tarfile.open(name=file_path, mode="r:gz").extractall(self.data_path)

            print("Done.")
        else:
            print("Data has apparently already been downloaded and unpacked.")


    def _print_download_progress(self, count, block_size, total_size):
        """
        Function used for printing the download progress.
        Used as a call-back function in maybe_download_and_extract().
        """

        # Percentage completion.
        pct_complete = float(count * block_size) / total_size

        # Status-message. Note the \r which means the line should overwrite itself.
        msg = "\r- Download progress: {0:.1%}".format(pct_complete)

        # Print it.
        sys.stdout.write(msg)
        sys.stdout.flush()

