import os
import tensorflow as tf

class LoadDataset:
    """
    This class can be used to load images, preprocess them, and create a TensorFlow dataset.
    """
    def __init__(self, image_format, height, width):
        """
        Initialize the class.

        Args:
        - image_format (str): the image format (e.g. "jpeg", "png", "bmp").
        - height (int): height of image.
        - width (int): width of image.
        """
        self.image_format = image_format
        self.height = height
        self.width = width

    def list_dataset(self, dataset_path):
        """
        This function lists all image dataset and converts it into a tensor list.

        Args:
        - dataset_path (str): path to the dataset directory.

        Returns:
        - tf_dataset (Tensorflow dataset): Tensorflow dataset including the images.
        """
        # List image dataset
        image_list = os.listdir(dataset_path)
        image_ds_list = [dataset_path + "/" + i for i in image_list]

        # Sort the list
        image_ds_list = sorted(image_ds_list)

        # Converts the list into a tensor list
        tf_dataset = tf.data.Dataset.from_tensor_slices(image_ds_list)

        return tf_dataset

    def decode_image(self, images):
        """
        This function decodes images from a given format into a TensorFlow tensor.

        Args:
        - images (tensor): the dataset to decode.

        Returns:
        - images (tensor): the decoded dataset.
        
        Raises:
        - ValueError: if the image format is not recognized.
        """
        # Decode the image based on its format
        if self.image_format == "jpeg" or self.image_format == "jpg":
            images = tf.image.decode_jpeg(images, channels=3)
        elif self.image_format == "png":
            images = tf.image.decode_png(images, channels=3)
        elif self.image_format == "bmp":
            images = tf.image.decode_bmp(images, channels=3)
        elif self.image_format == "others":
            images = tf.image.decode_image(images, channels=3)
        else:
            raise ValueError("Unsupported image format: {}".format(self.image_format))
            
        return images
    
    def preprocessing(self, dataset_path):
        """
        This function preprocesses image dataset.

        Args:
        - dataset_path (tensor): path to the dataset directory.

        Returns:
        - images (tensor): preprocessed dataset.
        """
        # Preprocess image dataset
        images = tf.io.read_file(dataset_path)
        images = self.decode_image(images)
        images = tf.image.convert_image_dtype(images, dtype=tf.float32)
        images = tf.image.resize(images, size=(self.height, self.width), method="nearest")

        return images

    def __call__(self, dataset_path):
        """
        Calling this class applies the preprocessing function to each element in the dataset.

        Args:
        - dataset_path (str): path to the dataset directory.

        Returns:
        - dataset (Tensorflow dataset): Tensorflow dataset to use for training.
        """
        tf_dataset = self.list_dataset(dataset_path)
        
        dataset = tf_dataset.map(self.preprocessing)
                        
        return dataset