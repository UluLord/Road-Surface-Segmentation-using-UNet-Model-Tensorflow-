import os
import tensorflow as tf

class LoadDataset:
    """
    This class can be used to load, preprocess, and create a TensorFlow dataset of images and masks.
    """
    def __init__(self, image_dataset_format, mask_dataset_format, height, width, batch_size, buffer_size):
        """
        Initialize the class.

        Args:
        - image_dataset_format (str): the image format in the image dataset (e.g. "jpeg", "png", "bmp").
        - mask_dataset_format (str): the image format in the masked image dataset (e.g. "jpeg", "png", "bmp").
        - height (int): height of image.
        - width (int): width of image.
        - batch_size (int): the batch size for dividing the dataset into chunks.
        - buffer_size (int): the buffer size for shuffling the dataset.
        """
        self.image_dataset_format = image_dataset_format
        self.mask_dataset_format = mask_dataset_format
        self.height = height
        self.width = width
        self.batch_size = batch_size
        self.buffer_size = buffer_size

    def list_dataset(self, image_dataset_path, mask_dataset_path):
        """
        This function lists all image and mask datasets and converts them into a tensor list.

        Args:
        - image_dataset_path (str): path to the image dataset directory.
        - mask_dataset_path (str): path to the mask dataset directory.

        Returns:
        - tf_dataset (Tensorflow dataset): Tensorflow dataset including the image and mask images.
        """
        # List image dataset
        image_list = os.listdir(image_dataset_path)
        image_ds_list = [image_dataset_path + "/" + i for i in image_list]

        # List mask dataset
        mask_list = os.listdir(mask_dataset_path)
        mask_ds_list = [mask_dataset_path + "/" + i for i in mask_list]

        # Sort the lists
        image_ds_list = sorted(image_ds_list)
        mask_ds_list = sorted(mask_ds_list)

        # Converts the lists into a tensor list
        tf_dataset = tf.data.Dataset.from_tensor_slices((image_ds_list, mask_ds_list))

        return tf_dataset
    
    @staticmethod
    def decode_image(images, image_format, channels):
        """
        This function decodes images from a given format into a TensorFlow tensor.

        Args:
        - images (tensor): the dataset to decode.
        - image_format (str): the image format (e.g. "jpeg", "png", "bmp").
        - channels (int): the number of color channels in the output tensor.

        Returns:
        - images (tensor): the decoded image.
        
        Raises:
        - ValueError: if the format is not recognized.
        """
        # Decode the images based on its format
        if image_format == "jpeg" or format == "jpg":
            images = tf.image.decode_jpeg(images, channels=channels)
        elif image_format == "png":
            images = tf.image.decode_png(images, channels=channels)
        elif image_format == "bmp":
            images = tf.image.decode_bmp(images, channels=channels)
        elif image_format == "others":
            images = tf.image.decode_image(images, channels=channels)
        else:
            raise ValueError("Unsupported image format: {}".format(image_format))
        return images

    def preprocessing(self, image_dataset_path, mask_dataset_path):
        """
        This function preprocesses image and mask datasets.

        Args:
        - image_dataset_path (tensor): path to the image dataset.
        - mask_dataset_path (tensor): path to the mask dataset.

        Returns:
        - images (tensor): preprocessed image dataset.
        - masks (tensor): preprocessed mask dataset.
        """
        # Preprocess image dataset
        images = tf.io.read_file(image_dataset_path)
        images = self.decode_image(images, self.image_dataset_format, 3)
        images = tf.image.convert_image_dtype(images, dtype=tf.float32)
        images = tf.image.resize(images, size=(self.height, self.width), method="nearest")

        # Preprocess mask dataset
        masks = tf.io.read_file(mask_dataset_path)
        masks = self.decode_image(masks, self.mask_dataset_format, 1)
        masks = tf.image.resize(masks, size=(self.height, self.width), method="nearest")
        return images, masks

    def train_val_split(self, dataset, validation_split):
        """
        This function splits a dataset into training and validation sets.

        Args:
        - dataset (Tensorflow dataset): Tensorflow dataset.
        - validation_split (float): the fraction of the dataset to use for validation.

        Returns:
        - train_dataset (Tensorflow dataset): Tensorflow dataset to use for training model.
        - validation_dataset (Tensorflow dataset): Tensorflow dataset to use for validating model performance.
        """
        # Get the length of the dataset
        len_dataset = len(dataset)

        # Calculate the size of the training set
        train_size = int((1. - validation_split) * len_dataset)

        # Split the dataset into training and validation sets
        train_dataset = dataset.take(train_size)
        validation_dataset = dataset.skip(train_size)

        # Print the number of elements in each dataset
        print("Number of elements in train dataset:", len(list(train_dataset)))
        print("Number of elements in validation dataset:", len(list(validation_dataset)))

        return train_dataset, validation_dataset

    def __call__(self, image_dataset_path, mask_dataset_path, validation_split):
        """
        Calling this class applies the preprocessing function to each element in the datasets, 
        shuffles the datasets, batches the datasets, and prefetches the data.

        Args:
        - image_dataset_path (str): path to the image dataset directory.
        - mask_dataset_path (str): path to the mask dataset directory.
        - validation_split (float): the fraction of the dataset to use for validation.

        Returns:
        - train_dataset (Tensorflow dataset): Tensorflow dataset to use for training model.
        - validation_dataset (Tensorflow dataset): Tensorflow dataset to use for validating model performance.
        """
        tf_dataset = self.list_dataset(image_dataset_path, mask_dataset_path)
        
        dataset = tf_dataset.map(self.preprocessing)
        
        dataset = dataset.shuffle(buffer_size=self.buffer_size)
        
        dataset = dataset.batch(batch_size=self.batch_size, drop_remainder=False)
        
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        train_dataset, validation_dataset = self.train_val_split(dataset, validation_split)
        
        return train_dataset, validation_dataset