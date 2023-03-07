import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from preprocessing.prediction_dataset import LoadDataset

def masking(pred_mask):
    """
    This function gets predicted mask from model output.

    Args:
    - pred_mask (tensor): model output.

    Returns:
    - pred_mask (tensor): the predicted mask to be visualized.
    """
    # Get the class with the highest probability
    pred_mask = tf.argmax(pred_mask, axis=-1)
    # Add channel dimension for display purposes
    pred_mask = pred_mask[..., tf.newaxis]

    return pred_mask[0]

def display(image_list):
    """
    This function displays images with given titles.

    Args:
    - image_list (list): the list of images to display.
    
    Returns:
    - None
    """
    # Titles for the images if there is no masks
    if len(image_list) == 2:
        title = ["Image", "Predicted Mask"]
        
    # Titles for the images if there is masks
    elif len(image_list) == 3:
        title = ["Image", "Mask", "Predicted Mask"]

    plt.figure(figsize=(18, 12))
    for i in range(len(image_list)):
        plt.subplot(1, len(image_list), i+1)
        plt.imshow(tf.keras.preprocessing.image.array_to_img(image_list[i]))
        if len(image_list) > 1:
            plt.title(title[i])
        plt.axis('off')

def change_colors(image, height, width):
    """
    This function changes predicted mask colors by UNet model according to classes.

    Args:
    - image (tensor): a predicted image in tensor format to be changed mask colors.
    - height (int): height of image.
    - width (int): width of image.
    
    Returns:
    - image (array): a colored mask image with RGB values for each class.
    """
    # Convert the image to numpy format and reshape it
    image = image.numpy()
    image = image.reshape(height, width, 1)

    # Concatenate the image with itself to obtain a 3-channel image
    image = np.concatenate((image, image, image), axis=2)

    # Apply the color map to each pixel in the image
    for x in range(height):
      for y in range(width):
        b, g, r = image[x, y]
        # for background
        if (b, g, r) == (0,0,0): 
          image[x, y] = (0,0,0)
        # for road asphalt
        elif (b, g, r) == (1,1,1): 
          image[x, y] = (85,85,255)
        # for road paved
        elif (b, g, r) == (2,2,2): 
          image[x, y] = (85,170,127)
        # for road unpaved
        elif (b, g, r) == (3,3,3): 
          image[x, y] = (255,170,127)
        # for road marking
        elif (b, g, r) == (4,4,4): 
          image[x, y] = (255,255,255) 
        # for speed bump
        elif (b, g, r) == (5,5,5): 
          image[x, y] = (255,85,255)
        # for cats eye
        elif (b, g, r) == (6,6,6): 
          image[x, y] = (255,255,127) 
        # for storm drain         
        elif (b, g, r) == (7,7,7): 
          image[x, y] = (170,0,127) 
        # for manhole cover
        elif (b, g, r) == (8,8,8): 
          image[x, y] = (0,255,255) 
        # for patchs
        elif (b, g, r) == (9,9,9): 
          image[x, y] = (0,0,127) 
        # for water puddle
        elif (b, g, r) == (10,10,10): 
          image[x, y] = (170,0,0)
        # for pothole
        elif (b, g, r) == (11,11,11): 
          image[x, y] = (255,0,0)
        # for cracks
        elif (b, g, r) == (12,12,12): 
          image[x, y] = (255,85,0)

    return image 

def preprocess_on_predict(model, image, height, width):
    """
    This function preprocesses the image for prediction using the specified model.
    
    Args:
    - model (Tensorflow Model object): model to use for predictions.
    - image (tensor): the input image with 3D.
    - height (int): height of image.
    - width (int): width of image.
    
    Returns:
    - pred (tensor): the masked image.
    """
    # Add batch dimension to image
    image = image[tf.newaxis, ...]
    # Predict the images
    pred = model.predict(image)
    # Create mask on the predicted image
    pred = masking(pred)
    # Change the masked image colors to specified colors
    pred = change_colors(pred, height, width)
    
    return pred

def load_sample_image(image_path, height, width):
    """
    This function loads sample image.
    
    Args:
    - image (str): image path to use for prediction .
    - height (int): height of image.
    - width (int): width of image.
    
    Returns:
    - image (tensor): the image in tensor format.
    """
    # Load image from the path
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(height, width))
    # Convert the image to array
    image = tf.keras.preprocessing.image.img_to_array(image) / 255.
    # Convert the array to tensor format
    image = tf.constant(image)
    return image

def show_prediction(model, image_path, masked_image_path, 
                    image_dataset_path, image_dataset_format, masked_dataset_path,  masked_dataset_format,
                    height, width, show_only_predicted):
    """
    This function generates, displays and saves model prediction.

    Args:
    - model (Tensorflow Model object): model to use for predictions.
    - image_path (str): path to a sample image to use for prediction (if no dataset is provided).
    - masked_image_path (str): path to a sample masked image to compare with prediction (if no dataset is provided).
    - image_dataset_path (str): path to image dataset to use for predictions (if no sample image is provided).
    - image_dataset_format (str): the image format in the image dataset.
    - masked_dataset_path (str): path to masked image dataset to compare with predictions (if no sample image is provided).
    - masked_dataset_format (str): the image format in the masked image dataset.
    - height (int): height of image.
    - width (int): width of image.
    - show_only_predicted (bool): Whether to show only predicted image by the model
    
    Returns:
    - None
    """

    # Create a file to save predicted images.
    if os.path.exists("./predicted_masks/") == False:
        os.makedirs("./predicted_masks/")

    # Check if dataset path specified
    if image_dataset_path is not None:
        # Initialize index number for the images
        idx = 0
        # Load and preprocess the image dataset
        load_image_dataset = LoadDataset(image_dataset_format, height, width)
        image_dataset = load_image_dataset(image_dataset_path)
        
        # Check if masked image dataset path specified
        if masked_dataset_path is None:
            # Display the images with predicted images
            for image in image_dataset:
                # Get image added dimension and predicted masked image by model
                pred = preprocess_on_predict(model, image, height, width)
                # Display and save the results
                if show_only_predicted:
                    display([pred])
                else:
                    display([image, pred])
                plt.savefig(f"./predicted_masks/predicted_mask{idx+1}.png")
                idx += 1
                plt.show()
                
        # Check if masked image dataset path is not specified    
        else:
            # Load and preprocess the masked image dataset
            load_mask_dataset = LoadDataset(masked_dataset_format, height, width)
            masked_dataset = load_mask_dataset(masked_dataset_path)
                
            # Display the images with predicted images
            for image, mask in zip(image_dataset, masked_dataset):
                # Get image added dimension and predicted masked image by model
                pred = preprocess_on_predict(model, image, height, width)
                # Display and save the results
                if show_only_predicted:
                    display([pred])
                else:
                    display([image, mask, pred])
                plt.savefig(f"./predicted_masks/predicted_mask{idx+1}.png")
                idx += 1
                plt.show()
        print("The masked images saved to './predicted_masks/' directory")
        
    # Check if sample image path specified
    else:
        # Load image
        image = load_sample_image(image_path, height, width)
        # Check if masked image path specified
        if masked_image_path is None:
            # Get image added dimension and predicted masked image by model
            pred = preprocess_on_predict(model, image, height, width)
            # Display and save the results
            if show_only_predicted:
                display([pred])
            else:
                display([image, pred])
            plt.savefig("./predicted_masks/predicted_mask.png")
            plt.show()
            
        # Check if masked image path is not specified
        else: 
            # Load masked image
            mask = load_sample_image(masked_image_path, height, width)
            # Get image added dimension and predicted masked image by model
            pred = preprocess_on_predict(model, image, height, width)
            # Display and save the results
            if show_only_predicted:
                display([pred])
            else:
                display([image, mask, pred])
            plt.savefig("./predicted_masks/predicted_mask.png")
            plt.show()       
        print("The masked image saved to './predicted_masks/' directory")