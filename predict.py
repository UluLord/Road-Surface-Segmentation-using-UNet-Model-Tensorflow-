import argparse
import tensorflow as tf
from preprocessing.prediction_display import show_prediction

def set_arguments():
    """
    This function parses command line arguments and returns them as a dictionary. 
    """ 

    # Initialize ArgumentParser
    parser = argparse.ArgumentParser(
        # Description of the project
        description="This project implements a style transfer algorithm for generating new images by transferring the style of one image to the content of another image. \n\nTo generate new image, adjust the parameters if necessary:",
        # Usage string to display
        usage="Generating new images by transfering style",
        # Set the formatter class to ArgumentDefaultsHelpFormatter
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        # Set prefix chars
        prefix_chars="-",
        # Set default value for argument_default
        argument_default=argparse.SUPPRESS,
        # Allow abbreviations of options
        allow_abbrev=True,
        # Add help argument
        add_help=True)

    #Add arguments
    parser.add_argument("--pretrained_model_path", default="./pretrained_model", type=str, required=False, 
                        help="Path to a pretrained model to use for predictions")
    parser.add_argument("--image_path", default=None, type=str, required=False, 
                        help="Path to a sample image to use for prediction. Required if no image dataset is provided")
    parser.add_argument("--masked_image_path", default=None, type=str, required=False, 
                        help="Path to a sample masked image path to compare with prediction (if no dataset is provided)")
    parser.add_argument("--image_dataset_path", default=None, type=str, required=False,
                        help="Path to an image dataset to use for predictions. Required if no sample image is provided")
    parser.add_argument("--image_dataset_format", default="png", type=str, required=False, choices=["jpeg", "jpg", "png", "bmp", "others"],
                      help="The image format in the image dataset (choose one of those: jpeg, jpg, png, bmp or others)")
    parser.add_argument("--masked_dataset_path", default=None, type=str, required=False,
                        help="Path to a masked image dataset path to compare with predictions (if no sample image is provided)")
    parser.add_argument("--masked_dataset_format", default="png", type=str, required=False, choices=["jpeg", "jpg", "png", "bmp", "others"],
                      help="The image format in the masked image dataset (choose one of those: jpeg, jpg, png, bmp or others)")
    parser.add_argument("--height", default=256, type=int, required=False, 
                        help="Height of image")
    parser.add_argument("--width", default=256, type=int, required=False, 
                        help="Width of image")
    parser.add_argument("--show_only_predicted", default=True, type=bool, required=False,
                        help="Whether to show only predicted image by the model")

    # Parse the arguments and convert them to a dictionary
    args = vars(parser.parse_args())

    return args

if __name__ == "__main__":

    # Set arguments
    args = set_arguments()

    # Load a pretained model
    model = tf.keras.models.load_model(args["pretrained_model_path"])

    # Predict an image or dataset
    show_prediction(model, args["image_path"], args["masked_image_path"], 
                    None, args["image_dataset_format"], None, args["masked_dataset_format"], 
                    args["height"], args["width"], args["show_only_predicted"])
    
    # Predict an image or dataset
    #show_prediction(model, args["image_path"], args["masked_image_path"], 
    #                args["image_dataset_path"], args["image_dataset_format"], args["masked_dataset_path"], args["masked_dataset_format"], 
    #                args["height"], args["width"], args["show_only_predicted"])