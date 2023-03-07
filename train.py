import argparse
from preprocessing.train_dataset import LoadDataset
from model import UNet

def set_arguments():
  """
  This function parses command line arguments and returns them as a dictionary. 
  """ 

  # Initialize ArgumentParser
  parser = argparse.ArgumentParser(
      # Description of the project
      description="This project implements UNet model for image segmentation. \n\nTo train the model or create segmentation on new images, adjust the parameters if necessary:",
      # Usage string to display
      usage="UNet for image segmentation",
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
  parser.add_argument("--image_dataset_path", type=str, required=True, 
                      help="Path to the image dataset")
  parser.add_argument("--image_dataset_format", default="png", type=str, required=False, choices=["jpeg", "jpg", "png", "bmp", "others"],
                      help="The image format in the image dataset (choose one of those: jpeg, jpg, png, bmp or others)")
  parser.add_argument("--mask_dataset_path", type=str, required=True, 
                      help="Path to the mask dataset")
  parser.add_argument("--mask_dataset_format", default="png", type=str, required=False, choices=["jpeg", "jpg", "png", "bmp", "others"],
                      help="The image format in the masked image dataset (choose one of those: jpeg, jpg, png, bmp or others)")
  parser.add_argument("--height", default=256, type=int, required=False, 
                        help="Height of image")
  parser.add_argument("--width", default=256, type=int, required=False, 
                        help="Width of image")
  parser.add_argument("--nb_classes", type=int, required=True,
                      help="Number of classes in the dataset")
  parser.add_argument("--epochs", default=100, type=int, required=False,
                      help="Number of training epochs")
  parser.add_argument("--batch_size", default=32, type=int, required=False, 
                        help="Batch size for dividing the dataset into chunks")
  parser.add_argument("--buffer_size", default=1000, type=int, required=False, 
                        help="Buffer size for shuffling the dataset")
  parser.add_argument("--validation_split", default=0., type=float, required=False,
                        help="Fraction of the dataset to use for validation")
  parser.add_argument("--nb_filters", default=32, type=int, required=False, 
                      help="Number of filters in the first encoder block")
  parser.add_argument("--kernel_size", default=3, type=int, required=False,
                      help="Kernel size in the layers")
  parser.add_argument("--activation", default="relu", type=str, required=False,
                      help="Activation function to use in the layers")
  parser.add_argument("--kernel_initializer", default="he_uniform", type=str, required=False,
                      help="Method to initialize the kernel")
  parser.add_argument("--optimizer", default="adam", type=str, required=False, choices=["sgd", "rmsprop", "adam"], 
                      help="Model optimizer type (choose one of those: sgd, rmsprop or adam)")
  parser.add_argument("--learning_rate", default=0.001, type=float, required=False,
                      help="Learning rate used during training")
  parser.add_argument("--beta_1", default=0.9, type=float, required=False, 
                      help="The first hyperparameter for the Adam optimizer")
  parser.add_argument("--beta_2", default=0.999, type=float, required=False, 
                      help="The second hyperparameter for the Adam optimizer")
  parser.add_argument("--epsilon", default=1e-7, type=float, required=False, 
                      help="A small constant added to the denominator to prevent division by zero for the Adam optimizer")
  parser.add_argument("--momentum", default=0., type=float, required=False, 
                      help="Momentum term for the SGD optimizer")
  parser.add_argument("--nesterov", default=False, type=bool, required=False, 
                      help="Whether to use Nesterov momentum for the SGD optimizer")
  parser.add_argument("--rho", default=0.9, type=float, required=False, 
                      help="Decay rate for the moving average of the squared gradient for the RMSprop optimizer") 
  parser.add_argument("--rmsprop_momentum", default=0., type=float, required=False, 
                      help="Momentum term for the RMSprop optimizer")
  parser.add_argument("--callbacks", default=False, type=bool, required=False, 
                      help="Whether to use callbacks during training. If True, 'EarlyStopping' and 'ModelCheckpoint' callbacks run")
  parser.add_argument("--monitor", default="loss", type=str, required=False, 
                      help="Type to monitor the evaulation metrics during training, if 'callbacks' is True")
  parser.add_argument("--mode", default="min", type=str, required=False, 
                      help="Whether the monitor should be minimized or maximized during training, if 'callbacks' is True") 
  parser.add_argument("--patience", default=5, type=int, required=False,
                      help="Number of epochs to wait before stopping training if no improvement is seen in the monitor, if 'callbacks' is True")

  # Parse the arguments and convert them to a dictionary
  args = vars(parser.parse_args())

  return args


if __name__ == "__main__":

    # Set arguments
    args = set_arguments()

    # Load and preprocess the datasets.
    load_dataset = LoadDataset(args["image_dataset_format"], args["mask_dataset_format"], args["height"], args["width"], args["batch_size"], args["buffer_size"])
    train_dataset, validation_dataset = load_dataset(args["image_dataset_path"], args["mask_dataset_path"], args["validation_split"])

    # Build the UNet model
    input_shape = (args["height"], args["width"], 3)
    unet = UNet(input_shape, args["nb_classes"], args["nb_filters"], args["kernel_size"], 
                args["activation"], args["kernel_initializer"], args["optimizer"], 
                args["learning_rate"], args["beta_1"], args["beta_2"], args["epsilon"], 
                args["momentum"], args["nesterov"], args["rho"], args["rmsprop_momentum"])

    # Train the model and store the logs in 'history' variable
    history = unet.train(train_dataset, validation_dataset, args["epochs"], args["batch_size"], 
                         args["callbacks"],  args["monitor"], args["mode"], args["patience"])
    