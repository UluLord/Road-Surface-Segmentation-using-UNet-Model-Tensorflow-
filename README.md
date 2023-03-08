# **Road Surface Segmentation using UNet Model [Tensorflow]**

This repository contains code for segmenting road surfaces in images using the UNet architecture. UNet is a deep learning model that is commonly used for image segmentation tasks, and it has shown promising results in segmenting road surfaces from images captured by various sensors such as LiDAR, radar, and cameras.

![UNet](https://user-images.githubusercontent.com/99184963/223558911-82c13536-5d6e-4c69-bc05-1a2ed65b33d4.png)
>Retrieved from [this article](https://arxiv.org/pdf/2004.10696v1.pdf) 

**NOTE:** You can use the repository not only for this dataset, but also for any dataset you want. You are very free to arrange hyperparameters of the UNet model to improve your model for your project.

## **1. Dataset**
The dataset contains 701 frames and 13 classes;

* Background
* Asphalt Road
* Paved Road
* Unpaved Road
* Road Marking
* Speed-Bump
* Cats-Eye
* Storm-Drain
* Manhole Cover
* Patchs
* Water-Puddle
* Pothole
* Cracks

However, the dataset is not included in this repository, but you can download from [here.](https://lapix.ufsc.br/pesquisas/projeto-veiculo-autonomo/datasets/?lang=en)

## **2. Usage**

#### *2.1. Cloning*
To use the UNet model in your project for image segmentation, clone this repository using your terminal like following command;

    git clone https://github.com/UluLord/Road-Surface-Segmentation-using-UNet-Model-Tensorflow-.git

After cloning, change the directory, you are working, to this repository directory;

    cd Road-Surface-Segmentation-using-UNet-Model-Tensorflow-

### *2.2 Requirements*

This work has been tested on these libraries;

* Tensorflow: 2.11.0
* Numpy: 1.22.4
* Pandas: 1.4.4
* Matplotlib: 3.7.0

To install the required packages, run the following command;

    pip install -r requirements.txt

**NOTE:** It may work with other versions of the libraries, but this has not been tested.

* This work has also been tested on NVIDIA GeForce RTX 3060 GPU.

**NOTE:** It is highly recommended to work with a GPU.

### *2.3. Training the Model*

Then, use the **train.py** with desired parameters to train the model on your dataset.

***Parameters***
  * **image_dataset_path:** Path to the image dataset. Required.
  * **image_dataset_format:** The image format in the image dataset (choose one of those: jpeg, jpg, png, bmp or others). Default is 'png'.
  * **mask_dataset_path:** Path to the mask dataset. Please note that the mask dataset should be in binary format. Required.
  * **mask_dataset_format:** The image format in the masked image dataset (choose one of those: jpeg, jpg, png, bmp or others). Default is 'png'.
  * **height:** Height of image. Default is 256.
  * **width:** Width of image. Default is 256.
  * **nb_classes:** Number of classes in the dataset. Required.
  * **epochs:** Number of training epochs. Default is 100.
  * **batch_size:** Batch size for dividing the dataset into chunks. Default is 32.
  * **buffer_size:** Buffer size for shuffling the dataset. Default is 1000.
  * **validation_split:** Fraction of the dataset to use for validation. Default is 0.
  * **nb_filters:** Number of filters in the first encoder block. Default is 32.
  * **kernel_size:** Kernel size in the layers. Default is 3.
  * **activation:** Activation function to use in the layers. Default is 'relu'.
  * **kernel_initializer:** Method to initialize the kernel. Default is 'he_uniform'.
  * **optimizer:** Model optimizer type (choose one of those: sgd, rmsprop or adam). Default is 'adam'.
  * **learning_rate:** Learning rate used during training. Default is 0.001.
  * **beta_1:** The first hyperparameter for the Adam optimizer. Default is 0.9.
  * **beta_2:** The second hyperparameter for the Adam optimizer. Default is 0.999.
  * **epsilon:** A small constant added to the denominator to prevent division by zero for the Adam optimizer. Default is 1e-7.
  * **momentum:** Momentum term for the SGD optimizer. Default is 0.
  * **nesterov:** Whether to use Nesterov momentum for the SGD optimizer. Default is False.
  * **rho:** Decay rate for the moving average of the squared gradient for the RMSprop optimizer. Default is 0.9.
  * **rmsprop_momentum:** Momentum term for the RMSprop optimizer. Default is 0.
  * **callbacks:** Whether to use callbacks during training. If True, 'EarlyStopping' and 'ModelCheckpoint' callbacks run. Default is False.
  * **monitor:** Type to monitor the evaulation metrics during training, if 'callbacks' is True. Default is 'loss'.
  * **mode:** Whether the monitor should be minimized or maximized during training, if 'callbacks' is True. Default is 'min'. 
  * **patience:** Number of epochs to wait before stopping training if no improvement is seen in the monitor, if 'callbacks' is True. Default is 5.

**IMPORTANT:** If you want to use a dataset that has more or less than 13 classes, please consider changing the predicted mask color types specified in the 'change_colors()' function of the 'prediction_display.py' file. The function is only suitable for a 13-class dataset as such. Otherwise, you may encounter an error.

***Example Usage***

    python train.py --image_dataset_path ./datasets/OriginalFrames --image_dataset_format png --mask_dataset_path ./datasets/NoColorMapMaskFrames --mask_dataset_format png --height 256 --width 256 --nb_classes 13 --epochs 100 --batch_size 32 --validation_split 0.2 --callbacks True --patience 10

***Sample Training History***

A sample figure showing model performance in the training process;

![plot_metrics](https://user-images.githubusercontent.com/99184963/223560057-65dbfb8c-5e47-4537-9cc8-3b84114b0bc1.png)

* As you can see, the model training stopped early. This is because the 'callback' argument was set to True, and the model training process was monitored every 10 epochs because the 'patience' argument was set to 10. So, when there was no improvement for 10 consecutive epochs, early stopping was triggered.

### *2.4. Predicting New Masked Images*

To predict new images by using a pre-trained model;

1. Specify a pre-trained model (download new one or train a model by using train.py).

2. Use **predict.py** with desired parameters;

***Parameters***  
  * **pretrained_model_path:** Path to a pretrained model to use for prediction. Default is ‘./pretrained_model’.
  * **image_path:** Path to a sample image to use for prediction. Required if no image dataset is provided. Default is None.
  * **masked_image_path:** Path to a sample masked image path to compare with prediction. Default is None.
  * **image_dataset_path:** Path to an image dataset to use for predictions. Required if no sample image is provided. Default is None.
  * **image_dataset_format:** The image format in the image dataset (choose one of those: jpeg, jpg, png, bmp or others). Default is 'png'.
  * **masked_dataset_path:** Path to a masked image dataset to compare with predictions. Default is None.
  * **masked_dataset_format:** The image format in the masked image dataset (choose one of those: jpeg, jpg, png, bmp or others). Default is 'png'.
  * **height:** Height of image. Default is 256.
  * **width:** Width of image. Default is 256.
  * **show_only_predicted:** Whether to show only predicted image by the model. If 'masked_image_path' or 'masked_dataset_path' which aim to compare with predicted image are specified, the argument should be False. Default is True.

***Example Usage***

    python predict.py --pretrained_model_path ./pretrained_model --image_path ./images/road_image.png --masked_image_path ./masks/masked_road_image.png --height 256 --width 256 --show_only_predicted False


***Some of Predicted Images***

* A figure showing only predicted masked image by the model;

![only_predicted](https://user-images.githubusercontent.com/99184963/223561277-a4da0cdd-cb9f-43a7-98f6-e20978408138.png)

* A figure showing image and predicted masked image by the model;

![image_predicted](https://user-images.githubusercontent.com/99184963/223561650-1d3b5748-9ce7-42b3-bd68-15d0e7383252.png)

* A figure showing image, masked image, and predicted masked image by the model;

![image_mask_predicted](https://user-images.githubusercontent.com/99184963/223561472-475ec59c-3743-4732-8a25-09ebeede3370.png)


## **3. Citation**

    @misc{rateke:2021,
	      author = {Thiago Rateke and Aldo von Wangenheim},
          title = {Road surface detection and differentiation considering surface damages},
	      journal = {Autonomous Robots},
	      date = {2021-01-11},
	      issn = {1573-7527},
	      url= {https://doi.org/10.1007/s10514-020-09964-3},
	      DOI={10.1007/s10514-020-09964-3}
          }

If you use this repository in your work, please consider citing us as the following.

    @misc{ululord2023ululord2023road-surface-segmentation-using-unet-model-tensorflow,
	      author = {Fatih Demir},
          title = {Road Surface Segmentation using UNet Model [Tensorflow]},
          date = {2023-03-08},
          url = {https://github.com/UluLord/Road-Surface-Segmentation-using-UNet-Model-Tensorflow-}
          }
