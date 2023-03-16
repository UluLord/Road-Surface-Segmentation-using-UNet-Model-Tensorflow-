import os
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

class UNet:
    """
    This class builds a U-Net model for image segmentation.
    """
    def __init__(self, input_shape, nb_classes, nb_filters, kernel_size, activation, kernel_initializer,
                 optimizer, learning_rate, beta_1, beta_2, epsilon, momentum, nesterov, rho, rmsprop_momentum):
        """
        Initialize the class.

        Args:
        - input_shape (tuple): input shape of the model.
        - nb_classes (int): number of classes in the dataset.
        - nb_filters (int): number of filters in each layer.
        - kernel_size (int): size of the kernel.
        - activation (str): activation function to use in the layers.
        - kernel_initializer (str): method to initialize the kernel.
        - optimizer (str): optimizer to use in training.
        - learning_rate (float): learning rate for the optimizer.
        - beta_1 (float): beta_1 for the Adam optimizer.
        - beta_2 (float): beta_2 for the Adam optimizer.
        - epsilon (float): epsilon for the Adam optimizer.
        - momentum (float): momentum for the SGD optimizer.
        - nesterov (bool): nesterov for the SGD optimizer.
        - rho (float): rho for the RMSprop optimizer.
        - rmsprop_momentum (float): rmsprop_momentum for the RMSprop optimizer.
        """
        # Input layer
        self.input = tf.keras.layers.Input(shape=input_shape)

        # Encoding blocks
        self.eblock1 = self.encoding_block(self.input, nb_filters*1, 0., True, kernel_size, activation, kernel_initializer)
        self.eblock2 = self.encoding_block(self.eblock1[0], nb_filters*2, 0., True, kernel_size, activation, kernel_initializer)
        self.eblock3 = self.encoding_block(self.eblock2[0], nb_filters*4, 0., True, kernel_size, activation, kernel_initializer)
        self.eblock4 = self.encoding_block(self.eblock3[0], nb_filters*8, 0., True, kernel_size, activation, kernel_initializer)
        self.eblock5 = self.encoding_block(self.eblock4[0], nb_filters*16, 0.3, True, kernel_size, activation, kernel_initializer)
        self.eblock6 = self.encoding_block(self.eblock5[0], nb_filters*32, 0.3, False, kernel_size, activation, kernel_initializer)

        # Decoding blocks
        self.dblock1 = self.decoding_block(self.eblock6[0], self.eblock5[1], nb_filters*16, kernel_size, activation, kernel_initializer)
        self.dblock2 = self.decoding_block(self.dblock1, self.eblock4[1], nb_filters*8, kernel_size, activation, kernel_initializer)
        self.dblock3 = self.decoding_block(self.dblock2, self.eblock3[1], nb_filters*4, kernel_size, activation, kernel_initializer)
        self.dblock4 = self.decoding_block(self.dblock3, self.eblock2[1], nb_filters*2, kernel_size, activation, kernel_initializer)
        self.dblock5 = self.decoding_block(self.dblock4, self.eblock1[1], nb_filters*1, kernel_size, activation, kernel_initializer)

        # Convolution layer
        self.conv = tf.keras.layers.Conv2D(nb_filters, kernel_size=kernel_size, padding="same", kernel_initializer=kernel_initializer, activation=activation)(self.dblock5)

        # Last layer with number of classes filters
        self.output = tf.keras.layers.Conv2D(filters=nb_classes, kernel_size=1, padding="same", activation="softmax")(self.conv)

        # Create the model
        self.model = tf.keras.models.Model(inputs=self.input, outputs=self.output)

        # Compile the model with specified loss and optimizer
        self.optimizer = self.get_optimizer(optimizer, learning_rate, beta_1, beta_2, epsilon, momentum, nesterov, rho, rmsprop_momentum)
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy()
        self.metrics = ["accuracy"]
        self.model.compile(loss=self.loss, 
                            optimizer=self.optimizer, 
                            metrics=self.metrics)
    
    @staticmethod
    def encoding_block(input, nb_filters, dropout_rate, max_pooling, kernel_size, activation, kernel_initializer):
        """
        This function creates an encoding block of the UNet architecture.

        Args:
        - input (tensor): input tensor to the encoding operation.
        - nb_filters (int): number of filters for the Conv2D layers.
        - dropout_rate (float, optional): dropout rate to apply after the Conv2D layers (defaults to 0.).
        - max_pooling (bool, optional): whether to use max pooling after the Conv2D layers (defaults to True).

        Returns:
        - next_layer (tensor): output layer of the encoding block.
        - skip_connection (tensor): last layer before pooling operation of the encoding block.
        """
        # First convolution layer of the block
        conv = tf.keras.layers.Conv2D(filters=nb_filters, 
                                      kernel_size=kernel_size, 
                                      strides=1, 
                                      padding="same", 
                                      kernel_initializer=kernel_initializer, 
                                      activation=activation)(input)

        # Second convolution layer of the block
        conv = tf.keras.layers.Conv2D(filters=nb_filters, 
                                      kernel_size=kernel_size, 
                                      strides=1, 
                                      padding="same", 
                                      kernel_initializer=kernel_initializer, 
                                      activation=activation)(conv)

        # Apply dropout if the 'dropout_rate' argument is greater than zero.
        if dropout_rate > 0.:
           conv = tf.keras.layers.Dropout(dropout_rate)(conv)

        # Apply maximum pooling if the 'max_pooling' argument is True.
        if max_pooling:
           next_layer = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(conv)
        
        # Otherwise, set the next layer to the last convolution layer.
        else:
           next_layer = conv

        skip_connection = conv

        return next_layer, skip_connection

    @staticmethod
    def decoding_block(input, skip_connection, nb_filters, kernel_size, activation, kernel_initializer):
        """
        This function creates a decoding block of the UNet architecture.

        Args:
        - input (tensor): input to the decoding layer.
        - skip_connection (tensor): tensor from the encoding layer to be concatenated with up_conv tensor.
        - nb_filters (int): number of filters used in Conv2D and Conv2DTranspose layers.

        Returns:
        - conv (tensor): output layer of the decoding block.
        """
        # Upsample the input layer taking as argument
        up_conv = tf.keras.layers.Conv2DTranspose(filters=nb_filters, 
                                                  kernel_size=kernel_size, 
                                                  strides=2, 
                                                  padding="same")(input)

        # Concatenate upsampled convolution layer with layer coming from encoding block
        added_layers = tf.keras.layers.concatenate(inputs=[up_conv, skip_connection], axis=3)

        # First convolution layer of the block
        conv = tf.keras.layers.Conv2D(filters=nb_filters,
                                      kernel_size=kernel_size,
                                      strides=1,
                                      padding="same",
                                      kernel_initializer=kernel_initializer,
                                      activation=activation)(added_layers)

        # Second convolution layer of the block
        conv = tf.keras.layers.Conv2D(filters=nb_filters,
                                      kernel_size=kernel_size,
                                      strides=1,
                                      padding="same",
                                      kernel_initializer=kernel_initializer,
                                      activation=activation)(conv)

        return conv

    @staticmethod
    def get_optimizer(optimizer, learning_rate, beta_1, beta_2, epsilon, momentum, nesterov, rho, rmsprop_momentum):
        """
        This function sets optimizer with it's parameters.

        Args:
        - optimizer (str): model optimizer type.
        - learning_rate (float): learning rate used during training.
        - beta_1 (float): the first hyperparameter for the Adam optimizer.
        - beta_2 (float): the second hyperparameter for the Adam optimizer.
        - epsilon (float): a small constant added to the denominator to prevent division by zero for the Adam optimizer.
        - momentum (float): momentum term for the SGD optimizer.
        - nesterov (bool): whether to use Nesterov momentum for the SGD optimizer.
        - rho (float): decay rate for the moving average of the squared gradient for the RMSprop optimizer. 
        - rmsprop_momentum (float): momentum term for the RMSprop optimizer.

        Returns:
        - optimizer: model optimizer with it's parameters.

        Raises:
        - ValueError: if the optimizer is not recognized.
        """
        # Initialize optimizer variable
        model_optimizer = None
        
        # Check if the model optimizer is SGD
        if optimizer == "sgd":
            # Use the SGD optimizer with specified learning rate, momentum, and nesterov attributes
            model_optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum,nesterov=nesterov)
            
        # Check if the model optimizer is RMSprop
        elif optimizer == "rmsprop":
            # Use the RMSprop optimizer with specified learning rate, rho, and momentum attributes
            model_optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate, rho=rho, momentum=rmsprop_momentum)
            
        # Check if the model optimizer is Adam
        elif optimizer == "adam":
            # Use the Adam optimizer with specified learning rate, beta_1, beta_2, and epsilon attributes
            model_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)
            
        # Give error if unsupported optimizer type specified.
        else:
            raise ValueError("Unsupported optimizer format: {}".format(optimizer))
            
        return model_optimizer
        
    @staticmethod
    def plot_results(history):
        """
        This function visualizes and saves the evaluation metrics of the model.

        Args:
        - history (dict): a dictionary containing the training history.

        Returns:
        - None 
        """
        pd.DataFrame(history.history).plot()
        plt.title("Result of the Evaluating Metrics")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.grid(True)
        if os.path.exists("./model_metrics/") == False:
            os.makedirs("./model_metrics/")
        plt.savefig("./model_metrics/plot_metrics.png")
        plt.show()

    def train(self, train_dataset, validation_dataset, epochs, batch_size, callbacks, monitor, mode, patience):
        """
        This function trains the UNet model.

        Args:
        - train_dataset (Tensorflow dataset): the dataset to use for training the model.
        - validation_dataset (Tensorflow dataset): the dataset to use for validating the model.
        - epochs (int): number of training epochs.
        - batch_size (int): batch size for the training data.
        - validation_split (float): fraction of the dataset to use for validation.
        - callbacks (bool): whether to use callbacks during training
        - monitor (str): metric to use for early stopping.
        - mode (str): whether the monitor should be minimized or maximized during training.
        - patience (int): number of epochs to wait before stopping training if no improvement is seen in the monitor.

        Returns:
        - hist (dict): a dictionary containing the training history.
        """
        # Set callbacks
        mycallbacks = None
        
        # Check if the 'callbacks' argument is True
        if callbacks:
            print("Callbacks enabled. To disenable, please set the 'callbacks' argument as False")
            print("Early Stopping setting up...")
            # Create a stopper if no any improvement in model.
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor=monitor,
                                                              mode=mode,
                                                              patience=patience,
                                                              restore_best_weights=True)
            print("Model Checkpoint setting up...")   
            # Create checkpoint saving.
            checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath="./weights/best_weights.h5",
                                                            monitor=monitor,
                                                            mode=mode,
                                                            save_best_only=True)   
            # Set all callbacks
            mycallbacks = [early_stopping, checkpoint]
        
        # Check if the 'callbacks' argument is False
        else:
            print("Callbacks disenabled. To enable, please set the 'callbacks' argument as True")
            
        # Train the model
        print("Model training ...")
        history = self.model.fit(train_dataset, batch_size=batch_size,
                                 epochs=epochs,
                                 validation_data=validation_dataset,
                                 callbacks=mycallbacks,
                                 verbose=1)

        print("\nThe training process has finished.")
        # If the model was stopped early, load the best weights
        model_epochs = len(history.history[monitor])
        if model_epochs < epochs:
            # Create a weights directory if no exists
            if os.path.exists("./weights") == False: 
                os.makedirs("./weights")

            print("Best weights saved to and loading from './weights/best_weights.h5'")
            self.model.load_weights("./weights/best_weights.h5")

        self.model.save("pretrained_model")
        print("The model saved to './pretrained_model' directory")
        self.plot_results(history)
        return history
