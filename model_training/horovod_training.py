import os
import tensorflow as tf
import horovod.tensorflow as hvd
import numpy as np
from tensorflow.keras import layers, models

# Initialize Horovod
hvd.init()

# Pin GPU to be used to process local rank (one GPU per process)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

# Load and preprocess MNIST dataset
def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255
    return (x_train, y_train), (x_test, y_test)

# Create simple CNN model
def create_model():
    model = models.Sequential([
        layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, 3, activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model

def main():
    # Load data
    (x_train, y_train), (x_test, y_test) = load_data()
    
    # Split data between processes
    train_size = len(x_train) // hvd.size()
    test_size = len(x_test) // hvd.size()
    
    x_train = x_train[hvd.rank()*train_size:(hvd.rank()+1)*train_size]
    y_train = y_train[hvd.rank()*train_size:(hvd.rank()+1)*train_size]
    x_test = x_test[hvd.rank()*test_size:(hvd.rank()+1)*test_size]
    y_test = y_test[hvd.rank()*test_size:(hvd.rank()+1)*test_size]

    # Build model
    model = create_model()
    
    # Horovod: adjust learning rate based on number of GPUs
    initial_lr = 0.001 * hvd.size()
    opt = tf.keras.optimizers.Adam(initial_lr)
    
    # Horovod: add Horovod DistributedOptimizer
    opt = hvd.DistributedOptimizer(opt)
    
    # Compile model
    model.compile(optimizer=opt,
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])

    # Horovod: broadcast initial variable states from rank 0 to all other processes
    callbacks = [
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),
        hvd.callbacks.MetricAverageCallback(),
    ]

    # Train the model
    model.fit(x_train, y_train,
              batch_size=64,
              epochs=5,
              callbacks=callbacks,
              validation_data=(x_test, y_test),
              verbose=1 if hvd.rank() == 0 else 0)

    # Evaluate the model
    if hvd.rank() == 0:
        score = model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

        # Save the model
        model.save('mnist_distributed_model.h5')

if __name__ == '__main__':
    main()