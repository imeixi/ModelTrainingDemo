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
    # Restrict TensorFlow to only use one GPU
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
    # Allow memory growth for the GPU
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

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

    # 添加学习率调度器回调
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        patience=2,
        verbose=1 if hvd.rank() == 0 else 0
    )

    # Horovod: broadcast initial variable states from rank 0 to all other processes
    callbacks = [
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),
        hvd.callbacks.MetricAverageCallback(),
        hvd.callbacks.LearningRateWarmupCallback(initial_lr, warmup_epochs=3),
        reduce_lr  # 添加学习率调度器
    ]

    # Train the model
    model.fit(x_train, y_train,
              batch_size=64,  # 增大batch size以充分利用GPU
              epochs=10,      # 增加训练轮数
              callbacks=callbacks,
              validation_data=(x_test, y_test),
              verbose=1 if hvd.rank() == 0 else 0)

    # 评估模型
    if hvd.rank() == 0:
        # 在测试集上评估
        score = model.evaluate(x_test, y_test, verbose=1)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

        # 进行预测并输出详细指标
        y_pred = model.predict(x_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # 计算混淆矩阵
        from sklearn.metrics import confusion_matrix, classification_report
        cm = confusion_matrix(y_test, y_pred_classes)
        print("\nConfusion Matrix:")
        print(cm)
        
        # 输出详细分类报告
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_classes))

        # 保存模型（多种格式）
        # 1. SavedModel 格式（推荐）
        model.save('models/mnist_model_savedmodel')
        
        # 2. HDF5 格式
        model.save('models/mnist_model.h5')
        
        # 3. 仅保存权重
        model.save_weights('models/mnist_model_weights')
        
        # 4. 导出为 TensorFlow Lite 格式
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        with open('models/mnist_model.tflite', 'wb') as f:
            f.write(tflite_model)

if __name__ == '__main__':
    main()