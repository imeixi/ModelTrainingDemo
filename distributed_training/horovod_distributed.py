import os
import horovod.tensorflow as hvd
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# 初始化 Horovod
hvd.init()

# 配置 GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[hvd.local_rank()], True)

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
    # 加载数据
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255

    # 数据分片
    train_size = len(x_train) // hvd.size()
    x_train = x_train[hvd.rank()*train_size:(hvd.rank()+1)*train_size]
    y_train = y_train[hvd.rank()*train_size:(hvd.rank()+1)*train_size]

    # 创建模型
    model = create_model()
    
    # 设置优化器
    opt = tf.keras.optimizers.Adam(learning_rate=0.001 * hvd.size())
    opt = hvd.DistributedOptimizer(opt)
    
    # 编译模型
    model.compile(optimizer=opt,
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])

    # Horovod 回调
    callbacks = [
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),
        hvd.callbacks.MetricAverageCallback(),
    ]

    # 训练模型
    model.fit(x_train, y_train,
              batch_size=32,
              epochs=10,
              callbacks=callbacks,
              verbose=1 if hvd.rank() == 0 else 0)

    # 在主进程上保存模型
    if hvd.rank() == 0:
        model.save('d:/Workspace/ModelTrainingDemo/distributed_training/models/horovod_model')

if __name__ == '__main__':
    main()