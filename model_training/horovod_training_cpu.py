import os
import tensorflow as tf
import horovod.tensorflow as hvd
import numpy as np
from tensorflow.keras import layers, models

# Initialize Horovod
hvd.init()

# Configure TensorFlow to use CPU
tf.config.set_visible_devices([], 'GPU')

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
    # Set number of threads for CPU training
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)

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
    
    # Horovod: adjust learning rate based on number of processes
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
        reduce_lr  # 添加学习率调度器
    ]

    # Train the model
    model.fit(x_train, y_train,
              batch_size=32,
              epochs=5,
              callbacks=callbacks,
              validation_data=(x_test, y_test),
              verbose=1 if hvd.rank() == 0 else 0)

    # 只在主进程(rank 0)进行模型评估和保存
    if hvd.rank() == 0:
        # 创建模型保存目录
        os.makedirs('d:/Workspace/ModelTrainingDemo/models', exist_ok=True)
        
        # 1. 详细评估模型性能
        print("\n=== 模型评估 ===")
        score = model.evaluate(x_test, y_test, verbose=1)
        print('测试集损失:', score[0])
        print('测试集准确率:', score[1])
        
        # 预测并计算详细指标
        y_pred = model.predict(x_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # 计算混淆矩阵
        from sklearn.metrics import confusion_matrix, classification_report
        cm = confusion_matrix(y_test, y_pred_classes)
        print("\n混淆矩阵:")
        print(cm)
        
        # 输出详细分类报告
        print("\n分类报告:")
        print(classification_report(y_test, y_pred_classes))
        
        # 2. 保存模型（多种格式）
        print("\n=== 保存模型 ===")
        # SavedModel 格式
        save_path = 'd:/Workspace/ModelTrainingDemo/models/mnist_cpu_model_savedmodel'
        model.save(save_path)
        print(f"模型已保存为 SavedModel 格式: {save_path}")
        
        # HDF5 格式
        h5_path = 'd:/Workspace/ModelTrainingDemo/models/mnist_cpu_model.h5'
        model.save(h5_path)
        print(f"模型已保存为 HDF5 格式: {h5_path}")
        
        # 保存模型权重
        weights_path = 'd:/Workspace/ModelTrainingDemo/models/mnist_cpu_model_weights'
        model.save_weights(weights_path)
        print(f"模型权重已保存: {weights_path}")
        
        # 转换并保存为 TFLite 格式
        tflite_path = 'd:/Workspace/ModelTrainingDemo/models/mnist_cpu_model.tflite'
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        print(f"模型已保存为 TFLite 格式: {tflite_path}")

if __name__ == '__main__':
    main()