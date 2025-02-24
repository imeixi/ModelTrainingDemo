import os
import tensorflow as tf
import keras_tuner as kt
from tensorflow.keras import layers
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# GPU 内存增长设置
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

class CNNHyperModel(kt.HyperModel):
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build(self, hp):
        model = tf.keras.Sequential()
        
        # 输入层
        model.add(layers.Input(shape=self.input_shape))
        
        # GPU版本：扩大搜索空间
        for i in range(hp.Int('conv_blocks', 1, 6)):  # 增加最大层数
            filters = hp.Int(f'conv_{i}_filters', min_value=32, max_value=256, step=32)  # 增加过滤器数量
            kernel_size = hp.Choice(f'conv_{i}_kernel', values=[3, 5, 7])  # 增加核大小选项
            
            model.add(layers.Conv2D(filters, kernel_size, activation='relu'))
            
            if hp.Boolean(f'batch_norm_{i}'):
                model.add(layers.BatchNormalization())
            
            if hp.Boolean(f'pooling_{i}'):
                pool_size = hp.Choice(f'pool_{i}_size', values=[2, 3])
                model.add(layers.MaxPooling2D(pool_size))
            
            if hp.Boolean(f'dropout_{i}'):
                dropout_rate = hp.Float(f'dropout_{i}_rate', min_value=0.1, max_value=0.5, step=0.1)
                model.add(layers.Dropout(dropout_rate))

        model.add(layers.Flatten())
        
        # GPU版本：扩大全连接层搜索空间
        for i in range(hp.Int('dense_blocks', 1, 4)):  # 增加最大层数
            units = hp.Int(f'dense_{i}_units', min_value=64, max_value=1024, step=64)  # 增加单元数
            model.add(layers.Dense(units, activation='relu'))
            
            if hp.Boolean(f'dense_batch_norm_{i}'):
                model.add(layers.BatchNormalization())
            
            if hp.Boolean(f'dense_dropout_{i}'):
                dropout_rate = hp.Float(f'dense_dropout_{i}_rate', min_value=0.1, max_value=0.5, step=0.1)
                model.add(layers.Dropout(dropout_rate))

        model.add(layers.Dense(self.num_classes, activation='softmax'))
        
        # GPU版本：使用更大的学习率范围
        learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'],
            # GPU版本：启用混合精度训练
            jit_compile=True
        )
        
        return model

def main():
    # 设置随机种子
    tf.random.set_seed(42)
    
    # 启用混合精度
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    
    # 加载数据
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255

    # 创建保存目录
    os.makedirs('d:/Workspace/ModelTrainingDemo/automl_results_gpu', exist_ok=True)
    
    # GPU版本：增加搜索次数和时间
    tuner = kt.Hyperband(
        CNNHyperModel(input_shape=(28, 28, 1), num_classes=10),
        objective='val_accuracy',
        max_epochs=20,  # 增加最大训练轮数
        factor=3,
        directory='d:/Workspace/ModelTrainingDemo/automl_results_gpu',
        project_name='mnist_automl_gpu'
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,  # 增加早停耐心值
        restore_best_weights=True
    )

    print("开始搜索最佳模型架构...")
    tuner.search(
        x_train, y_train,
        validation_split=0.2,
        callbacks=[early_stopping],
        epochs=20,  # 增加训练轮数
        batch_size=128  # 增大批量大小
    )

    # ... 后续代码与原始版本相同 ...