import os
import tensorflow as tf
import keras_tuner as kt
from tensorflow.keras import layers
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# 强制使用 CPU
tf.config.set_visible_devices([], 'GPU')

class CNNHyperModel(kt.HyperModel):
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build(self, hp):
        model = tf.keras.Sequential()
        
        # 输入层
        model.add(layers.Input(shape=self.input_shape))
        
        # CPU版本：减少搜索空间
        for i in range(hp.Int('conv_blocks', 1, 3)):  # 减少最大层数
            filters = hp.Int(f'conv_{i}_filters', min_value=32, max_value=64, step=32)  # 减少过滤器数量
            kernel_size = hp.Choice(f'conv_{i}_kernel', values=[3])  # 固定核大小
            
            model.add(layers.Conv2D(filters, kernel_size, activation='relu'))
            
            if hp.Boolean(f'batch_norm_{i}'):
                model.add(layers.BatchNormalization())
            
            if hp.Boolean(f'pooling_{i}'):
                model.add(layers.MaxPooling2D(2))  # 固定池化大小
            
            if hp.Boolean(f'dropout_{i}'):
                dropout_rate = hp.Float(f'dropout_{i}_rate', min_value=0.2, max_value=0.4, step=0.2)
                model.add(layers.Dropout(dropout_rate))

        model.add(layers.Flatten())
        
        # CPU版本：减少全连接层搜索空间
        for i in range(hp.Int('dense_blocks', 1, 2)):  # 减少最大层数
            units = hp.Int(f'dense_{i}_units', min_value=32, max_value=128, step=32)  # 减少单元数
            model.add(layers.Dense(units, activation='relu'))
            
            if hp.Boolean(f'dense_dropout_{i}'):
                model.add(layers.Dropout(0.3))  # 固定dropout率

        model.add(layers.Dense(self.num_classes, activation='softmax'))
        
        # CPU版本：使用较小的学习率范围
        learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-3, sampling='log')
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model

def main():
    # 设置随机种子
    tf.random.set_seed(42)
    
    # 加载数据
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255

    # 创建保存目录
    os.makedirs('d:/Workspace/ModelTrainingDemo/automl_results_cpu', exist_ok=True)
    
    # CPU版本：减少搜索次数和时间
    tuner = kt.Hyperband(
        CNNHyperModel(input_shape=(28, 28, 1), num_classes=10),
        objective='val_accuracy',
        max_epochs=5,  # 减少最大训练轮数
        factor=3,
        directory='d:/Workspace/ModelTrainingDemo/automl_results_cpu',
        project_name='mnist_automl_cpu'
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=2,  # 减少早停耐心值
        restore_best_weights=True
    )

    print("开始搜索最佳模型架构...")
    tuner.search(
        x_train, y_train,
        validation_split=0.2,
        callbacks=[early_stopping],
        epochs=5,  # 减少训练轮数
        batch_size=32  # 减小批量大小
    )

    # ... 后续代码与原始版本相同 ...