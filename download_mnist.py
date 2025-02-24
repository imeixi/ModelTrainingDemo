import tensorflow as tf
import numpy as np
import os
import pickle

def download_and_save_mnist(save_dir='./mnist_data'):
    # 创建保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 下载MNIST数据集
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # 数据预处理
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255
    
    # 保存为numpy格式
    np.save(os.path.join(save_dir, 'x_train.npy'), x_train)
    np.save(os.path.join(save_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(save_dir, 'x_test.npy'), x_test)
    np.save(os.path.join(save_dir, 'y_test.npy'), y_test)
    
    # 保存为pickle格式（可选）
    dataset = {
        'x_train': x_train,
        'y_train': y_train,
        'x_test': x_test,
        'y_test': y_test
    }
    with open(os.path.join(save_dir, 'mnist.pkl'), 'wb') as f:
        pickle.dump(dataset, f)
    
    # 打印数据集信息
    print("数据集已保存到:", save_dir)
    print("训练集形状:", x_train.shape)
    print("测试集形状:", x_test.shape)
    print("保存的文件:")
    for file in os.listdir(save_dir):
        file_path = os.path.join(save_dir, file)
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        print(f" - {file}: {size_mb:.2f} MB")

def load_local_mnist(data_dir='./mnist_data'):
    """
    从本地加载MNIST数据集
    """
    # 从numpy文件加载
    try:
        x_train = np.load(os.path.join(data_dir, 'x_train.npy'))
        y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
        x_test = np.load(os.path.join(data_dir, 'x_test.npy'))
        y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
        print("成功从本地加载数据集")
        return (x_train, y_train), (x_test, y_test)
    except:
        print("无法从本地加载数据集")
        return None

if __name__ == '__main__':
    # 下载并保存数据集
    download_and_save_mnist()
    
    # 测试加载数据集
    data = load_local_mnist()
    if data:
        (x_train, y_train), (x_test, y_test) = data
        print("\n验证加载的数据:")
        print("训练集样本数:", len(x_train))
        print("测试集样本数:", len(x_test))