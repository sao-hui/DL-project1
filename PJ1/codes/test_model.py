import mynn as nn
import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle

def test_model_MLP():
        model = nn.models.Model_MLP()
        model.load_model(r'.\best_models\best_model_MLP.pickle')

        test_images_path = r'.\dataset\MNIST\t10k-images-idx3-ubyte.gz'
        test_labels_path = r'.\dataset\MNIST\t10k-labels-idx1-ubyte.gz'

        with gzip.open(test_images_path, 'rb') as f:
                magic, num, rows, cols = unpack('>4I', f.read(16))
                test_imgs=np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
        
        with gzip.open(test_labels_path, 'rb') as f:
                magic, num = unpack('>2I', f.read(8))
                test_labs = np.frombuffer(f.read(), dtype=np.uint8)

        test_imgs = test_imgs / test_imgs.max()

        logits = model(test_imgs)
        print(nn.metric.accuracy(logits, test_labs))

def test_model_CNN():

    cnn_model = nn.models.Model_CNN(conv_params_list=[], fc_dims=[])
    
    model_path = r'.\best_models\best_model_CNN.pickle' 
    cnn_model.load_model(model_path)

    test_images_path = r'.\dataset\MNIST\t10k-images-idx3-ubyte.gz'
    test_labels_path = r'.\dataset\MNIST\t10k-labels-idx1-ubyte.gz'

    with gzip.open(test_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        test_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 1, 28, 28)
        
    with gzip.open(test_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        test_labs = np.frombuffer(f.read(), dtype=np.uint8)

    test_imgs = test_imgs.astype(np.float32) / 255.0
    
    batch_size = 64
    num_batches = (test_imgs.shape[0] + batch_size - 1) // batch_size
    total_score = 0.0

    for iteration in range(num_batches):
        batch_imgs = test_imgs[iteration * batch_size : (iteration + 1) * batch_size]
        batch_labs = test_labs[iteration * batch_size : (iteration + 1) * batch_size]      
        logits = cnn_model(batch_imgs)
        score = nn.metric.accuracy(logits, batch_labs)

        total_score += score * batch_imgs.shape[0]
        
        # 打印一下进度条，免得干等
        if (iteration + 1) % 20 == 0 or (iteration + 1) == num_batches:
            print(f"⏳ 进度: [{iteration + 1}/{num_batches} batches]")

    final_accuracy = total_score / test_imgs.shape[0]
    print("-" * 40)
    print(f"最终准确率为: {final_accuracy * 100:.2f}%")


if __name__ == "__main__":
        test_model_MLP()
        test_model_CNN()