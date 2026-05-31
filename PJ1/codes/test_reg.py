import mynn as nn
import numpy as np
import gzip
from struct import unpack
import matplotlib.pyplot as plt
import copy

def load_mnist_data():
    """读取并处理数据"""
    train_images_path = r'.\dataset\MNIST\train-images-idx3-ubyte.gz'
    train_labels_path = r'.\dataset\MNIST\train-labels-idx1-ubyte.gz'
    with gzip.open(train_images_path, 'rb') as f:
        _, num, _, _ = unpack('>4I', f.read(16))
        imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28).astype(np.float32) / 255.0
    with gzip.open(train_labels_path, 'rb') as f:
        _, num = unpack('>2I', f.read(8))
        labs = np.frombuffer(f.read(), dtype=np.uint8)
    
    # 随机划分
    np.random.seed(309)
    idx = np.random.permutation(np.arange(num))
    return imgs[idx], labs[idx]

def run_experiment(imgs, labs, reg_lambda):
    """配置并训练一个模型"""
    train_imgs, train_labs = imgs[10000:], labs[10000:]
    valid_imgs, valid_labs = imgs[:10000], labs[:10000]
    
    # 初始化模型：如果 lambda > 0 则开启正则化
    # 假设你的 Model_MLP 支持 weight_decay 列表传递
    use_decay = (reg_lambda > 0)
    model = nn.models.Model_MLP([784, 256, 10], 'ReLU', [reg_lambda, reg_lambda] if use_decay else None)
    
    optimizer = nn.optimizer.SGD(init_lr=0.05, model=model)
    loss_fn = nn.op.MultiCrossEntropyLoss(model=model, max_classes=10)
    runner = nn.runner.RunnerM(model, optimizer, nn.metric.accuracy, loss_fn, batch_size=64)
    
    print(f"开始训练 (Lambda={reg_lambda})...")
    runner.train([train_imgs, train_labs], [valid_imgs, valid_labs], num_epochs=20, log_iters=100)
    return runner

def plot_multi_reg_comparison(results):
    """
    results: 字典，格式为 {lambda_value: runner_object}
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 颜色列表
    colors = ['#E63946', '#457B9D', '#2A9D8F', '#F4A261', '#9B5DE5']
    
    for i, (lmbda, runner) in enumerate(results.items()):
        label_name = f'Lambda={lmbda}'
        color = colors[i % len(colors)]
        
        axes[0].plot(runner.dev_loss, label=label_name, color=color, linewidth=2)
        axes[1].plot(runner.dev_scores, label=label_name, color=color, linewidth=2)
    
    axes[0].set_title("Validation Loss Comparison")
    axes[0].set_xlabel("Iterations (x100)")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle=':', alpha=0.6)
    axes[0].legend()
    
    axes[1].set_title("Validation Accuracy Comparison")
    axes[1].set_xlabel("Iterations (x100)")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True, linestyle=':', alpha=0.6)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('multi_reg_comparison.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    imgs, labs = load_mnist_data()
    
    # 定义你要测试的正则化强度梯度
    lambdas_to_test = [0, 1e-5, 1e-4, 1e-3, 1e-2]
    results = {}
    
    for lmbda in lambdas_to_test:
        print(f"\n>>> 开始测试正则化强度: {lmbda}")
        runner = run_experiment(imgs, labs, reg_lambda=lmbda)
        results[lmbda] = runner
        
    plot_multi_reg_comparison(results)