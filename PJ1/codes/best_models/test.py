import pickle

# 将这里的路径替换为你想要检查的 .pickle 文件的实际绝对路径
file_path = r'C:\Users\刘宇轩\Desktop\神经网络与深度学习\Project1-2026\PJ1\codes\best_models\best_model.pickle'

with open(file_path, 'rb') as f:
    model_data = pickle.load(f)

# 判断类型并输出信息
if isinstance(model_data, list):
    print("✅ 鉴定结果：这是一个 【MLP】 模型！")
    print(f"-> 网络尺寸结构: {model_data[0]}")
    print(f"-> 激活函数: {model_data[1]}")
    
elif isinstance(model_data, dict) and 'conv_params_list' in model_data:
    print("✅ 鉴定结果：这是一个 【CNN】 模型！")
    print(f"-> 卷积层配置: {model_data['conv_params_list']}")
    print(f"-> 全连接层维度: {model_data['fc_dims']}")
    print(f"-> 激活函数: {model_data['activation']}")
    
else:
    print("❌ 无法识别的模型格式，可能是文件损坏或不是通过你的 models.py 保存的。")