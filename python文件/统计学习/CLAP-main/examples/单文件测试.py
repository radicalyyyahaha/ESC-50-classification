if __name__ == '__main__':
    from msclap import CLAP
    from esc50_dataset import ESC50
    import torch.nn.functional as F
    import numpy as np

    # Load dataset
    root_path = "C:\\Users\\wyy\\python文件\\统计学习\\CLAP-main\\CLAP-main\\examples"
    dataset = ESC50(root=root_path)
    prompt = 'this is the sound of '
    y = [prompt + x for x in dataset.classes]
    clap_model = CLAP(version='2023', use_cuda=False)  # 加载 CLAP
    text_embeddings = clap_model.get_text_embeddings(y)  # 计算 text embeddings
    x = r"D:\桌面\统计学习\assets\125250056-1-192 00_00_08-00_00_13.wav"  # 文件地址
    audio_embeddings = clap_model.get_audio_embeddings([x], resample=True)
    similarity = clap_model.compute_similarity(audio_embeddings, text_embeddings)
    y_pred = F.softmax(similarity.detach().cpu(), dim=1).numpy()
    sorted_indices = np.argsort(-y_pred, axis=1)  # 逆序排列每行概率的索引
    second_max_indices = sorted_indices[:, 1]  # 获取第二高概率的索引
    second_max_values = y_pred[np.arange(len(y_pred)), second_max_indices]  # 获取第二高概率的值
    max_indices = np.argmax(y_pred, axis=1)  # 找到每行最大值的索引
    predicted_labels = [y[code] for code in max_indices]  # 使用这些索引找到对应的类别标签
    second_predicted_labels = [y[code] for code in second_max_indices]  # 使用第二高概率的索引找到对应的类别标签
    print("Predicted Labels:", predicted_labels)
    print("Second Predicted Labels:", second_predicted_labels)
    print("Second Max Probabilities:", second_max_values)
