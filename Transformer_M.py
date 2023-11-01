#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : Zhenghliu5

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import math

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn.functional as F

'''
    1. 数据读取及预处理
'''
StockList = ['中国人寿','中国平安','五粮液','贵州茅台','工商银行','招商银行']
EStockList = ['China Life Insurance', 'Ping An of China', 'Wuliangye', 'Kweichow Moutai', 'ICBC', 'China Merchants Bank']
for ss in range(4, len(StockList)-1):

    data = pd.read_csv('./data/'+StockList[ss]+'technical_indicators.csv', encoding='utf-8-sig')

    # 填充缺失值
    # 假设缺失值用NaN表示
    # data.fillna(method='ffill', inplace=True)
    data.fillna(method='bfill', inplace=True)

    # 检查是否还存在缺失值
    # print(data.isnull().sum())

    # 将交易日期列转换为日期类型
    data['Trade_Date'] = pd.to_datetime(data['Date'])

    # 按照交易日期升序排序
    data.sort_values('Trade_Date', inplace=True)

    # 重置索引
    data.reset_index(drop=True, inplace=True)

    # 选择合适的特征列
    selected_features1 = ['Amount_x', 'Change', 'High', 'Low', 'Open', 'Vol', 'Close']
                         # 'news_sentiment', 'comment_sentiment',
                         # 'r1', 'r3', 'r4',
                         # 'MA_7', 'MA_21', 'RSI', 'EMA_12', 'MACD', 'Signal_Line', 'W%R', 'TR','momentum']
    # 'Ret', 'PE',	'PB', 'PS', 'Turnover', 'ChangeRatio'
    # selected_features1 = list(data.columns[4:12]) + list(data.columns[14:19])
    # target_column = 'Close'

    # 提取所选特征和目标列
    data1 = data[selected_features1]

    # 将特征列转换为PyTorch张量
    features1 = torch.tensor(data1[selected_features1].values, dtype=torch.float32)

    # 使用MinMaxScaler对选择的特征进行归一化处理
    scaler = MinMaxScaler()
    scaled_features1 = scaler.fit_transform(features1)

    # 将归一化后的特征更新到数据集中
    data1[selected_features1] = scaled_features1

    # 将三个特征矩阵转化为Tensor格式

    feature_matrix1 = torch.tensor(data1[selected_features1].values, dtype=torch.float32)  # 将DataFrame转换为Tensor
    labels = torch.tensor(data1['Close'].values, dtype=torch.float32)  # 提取标签列并转换为Tensor

    '''
        2. 创建滑动窗口数据集
    '''

    def create_sliding_window_dataset(data, window_size):
        inputs = []
        targets = []

        for i in range(len(data) - window_size):
            window = data[i:i+window_size]
            # print(window)
            target = data[i+window_size,-1]
            # print(target)
            inputs.append(window)
            targets.append(target)

        return np.array(inputs), np.array(targets).reshape(-1, 1)

    '''
        3. 构建自定义数据集
    '''
    class StockDataset(Dataset):
        def __init__(self, inputs, targets):
            self.inputs = inputs
            self.targets = targets

        def __len__(self):
            return len(self.inputs)

        def __getitem__(self, index):
            input_data = torch.from_numpy(self.inputs[index])
            target = torch.from_numpy(self.targets[index])
            return input_data, target

    '''
        4. 划分训练集和测试集
    '''

    # 调用滑动窗口数据集

    batch_size = 128   # 每一次处理的数据量
    window_size = 28    # 滑动窗口大小

    inputs, targets = create_sliding_window_dataset(data1.values, window_size)
    # inputs = np.transpose(inputs, (0, 2, 1))

    print(inputs.shape)
    print(targets.shape)
    train_dataset = StockDataset(inputs[:800], targets[:800])
    test_dataset = StockDataset(inputs[800:], targets[800:])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    '''
        5. 搭建模型
    '''

    import torch
    import torch.nn as nn


    class TransformerModel(nn.Module):

        torch.set_default_tensor_type(torch.DoubleTensor)

        def __init__(self, input_size, hidden_size, num_classes, num_layers, num_heads, dropout):
            super(TransformerModel, self).__init__()

            self.embedding = nn.Linear(input_size, hidden_size)
            self.positional_encoding = PositionalEncoding(hidden_size, dropout)

            encoder_layer = nn.TransformerEncoderLayer(hidden_size, num_heads)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

            self.attention = nn.MultiheadAttention(hidden_size, num_heads)
            self.fc = nn.Linear(hidden_size, num_classes)

        def forward(self, x):
            x = self.embedding(x)
            x = self.positional_encoding(x)

            x = x.permute(1, 0, 2)  # 将维度顺序变为 (seq_len, batch_size, hidden_size)

            x, _ = self.attention(x, x, x)  # 使用自注意力机制

            x = x.permute(1, 0, 2)  # 将维度顺序还原为 (batch_size, seq_len, hidden_size)

            output = self.fc(x.mean(dim=1))  # 对序列维度进行平均池化

            return output


    class PositionalEncoding(nn.Module):
        def __init__(self, hidden_size, dropout, max_length=1000):
            super(PositionalEncoding, self).__init__()

            self.dropout = nn.Dropout(p=dropout)

            positional_encoding = torch.zeros(max_length, hidden_size)
            position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * (-math.log(10000.0) / hidden_size))
            positional_encoding[:, 0::2] = torch.sin(position * div_term)
            positional_encoding[:, 1::2] = torch.cos(position * div_term)
            positional_encoding = positional_encoding.unsqueeze(0).transpose(0, 1)

            self.register_buffer('positional_encoding', positional_encoding)

        def forward(self, x):
            x = x + self.positional_encoding[:x.size(0), :]
            return self.dropout(x)

    '''
        6.加载模型并设置优化器
    '''
    import torch.optim as optim

    input_size = data1.shape[1]
    hidden_size = 64  # 隐藏层维度
    num_classes = 1  # 输出类别的数量
    num_layers = 2  # Transformer 模型的层数
    num_heads = 4  # 注意力头的数量
    dropout = 0.1  # Dropout 比例

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model = TransformerModel(input_size, hidden_size, num_classes, num_layers, num_heads, dropout)
    # model = model.to(device)

    # 损失函数
    criterion = nn.MSELoss()

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    '''
        7.模型训练和测试
    '''
    # 定义训练和测试循环
    def train(model, dataloader, criterion, optimizer):
        model.train()
        total_loss = 0.0

        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs,targets)
            loss.backward()
            optimizer.step()

            total_loss += loss
            # 计算平均损失
            avg_loss = total_loss / len(dataloader)
        return avg_loss

    def test(model, dataloader, criterion):
        model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for inputs, targets in dataloader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss

        avg_loss = total_loss / len(dataloader)
        return avg_loss

    # 创建空列表以存储训练和测试损失
    train_losses = []
    test_losses = []

    # 训练和测试循环
    num_epochs = 100

    for epoch in range(num_epochs):
        train_loss = train(model, train_dataloader, criterion, optimizer)
        test_loss = test(model, test_dataloader, criterion)

        # 记录训练和测试损失
        train_losses.append(train_loss.detach().numpy())
        test_losses.append(test_loss.detach().numpy())

        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss = {train_loss:.6f}, Test Loss = {test_loss:.6f}")

    # 绘制损失曲线

    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("./data/pic/Transformer_M_Training Loss of " + EStockList[ss] + ".png", bbox_inches="tight")
    plt.show()


    '''
        8.预测和评估
    '''
    ## 预测与评估

    # 加载训练好的模型权重
    # model.load_state_dict(torch.load('model.pth'))

    # 进行预测
    # device = torch.device('cuda')
    device = torch.device('cpu')
    with torch.no_grad():
        model.eval()
        predictions = []
        labels = []
        for inputs, labels_batch in test_dataloader:
            inputs = inputs.to(device)
            predicted_batch = model(inputs)
            predictions.extend(predicted_batch.cpu().numpy())
            labels.extend(labels_batch.numpy())

    predictions = [arr[0] for arr in predictions]
    # 还原数据
    original_max = data['Close'].max()
    original_min = data['Close'].min()
    original_predictions = [pred * (original_max - original_min) + original_min for pred in predictions]
    labels = [arr[0] for arr in labels]

    # 可视化预测结果
    '''
    plt.plot(labels, label='真实值')
    plt.plot(predictions, label='预测值')
    plt.xlabel('样本索引')
    plt.ylabel('股票价格')
    plt.legend()
    plt.show()
    '''

    sns.set_style("darkgrid")

    fig = plt.figure(figsize=(14, 6))  # 宽度为 8 inches，高度为 6 inches

    plt.subplot(1, 1, 1)
    ax = sns.lineplot(x=data['Trade_Date'][:],y = data['Close'], label="Data", color='royalblue')
    ax = sns.lineplot(x=data['Trade_Date'][828:],y=original_predictions, label="Training Prediction", color='tomato')
    ax.set_title('Transformer_M:Stock price of '+EStockList[ss], size=14, fontweight='bold')
    ax.set_xlabel("Dates", size=14)
    ax.set_ylabel("Close Prices", size=14)
    plt.vlines(datetime.date(2020, 4, 22), original_min, original_max, linestyles='--', colors='gray', label='Train/Test data cut-off')
    plt.savefig("./data/pic/"+" Transformer_M_Stock price of "+EStockList[ss]+".png", bbox_inches="tight")
    plt.show()


    # 评估指标
    # 将预测值和真实值转换为 NumPy 数组
    predictions = np.array(predictions)
    labels = np.array(labels)

    # 计算均方误差（MSE）
    mse = mean_squared_error(labels, predictions)

    # 计算均方根误差（RMSE）
    rmse = np.sqrt(mse)

    # 计算平均绝对误差（MAE）
    mae = mean_absolute_error(labels, predictions)


    # 计算平均绝对百分比误差（MAPE）
    # mape = mean_absolute_percentage_error(labels, predictions)
    def MAPE(labels, predicts, mask):
        """
            Mean absolute percentage. Assumes ``y >= 0``.
            Defined as ``(y - y_pred).abs() / y.abs()``
        """
        loss = np.abs(predicts - labels) / (np.abs(labels) + 1)
        loss *= mask
        non_zero_len = mask.sum()
        return np.sum(loss) / non_zero_len


    # mape = mean_absolute_percentage_error(labels, predictions)
    real_y_true_mask = (1 - (labels == 0))
    mape = MAPE(labels, predictions, real_y_true_mask)

    # 计算相关系数（R^2 Score）
    r2 = r2_score(labels, predictions)

    # 打印评估指标的值
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("MAE:", mae)
    print("MAPE:", mape)
    print("R^2 Score:", r2)

    evaluation_metrics = {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "R^2 Score": r2
    }

    # 创建一个DataFrame对象
    df = pd.DataFrame.from_dict(evaluation_metrics, orient="index", columns=["Values"])

    # 将DataFrame保存到CSV文件
    df.to_csv("./data/results/"+StockList[ss]+"Transformer_M_evaluation_metrics.csv", encoding='utf-8-sig')
