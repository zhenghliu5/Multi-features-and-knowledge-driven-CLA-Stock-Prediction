#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : Zhenghliu5
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

StockList = ['中国人寿','中国平安','五粮液','贵州茅台','工商银行','招商银行']  # 待预测的股票列表
EStockList = ['China Life Insurance', 'Ping An of China', 'Wuliangye', 'Kweichow Moutai', 'ICBC', 'China Merchants Bank']



for ss in range(len(StockList)):
    # 读取股票数据CSV文件
    data = pd.read_csv('./data/'+StockList[ss]+'_预处理后.csv', encoding='utf-8-sig')
    df = pd.DataFrame(data.fillna(method='ffill'))
    print(df.head(10))

    # 计算移动平均线（MA）
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_7'] = df['Close'].rolling(window=7).mean()
    df['MA_21'] = df['Close'].rolling(window=21).mean()

    # Create Bollinger Bands
    df['20sd'] = df['Close'].rolling(window=20).std()
    df['upper_band'] = df['MA_21'] + (df['20sd'] * 2)
    df['lower_band'] = df['MA_21'] - (df['20sd'] * 2)


    # 计算相对强弱指标（RSI）
    delta = df['Close'].diff()
    gain = delta.mask(delta < 0, 0)
    loss = -delta.mask(delta > 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))


    # 计算MACD指标
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()


    # 威廉指标（Williams %R）
    highest_high = df['High'].rolling(window=14).max()
    lowest_low = df['Low'].rolling(window=14).min()
    df['W%R'] = (highest_high - df['Close']) / (highest_high - lowest_low) * -100

    # 随机指标（Stochastic Oscillator）
    lowest_low = df['Low'].rolling(window=14).min()
    highest_high = df['High'].rolling(window=14).max()
    df['K'] = (df['Close'] - lowest_low) / (highest_high - lowest_low) * 100
    df['D'] = df['K'].rolling(window=3).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']

    # 动向指数（Directional Movement Index，DMI）
    df['up_move'] = df['High'].diff()
    df['down_move'] = df['Low'].diff().abs()
    df['plus_dm'] = np.where(df['up_move'] > df['down_move'], df['up_move'], 0)
    df['minus_dm'] = np.where(df['down_move'] > df['up_move'], df['down_move'], 0)
    df['TR'] = df[['High', 'Low', 'Close']].apply(lambda x: max(x) - min(x), axis=1)
    df['plus_di'] = 100 * (df['plus_dm'].rolling(window=14).mean() / df['TR'].rolling(window=14).mean())
    df['minus_di'] = 100 * (df['minus_dm'].rolling(window=14).mean() / df['TR'].rolling(window=14).mean())
    df['DX'] = 100 * (df['plus_di'] - df['minus_di']).abs() / (df['plus_di'] + df['minus_di'])
    df['ADX'] = df['DX'].rolling(window=14).mean()
    df['ADXR'] = (df['ADX'] + df['ADX'].shift(14)) / 2

    # Create Momentum
    df['momentum'] = df['Close'] - 1


    # 存储技术指标数据到新的CSV文件
    df.to_csv('./data/'+StockList[ss]+'technical_indicators.csv', encoding='utf-8-sig', index=False)
    print("第" + str(ss) + "个已完成")


    # 绘制技术指标曲线
    last_days = len(df)-800
    plt.figure(figsize=(16, 10), dpi=100)
    shape_0 = df.shape[0]
    xmacd_ = shape_0 - last_days

    df = df.iloc[-last_days:, :]
    x_ = range(3, df.shape[0])
    x_ = list(df.index)

    # Plot first subplot
    plt.subplot(2, 1, 1)
    plt.plot(df['MA_7'], label='MA 7', color='g', linestyle='--')
    plt.plot(df['Close'], label='Closing Price', color='b')
    plt.plot(df['MA_21'], label='MA 21', color='r', linestyle='--')
    plt.plot(df['upper_band'], label='Upper Band', color='c')
    plt.plot(df['lower_band'], label='Lower Band', color='c')
    plt.fill_between(x_, df['lower_band'], df['upper_band'], alpha=0.35)
    plt.title('Technical indicators for '+ EStockList[ss] + '- last {} days.'.format(last_days))
    plt.ylabel('Price')
    plt.legend()

    # Plot second subplot
    plt.subplot(2, 1, 2)
    plt.title('MACD')
    plt.plot(df['MACD'], label='MACD', linestyle='-.')
    plt.hlines(15, xmacd_, shape_0, colors='g', linestyles='--')
    plt.hlines(-15, xmacd_, shape_0, colors='g', linestyles='--')
    plt.plot(df['momentum'], label='Momentum', color='b', linestyle='-')

    plt.legend()
    plt.savefig("./data/pic/Technical indicators for  " + EStockList[ss] + ".png", bbox_inches="tight")
    plt.show()



