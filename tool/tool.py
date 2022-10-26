import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

from tqdm import tqdm

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge

import lightgbm as lgb
import xgboost as xgb

def describe(df):
    stats = []
    for col in df.columns:
        stats.append((
            col, # 特征名称
            df[col].nunique(), # 属性个数
            round(df[col].isnull().sum() * 100 / df.shape[0], 3), # 缺失值占比
            round(df[col].value_counts(normalize=True, dropna=False).values[0] * 100, 3), # 最大属性占比
            df[col].dtype # 特征类型
        ))
    
    return pd.DataFrame(stats, columns=['特征','属性个数','缺失值占比','最大属性占比','特征类型']).sort_values('缺失值占比',ascending=False)


def plot_boxplot(x,y,figsize=(12,8)):
    fig = plt.figure(figsize=figsize)  # 指定绘图对象宽度和高度
    sns.boxplot(x,y,orient="v", width=0.5)


# 直方图和QQ图
def plot_qq(data,figsize=(10,5)):
    plt.figure(figsize=figsize)
    ax=plt.subplot(1,2,1)
    sns.distplot(data,fit=stats.norm)
    ax=plt.subplot(1,2,2)
    res = stats.probplot(data, plot=plt)

# 对比分布
def comparison(data_train,data_test):
    ax = sns.kdeplot(data_train, color="Red", shade=True)
    ax = sns.kdeplot(data_test, color="Blue", shade=True)
    ax = ax.legend(["train","test"])

# rolling特征函数
def fe_rolling_stat(data, time_col,time_varying_cols, window_size):
    """
    :param df: DataFrame原始数据
    :param time_varying_cols: time varying columns 需要rolling的列
    :param window_size: window size 窗口大小
    :return: DataFrame rolling features
    """
    df = data.copy() 
    result = pd.DataFrame({time_col:df[time_col]},index=df[time_col].index)
    
    add_feas = []
    for cur_ws in tqdm(window_size):
        for val in time_varying_cols:
            for op in ['mean','std','median','max','min','kurt','skew']:
                name = f'{val}__{cur_ws}h__{op}'
                if op == 'mean':
                    df[name] = df[val].transform(
                        lambda x: x.rolling(window=cur_ws).mean())
                    add_feas.append(name)
                if op == 'std':
                    df[name] = df[val].transform(
                        lambda x: x.rolling(window=cur_ws).std())
                    add_feas.append(name)
                if op == 'median':
                    df[name] = df[val].transform(
                        lambda x: x.rolling(window=cur_ws).median())
                    add_feas.append(name)
                if op == 'max':
                    df[name] = df[val].transform(
                        lambda x: x.rolling(window=cur_ws).max())
                    add_feas.append(name)
                if op == 'min':
                    df[name] = df[val].transform(
                        lambda x: x.rolling(window=cur_ws).min())
                    add_feas.append(name)
                if op == 'kurt' and cur_ws == 24:  # 只在rolling的24h计算峰度和下面的偏度
                    df[name] = df[val].transform(
                        lambda x: x.rolling(window=cur_ws).kurt())
                    add_feas.append(name)
                if op == 'skew' and cur_ws == 24:
                    df[name] = df[val].transform(
                        lambda x: x.rolling(window=cur_ws).skew())
                    add_feas.append(name)
    return result.merge(df[[time_col,] + add_feas], on = [time_col], how = 'left')[add_feas]   # 拼接result表格和df[]表格
    # how = left，保留所有左表;result的信息


# max - min
def fe_max_min(data, time_col,time_varying_cols):
    """
    构造最大最小值
    :param df: DataFrame
    :param time_col: time column
    :param time_varying_cols: time varying columns
    :param lag: lag list
    :return: DataFrame lag
    """
    df = data.copy()
    result = pd.DataFrame({time_col:df[time_col].values},index=df.index)
    add_fetures = []
    for column in time_varying_cols:
        name = f'{column}_max_min'
        add_fetures.append(name)
        df[name] = df.groupby(["Day"])[column].transform(lambda x: (x.max() - x.min()))
    return result.merge(df[[time_col,] + add_fetures], on=time_col, how='left')[add_fetures]

# 构造滞后特征
def fe_lag(data, time_col,time_varying_cols, lags):
    """
    滞后特征
    :param df: DataFrame
    :param time_col: time column
    :param time_varying_cols: time varying columns
    :param lag: lag list
    :return: DataFrame lag features
    """
    df = data.copy()
    result = pd.DataFrame({time_col:df[time_col].values},index=df.index)
    add_fetures = []
    for column in time_varying_cols:
        for lag in lags:
            name = f'{column}_lag_{lag}'
            add_fetures.append(name)
            df[name] = df[column].shift(lag)
    
    return result.merge(df[[time_col,] + add_fetures], on=time_col, how='left')[add_fetures]

# 构造差分特征
def fe_diff(data, time_col,time_varying_cols, lags):
    """
    构造差分特征
    lag0 - lag1
    lag0 - lag2
    lag0 - lag3
    :param df: DataFrame
    :param time_col: time column
    :param time_varying_cols: time varying columns
    :param lag: lag list
    :return: DataFrame lag
    """
    df = data.copy()
    result = pd.DataFrame({time_col:df[time_col].values},index=df.index)
    add_fetures = []
    for column in time_varying_cols:
        for lag in lags:
            name = f'{column}_diff_{lag}'
            add_fetures.append(name)
            df[name] = df[column].diff(lag)
    
    return result.merge(df[[time_col,] + add_fetures], on=time_col, how='left')[add_fetures]

# 构造差除特征
def fe_div(data, time_col,time_varying_cols,lags):    
    """
    构造一阶差分特征,使用不同的lag
    注意不要有0值
    lag0/lag1
    lag0/lag2
    lag0/lag3
    :param df: DataFrame
    :param time_col: time column
    :param time_varying_cols: time varying columns
    :param lag: lag list
    :return: DataFrame lag
    """
    df = data.copy()
    result = pd.DataFrame({time_col:df[time_col].values},index=df.index)
    add_fetures = []
    for column in time_varying_cols:
        for lag in lags:
            name = f'{column}_div_{lag}'
            add_fetures.append(name)
            df[name] = np.divide(df[column], df[column].shift(1))
    return result.merge(df[[time_col,] + add_fetures], on=time_col, how='left')[add_fetures]




# groupby特征构造
def fe_groupby(data, time_col, by, time_varying_cols):
    df = data.copy()
    result = pd.DataFrame({time_col:df[time_col].values},index=df.index)
    add_fetures = []
    for val in time_varying_cols:
        for op in ['mean','std','median','max','min','skew']:
            name = f'{val}__{by[0]}_{by[1]}__{op}'
            add_fetures.append(name)
            df[name] = df.groupby(by)[val].transform(op)
    
    return result.merge(df[[time_col,] + add_fetures], on=time_col, how='left')[add_fetures]

def feature_combination(df_list):
    """
    特征融合
    :param df_list: DataFrame list; 每个DataFrame的第一列特征是“date”
    :return: DataFrame
    """
    result = df_list[0]
    for df in tqdm(df_list[1:], total=len(df_list[1:])):
        if df is None or df.shape[0] == 0:
            continue

        assert (result.shape[0] == df.shape[0])
        result = pd.concat([result, df], axis=1)
        print(result.shape[0], df.shape[0])
    return result


# 绘制训练时函数图
def train_plot(train_loss,val_loss):
    log_path = ''  # 存储训练图片的地址
    plt.figure()
    iters = len(train_loss)
    plt.plot(range(1,iters+1), train_loss, 'red', linewidth=2, label = 'train_loss')
    plt.plot(range(1,iters+1), val_loss, 'green', linewidth=2, label = 'val_loss')

    plt.legend(loc='upper right')
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.savefig(os.path.join(log_path, "epoch_loss.png"))
    plt.cla()
    plt.close('all')

# lgb训练函数
def train_lgb(params, X_train, y_train, n_splits, model_dir):
    train_loss = []
    val_loss = []
    print('Start Train')
    # K折交叉验证
    folds = KFold(n_splits=n_splits)
    for fold_, (trn_idx, val_idx) in tqdm(enumerate(folds.split(X_train.values, y_train.values)),total=n_splits):
        trn_data = lgb.Dataset(X_train.iloc[trn_idx],label=y_train.iloc[trn_idx])
        
        val_data = lgb.Dataset(X_train.iloc[val_idx],label=y_train.iloc[val_idx])
        

        num_round = 5000
        clf = lgb.train(params,
                        trn_data,
                        num_round,
                        valid_sets = [trn_data, val_data],
                        verbose_eval=0,
                        early_stopping_rounds = 200,
                        )
        
        val_pred = clf.predict(X_train.iloc[val_idx], num_iteration=clf.best_iteration)
        train_pred = clf.predict(X_train.iloc[trn_idx], num_iteration=clf.best_iteration)

        train_metric = mean_squared_error(y_train[trn_idx],train_pred)
        train_loss.append(train_metric)

        val_metric = mean_squared_error(y_train[val_idx],val_pred)
        val_loss.append(val_metric)

        train_plot(train_loss, val_loss)
        # print(cur_metric)
        file_name = '{}_model_mse_{}'.format(fold_, val_metric)
        clf.save_model(f'{model_dir}/{file_name}.m')
    print('Finish Train')
    print("训练集_mse:{:.4f}\n测试集_mse:{:.4f}".format(np.mean(np.array(train_loss)), np.mean(np.array(val_loss))))


class BasicModel(object):
    """Parent class of basic models"""
    def train(self, x_train, y_train, x_val, y_val):
        """return a trained model and eval metric of validation data"""
        pass
    
    def predict(self, model, x_test):
        """return the predicted result of test data"""
        pass
    
    def get_pred(self, x_train, y_train, x_test, n_folds = 10):
        """K-fold stacking"""
        num_train, num_test = x_train.shape[0], x_test.shape[0]
        train_pred = np.zeros((num_train,)) 
        test_pred = np.zeros((num_test,))
        losses = []
        folds = KFold(n_splits = n_folds)
        with tqdm(enumerate(folds.split(x_train.values, y_train.values)),total=n_folds,postfix=dict) as par:
            for i, (train_index, val_index) in par:
                # 获取每一折训练集和测试集
                x_tra, y_tra = x_train.loc[train_index], y_train.loc[train_index]
                x_val, y_val = x_train.loc[val_index], y_train.loc[val_index]

                model = self.train(x_tra, y_tra, x_val, y_val)
                train_pred[val_index] = self.predict(model, x_val)
                loss = mean_squared_error(y_train.loc[val_index],train_pred[val_index])
                losses.append(loss)

                test_pred += self.predict(model, x_test) / n_folds
            print('all mse {0}, average {1}'.format(losses, np.mean(losses)))
            return train_pred, test_pred



class LGBRegression(BasicModel):
    def __init__(self, parmes, num_boost_round=10000, early_stopping_rounds=200):
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.params = parmes
        
    def train(self, x_train, y_train, x_val, y_val):
        trn_data = lgb.Dataset(x_train, y_train)
        val = lgb.Dataset(x_val, y_val)
        model = lgb.train(self.params, 
                            trn_data,
                            valid_sets = val,
                            verbose_eval = self.num_boost_round,
                            num_boost_round = self.num_boost_round,
                            early_stopping_rounds = self.early_stopping_rounds
                            )
        
        return model
    
    def predict(self, model, x_test):
        return model.predict(x_test, num_iteration=model.best_iteration)


class XGBRegression(BasicModel):
    def __init__(self, params,num_rounds=10000, early_stopping_rounds = 200):
        """set parameters"""
        self.num_rounds=num_rounds
        self.early_stopping_rounds = early_stopping_rounds
        self.params = params
        
        
    def train(self, x_train, y_train, x_val, y_val):
        train_data = xgb.DMatrix(x_train, y_train)
        val_data = xgb.DMatrix(x_val, y_val)
        watchlist = [(train_data,'train'), (val_data, 'val')]
        model = xgb.train(self.params, 
                          train_data, 
                          self.num_rounds,
                          watchlist,
                          early_stopping_rounds = self.early_stopping_rounds)
        return model
 
    def predict(self, model, x_test):
        xgbtest = xgb.DMatrix(x_test)
        return model.predict(xgbtest)

class RidgeRegression(BasicModel):
    def __init__(self, alpha = 1.0):
        """set parameters"""
        self.alpha = alpha
    
    def train(self, x_train, y_train, x_val, y_val):
        return Ridge(alpha=self.alpha).fit(x_train,y_train)
    
    def predict(self, model, x_test):
        return model.predict(x_test)