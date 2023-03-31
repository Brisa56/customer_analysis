# 构建会员用户业务特征标签
import base64
from io import BytesIO

import matplotlib
import warnings
import re
import pandas as pd
import numpy as np
from wordcloud import WordCloud
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# %matplotlib inline
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams.update({'font.size' : 16})
plt.style.use('ggplot')
warnings.filterwarnings('ignore')

"""
# 任务4 会员用户画像和特征字段创造
# 任务4.1 构建会员用户基本特征标签
# 导入上面保存的数据集
df_vip = pd.read_csv('./static/data/vip_info.csv', encoding ='utf-8')
# print(df_vip)

# 说明积分这一列没有存在异常值
# print(len(df_vip['此次消费的会员积分'] > 0) == df_vip.shape[0])  # True
# print(len(df_vip[df_vip['消费金额'] > 0]))  # 393482

# print(df_vip.info())
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 393482 entries, 0 to 393481
# Data columns (total 16 columns):
#  #   Column     Non-Null Count   Dtype
# ---  ------     --------------   -----
#  0   会员卡号       393482 non-null  object
#  1   消费产生的时间    393482 non-null  object
#  2   商品编码       393482 non-null  object
#  3   销售数量       393482 non-null  int64
#  4   商品售价       393482 non-null  float64
#  5   消费金额       393482 non-null  float64
#  6   商品名称       393482 non-null  object
#  7   此次消费的会员积分  393482 non-null  float64
#  8   单据号        393482 non-null  object
#  9   性别         393482 non-null  float64
#  10  登记时间       393482 non-null  object
#  11  年份         393482 non-null  int64
#  12  月份         393482 non-null  int64
#  13  季度         393482 non-null  int64
#  14  天          393482 non-null  int64
#  15  时间         393482 non-null  int64
# dtypes: float64(4), int64(6), object(6)
# memory usage: 48.0+ MB

# 查看登记时间和消费产生的时间是否存在异常值，即大于2018-01-03
# print('消费产生的时间存在异常值的数量为：{}\t登记时间存在的异常值数量为：{}'.format(len(df_vip[df_vip['消费产生的时间'] >= '2018-01-03']),
#                                                      len(df_vip[df_vip['登记时间'] >= '2018-01-03'])))
# 消费产生的时间存在异常值的数量为：469	登记时间存在的异常值数量为：36

# 筛掉两列异常时间的数据
index1 = df_vip['消费产生的时间'] < '2018-01-03'
index2 = df_vip['登记时间'] < '2018-01-03'
df_vip = df_vip[index1 & index2]
df_vip.index = range(df_vip.shape[0])
# print('筛除全部异常值之后数据的记录数为：{}\t共有{}个字段'.format(df_vip.shape[0], df_vip.shape[1]))
# 筛除全部异常值之后数据的记录数为：393006	共有16个字段

# 说明单个会员有多条消费记录数
print('会员总数：{}\t 记录数：{}'.format(len(df_vip['会员卡号'].unique()), df_vip.shape[0]))
# 会员总数：42548	 记录数：393006
# print(df_vip.columns)
# Index(['会员卡号', '消费产生的时间', '商品编码', '销售数量', '商品售价', '消费金额', '商品名称', '此次消费的会员积分',
#        '单据号', '性别', '登记时间', '年份', '月份', '季度', '天', '时间'],
#       dtype='object')

#可以先筛选每位会员，然后依据各个字段对进行运算，求出对应的LRFMP

# 自定义一个函数来实现两列数据时间相减
def time_minus(df, end_time):
    '''
    df: 为DataFrame形式，有列数据，第一列为“会员卡号”，第二列为被减的时间
    end_time: 结束时间
    '''
    df.columns = ['A', 'B']
    df['C'] = end_time
    l = pd.to_datetime(df['C']) - pd.to_datetime(df['B'])
    l = l.apply(lambda x: str(x).split(' ')[0])
    l = l.astype(int) / 30
    return l

# 开始登记的时间
df_L = df_vip.groupby('会员卡号')['登记时间'].agg(lambda x: x.values[-1]).reset_index()
# 最后一次消费的时间
df_R = df_vip.groupby('会员卡号')['消费产生的时间'].agg(lambda x: x.values[-1]).reset_index()


# 调用函数，end_time为“2018-1-3”
end_time = '2018-1-3'
L = time_minus(df_L, end_time)
R = time_minus(df_R, end_time)
# 会员消费的总次数
F = df_vip.groupby('会员卡号')['消费产生的时间'].agg(lambda x: len(np.unique(x.values))).reset_index(drop = True)
# 会员消费的总金额
M = df_vip.groupby('会员卡号')['消费金额'].agg(lambda x: np.sum(x.values)).reset_index(drop = True)
# 会员的积分总数
P = df_vip.groupby('会员卡号')['此次消费的会员积分'].agg(lambda x: np.sum(x.values)).reset_index(drop = True)

# 创造一列特征字段“消费时间偏好”（凌晨、上午、中午、下午、晚上）
'''
凌晨：0-5点
上午：6-10点
中午：11-13点
下午：14-17点
晚上：18-23点
'''

df_vip['消费时间偏好'] = df_vip['时间'].apply(lambda x: '晚上' if x >= 18 else '下午' if x >= 14 else '中午'
                                      if x >= 11 else '上午' if x >= 6 else '凌晨')
# print(df_vip)

# 会员消费的时间偏好，在多项记录中取众数
S = df_vip.groupby('会员卡号')['消费时间偏好'].agg(lambda x: x.mode().values[0]).reset_index(drop = True)

# 会员性别，取unique()
X = df_vip.groupby('会员卡号')['性别'].agg(lambda x: '女' if x.unique()[0] == 0 else '男').reset_index(drop = True)

# 开始构建对应的特征标签
df_i = pd.Series(df_vip['会员卡号'].unique())
df_LRFMPSX = pd.concat([df_i, L, R, F, M, P, S, X], axis = 1)
df_LRFMPSX.columns = ['id', 'L', 'R', 'F', 'M', 'P', 'S', 'X']
# print(df_LRFMPSX.head())
# 保存数据
df_LRFMPSX.to_csv('./static/data/LRFMPSX.csv', encoding = 'utf-8', index = None)


# 取DataFrame之后转置取values得到一个列表，再绘制对应的词云，可以自定义一个绘制词云的函数，输入参数为df和会员卡号
'''
L: 入会程度（新用户、中等用户、老用户）
R: 最近购买的时间（月）
F: 消费频数（低频、中频、高频）
M: 消费总金额（高消费、中消费、低消费）
P: 积分（高、中、低）
S: 消费时间偏好（凌晨、上午、中午、下午、晚上）
X：性别
'''
# 读取数据集
df = pd.read_csv('./static/data/LRFMPSX.csv', encoding ='utf-8')
# print(df.head())

# 查看数据的基本特征
# print(f'数据集的shape:{df.shape}')  # 数据集的shape:(42548, 8)
# print(df.isnull().mean())
# id    0.0
# L     0.0
# R     0.0
# F     0.0
# M     0.0
# P     0.0
# S     0.0
# X     0.0
# dtype: float64

# 进行描述性统计
# print(df.describe())

# 开始对数据进行分组

# L（入会程度）：3个月以下为新用户，4-12个月为中等用户，13个月以上为老用户
# R（最近购买的时间）
# F（消费频次）：次数20次以上的为高频消费，6-19次为中频消费，5次以下为低频消费
# M（消费金额）：10万以上为高等消费，1万-10万为中等消费，1万以下为低等消费
# P（消费积分）：10万以上为高等积分用户，1万-10万为中等积分用户，1万以下为低等积分用户
df_profile = pd.DataFrame()
df_profile['会员卡号'] = df['id']
df_profile['性别'] = df['X']
df_profile['消费偏好'] = df['S'].apply(lambda x: '您喜欢在' + str(x) + '时间进行消费')
df_profile['入会程度'] = df['L'].apply(lambda x: '老用户' if int(x) >= 13 else '中等用户' if int(x) >= 4 else '新用户')
df_profile['最近购买的时间'] = df['R'].apply(lambda x: '您最近' + str(int(x) * 30) + '天前进行过一次购物')
df_profile['消费频次'] = df['F'].apply(lambda x: '高频消费' if x >= 20 else '中频消费' if x >= 6 else '低频消费')
df_profile['消费金额'] = df['M'].apply(lambda x: '高等消费用户' if int(x) >= 1e+05 else '中等消费用户' if int(x) >= 1e+04 else '低等消费用户')
df_profile['消费积分'] = df['P'].apply(lambda x: '高等积分用户' if int(x) >= 1e+05 else '中等积分用户' if int(x) >= 1e+04 else '低等积分用户')
# print(df_profile)


# 保存数据
df_profile.to_csv('./static/data/consumers_profile.csv', encoding = 'utf-8', index = None)
"""

df_profile = pd.read_csv('./static/data/consumers_profile.csv', encoding = 'utf-8')

# 任务4.2 会员用户词云分析
# 开始绘制用户词云，封装成一个函数来直接显示词云
def wc_plot(df, id_label = None):
    """
    df: 为DataFrame的数据集
    id_label: 为输入用户的会员卡号，默认为随机取一个会员进行展示
    """
    myfont = './fonts/Simfang.ttf'
    if id_label == None:
        id_label = df.loc[np.random.choice(range(df.shape[0])), '会员卡号']
    text = df[df['会员卡号'] == id_label].T.iloc[:, 0].values.tolist()
    plt.figure(dpi = 100)
    wc = WordCloud(font_path = myfont, background_color = 'white', width = 500, height = 400).generate_from_text(' '.join(text))
    plt.imshow(wc)
    plt.axis('off')
    sio = BytesIO()
    plt.savefig(sio, format='png', bbox_inches='tight', pad_inches=0.0)
    data = base64.encodebytes(sio.getvalue()).decode()
    src = 'data:image/png;base64,' + str(data)
    # 记得关闭，不然画出来的图是重复的
    plt.close()
    return src
    # plt.savefig(f'./static/data/会员卡号为{id_label}的用户画像.png')
    # plt.show()
# 调用词云函数来绘制用户画像
# wc_plot(df_profile, '8527d4d0')

# 随机查找一个会员来绘制用户画像
# wc_plot(df_profile)




















"""
# 任务5 会员用户细分和营销方案制定
# 任务5.1 会员用户的聚类分析及可视化
# 先对数据进行标准化处理
df = pd.read_csv('./static/data/LRFMPSX.csv', encoding ='utf-8')
df0 = df.iloc[:, 1:6]
res_std = StandardScaler().fit_transform(df0)

# 对数据进行聚类
n_clusters = range(2, 7)
scores = []
for i in range(len(n_clusters)):
    clf = KMeans(n_clusters = n_clusters[i], random_state = 20).fit(res_std)
    scores.append(silhouette_score(res_std, clf.labels_))
maxindex = scores.index(max(scores))
plt.figure(figsize = (8, 6), dpi = 100)
plt.plot(n_clusters, scores, linestyle = '-.', c = 'b', alpha = 0.6, marker = 'o')
plt.axvline(x = n_clusters[maxindex], linestyle = '--', c = 'r', alpha = 0.5)
plt.title('LRFMP的聚类轮廓系数图')
plt.ylabel('silhouette_score')
plt.xlabel('n_clusters')
plt.savefig('./static/data/LRFMP聚类轮廓系数图.png')

# 构造一个绘制聚类可视化效果雷达图的函数
def plot(features, clf_list, nrow, ncol, title):
    '''
    features: 字段名
    clf_list：list，为聚类器列表
    nrow: 图的行数
    ncol: 图的列数
    title: 图的名称
    '''
    N = len(features)
    angles = np.linspace(0, 2 * np.pi, N, endpoint = False)
    angles = np.concatenate([angles, [angles[0]]])
    features = np.concatenate([features, [features[0]]])
    fig = plt.figure(figsize = (14, 14), dpi = 100)
    for i in range(len(clf_list)):
        clf = clf_list[i]
        centers = clf.cluster_centers_
        # add_subplot的index从1开始
        ax = fig.add_subplot(nrow, ncol, i + 1, polar = True)
        ax.set_thetagrids(angles * 180 / np.pi, features)
        # 随机取不同的颜色
        colors = np.random.choice(['r', 'g', 'b', 'y', 'k', 'orange'], replace = False, size = len(centers))
        for j in range(len(centers)):
            values = np.concatenate([centers[j, :], [centers[j, :][0]]])
            ax.plot(angles, values, c = colors[j], alpha = 0.6, linestyle = '-.', label = '类别' + str(j + 1))
            ax.fill(angles, values, c = colors[j], alpha = 0.2)
        ax.set_title(f'n_clusters = {len(centers)}')
        ax.legend()
    plt.suptitle(title)
    plt.savefig(f'./static/data/{title}.png')


features = list('LRFMP')
res_std = StandardScaler().fit_transform(df0)
res_mm = MinMaxScaler().fit_transform(df0)
res = [res_std, res_mm]
titles = ['标准化处理后的聚类雷达图', '归一化处理后的聚类雷达图']
for i in range(len(res)):
    clf = []
    for j in range(2, 6):
        clf.append(KMeans(n_clusters = j, random_state = 20).fit(res[i]))
    plot(features, clf, 2, 2, titles[i])

# 从上面可以看出，标准化后的数据聚类效果相较于归一化的更好，且从轮廓系数和聚类雷达图也可以看出，聚类数最佳为2。因此，下面我们使用聚类数为2的标准化数据进行聚类，得到两类客户的LRFMP均值数据，以此来判断两者之间的差异
# 任务5.2 对会员用户进行精细划分并分析不同群体带来的价值差异
# 以聚类数为2贴上对应的标签
clf = KMeans(n_clusters = 2, random_state = 20).fit(res_std)
df0['labels'] = clf.labels_
# print(df0)

# 统计一下两类用户之间的差异，发现两类客户之间数量相差过大
# print(f"类别0所占比例为：{df0['labels'].value_counts().values[0] / df0.shape[0]} \t 类别1所占的比例为：\
#       {df0['labels'].value_counts().values[1] / df0.shape[0]}")
#类别0所占比例为：0.9770141957318793 	 类别1所占的比例为：0.02298580426812071
# print(df0['labels'].value_counts())
# 0    41570
# 1      978
# Name: labels, dtype: int64

# 用均值来计算两类样本之间的LRFMP
L_avg = df0.groupby('labels').agg({'L': np.mean}).reset_index()
R_avg = df0.groupby('labels').agg({'R': np.mean}).reset_index()
F_avg = df0.groupby('labels').agg({'F': np.mean}).reset_index()
M_avg = df0.groupby('labels').agg({'M': np.mean}).reset_index()
P_avg = df0.groupby('labels').agg({'P': np.mean}).reset_index()


# 绘制相关的条形图
def plot_bar(df_list, nrow, ncol):
    fig, axs = plt.subplots(nrow, ncol, figsize = (2 * (ncol + 2), 2.5), dpi = 100)
    for i in range(len(axs)):
        ax = axs[i]
        df = df_list[i]
        ax.bar(df.iloc[:, 0], df.iloc[:, 1], color = 'm', alpha = 0.4, width = 0.5)
        for x, y in enumerate(df.iloc[:, 1].tolist()):
            ax.text(x, y / 2, '%.0f' % y, va = 'bottom', ha = 'center', fontsize = 12)
        ax.set_xticks([0, 1])
        ax.set_yticks(())
        ax.set_title(f'{df.columns[1]}')
    plt.suptitle('两类客户的LRFMP均值差异', y = 1.1, fontsize = 14)
    plt.savefig('./static/data/两类客户的LRFMP均值差异.png')

df_list = [L_avg, R_avg, F_avg, M_avg, P_avg]
plot_bar(df_list, 1, 5)
# 从上面可以看出，标签为1的客户消费频次、消费金额和消费积分均远大于标签为0的客户，且这类客户所占的比例仅有2.3%，可以将其定义为“重要保持会员”。标签为0的客户所占比例为97.7%，其会员登记时间跟标签为1的比较接近，但最近一次消费时间较标签1的还要长，可以将其定义为“一般发展会员”
"""