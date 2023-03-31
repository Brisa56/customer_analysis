import matplotlib
import warnings
import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# %matplotlib inline
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams.update({'font.size' : 16})
# plt.style.use('fivethirtyeight')
plt.style.use('seaborn-pastel')

warnings.filterwarnings('ignore')

# #####任务1.1 项目背景与挖掘目标#####

# 1.1.1查看会员信息表
df_cum = pd.read_excel(".\\static\data\百货商场数据集\\cumcm2018c1.xlsx")
# print(df_cum.head())
#        会员卡号                 出生日期   性别                    登记时间
# 0  c68b20b4  2002-11-02 00:00:00  0.0 2013-05-11 00:00:00.000
# 1  1ca15332                  NaN  0.0 2004-11-04 16:31:52.436
# 2  a37cc182  1967-02-17 00:00:00  0.0 2004-12-31 21:24:34.216
# 3  2ab88539  1982-06-01 00:00:00  0.0 2010-11-19 00:00:00.000
# 4  b4c77269  1964-02-05 00:00:00  0.0 2007-12-14 00:00:00.000

# 1.1.2查看销售流水表
df_sale = pd.read_csv('.\\static\\data\百货商场数据集\\cumcm2018c2.csv')
# print(df_sale.head())
#        会员卡号                  消费产生的时间      商品编码  销售数量  ...  收银机号   单据号    柜组编码  柜组名称
# 0  1be1e3fe  2015-01-01 00:05:41.593  f09c9303     1  ...     6  25bb  8077.0   兰芝柜
# 1  1be1e3fe  2015-01-01 00:05:41.593  f09c9303     1  ...     6  25bb  8077.0   兰芝柜
# 2  1be1e3fe  2015-01-01 00:05:41.593  f09c9303     1  ...     6  25bb  8077.0   兰芝柜
# 3  1be1e3fe  2015-01-01 00:05:41.593  f09c9303     1  ...     6  25bb  8077.0   兰芝柜
# 4  1be1e3fe  2015-01-01 00:05:41.593  f09c9303     2  ...     6  25bb  8077.0   兰芝柜

# #####任务2 数据探索与预处理#####

# 任务2.1 结合业务对数据进行探索并进行预处理

# 2.1.1会员信息表数据探索与预处理
# 2.1.1.1先来对会员信息表进行分析
# print('会员信息表一共有{}行记录，{}列字段'.format(df_cum.shape[0], df_cum.shape[1]))
# print('数据缺失的情况为：\n{}'.format(df_cum.isnull().mean()))
# print('会员卡号（不重复）有{}条记录'.format(len(df_cum['会员卡号'].unique())))
# 会员信息表一共有194760行记录，4列字段
# 数据缺失的情况为：
# 会员卡号    0.000000
# 出生日期    0.175539
# 性别      0.048444
# 登记时间    0.065126
# dtype: float64
# 会员卡号（不重复）有194754条记录

# 从上面可以简要看出，数据中会员卡号存在一些重复值，且会员入会登记时间都有缺失，需要去重、去缺失值，因为性别比例缺失较少，故用众数来填补性别上的缺失值
# 存在部分会员登记时间小于出生时间，这列数据所占比例较少，可以直接进行删除
# 这个跟我下面的做法其实有区别，我并没有把重心放在“出生日期”这一字段上，因为“出生日期”缺失太多了，如果考虑进去会损失模型精度

# 2.1.1.2会员信息表去重
df_cum.drop_duplicates(subset = '会员卡号', inplace = True)
# print('会员卡号（去重）有{}条记录'.format(len(df_cum['会员卡号'].unique()))) # 会员卡号（去重）有194754条记录

# 2.1.1.3去除登记时间的缺失值，不能直接dropna，因为我们需要保留一定的数据集进行后续的LRFM建模操作
df_cum.dropna(subset = ['登记时间'], inplace = True)
# print('df_cum（去重和去缺失）有{}条记录'.format(df_cum.shape[0]))  # df_cum（去重和去缺失）有182070条记录

# 2.1.1.4性别上缺失的比例较少，所以下面采用众数填充的方法
df_cum['性别'].fillna(df_cum['性别'].mode().values[0], inplace = True)
# print(df_cum.info())
# <class 'pandas.core.frame.DataFrame'>
# Int64Index: 182070 entries, 0 to 194759
# Data columns (total 4 columns):
#  #   Column  Non-Null Count   Dtype
# ---  ------  --------------   -----
#  0   会员卡号    182070 non-null  object
#  1   出生日期    154474 non-null  object
#  2   性别      182070 non-null  float64
#  3   登记时间    182070 non-null  datetime64[ns]
# dtypes: datetime64[ns](1), float64(1), object(2)
# memory usage: 6.9+ MB

# 2.1.1.5检验是否在“登记时间”这一字段上是否存在异常值，若存在异常值，则无法进行基础的运算操作，下面操作能正常执行，说明不存在异常值
# df = df_cum['登记时间'] + pd.Timedelta(days = 1)
# print(df)

# 查看处理后数据缺失值情况
# print(df_cum.isnull().mean())
# 会员卡号    0.000000
# 出生日期    0.151568
# 性别      0.000000
# 登记时间    0.000000
# dtype: float64

# 2.1.1.6出生日期的处理
# 由于出生日期这一列的缺失值过多，且存在较多的异常值，不能贸然删除
# 故下面另建一个数据集L来保存“出生日期”和“性别”信息，方便下面对会员的性别和年龄信息进行统计
L = pd.DataFrame(df_cum.loc[df_cum['出生日期'].notnull(), ['出生日期', '性别']])
L['年龄'] = L['出生日期'].astype(str).apply(lambda x: x[:3] + '0')  # 出生年代 1960/1970/1980/....
L.drop('出生日期', axis = 1, inplace = True)
# print(L['年龄'].value_counts())

# 出生日期这列值出现较多的异常值，以一个正常人寿命为100年算起，我们假定会员年龄范围在1920-2020之间，将超过该范围的值当作异常值进行剔除
L['年龄'] = L['年龄'].astype(int)
condition = "年龄 >= 1920 and 年龄 <= 2020"
L = L.query(condition)
L.index = range(L.shape[0])
# print(L['年龄'].value_counts())
# 1980    47142
# 1970    43407
# 1960    26678
# 1990    11811
# 1950     7078
# 1940      893
# 2010      370
# 2000      139
# 1930      110
# 1920       32
# Name: 年龄, dtype: int64


# df_cum.to_csv('./static/data/vip_info1.csv', encoding = 'utf-8', index = None)


# 2.1.1.7用于与销售流水表进行合并的数据只取['会员卡号', '性别', '登记时间']这三列，将出生日期这列意义不大的进行删除（这列信息最有可能出错），并重置索引
df_cum.drop('出生日期', axis = 1, inplace = True)
df_cum.index = range(df_cum.shape[0])
# print('数据清洗之后共有{}行记录，{}列字段，字段分别为{}'.format(df_cum.shape[0], df_cum.shape[1], df_cum.columns.tolist()))
# 数据清洗之后共有182070行记录，3列字段，字段分别为['会员卡号', '性别', '登记时间']

# print(df_cum)

"""
# 2.1.2销售流水表数据探索和预处理

# 2.1.2.1数据探索
# print(df_sale.columns)
# Index(['会员卡号', '消费产生的时间', '商品编码', '销售数量', '商品售价', '消费金额', '商品名称', '此次消费的会员积分',
#        '收银机号', '单据号', '柜组编码', '柜组名称'],
#       dtype='object')
# print(df_sale.info())
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 1893532 entries, 0 to 1893531
# Data columns (total 12 columns):
 #   Column     Dtype
# ---  ------     -----
#  0   会员卡号       object
#  1   消费产生的时间    object
#  2   商品编码       object
#  3   销售数量       int64
#  4   商品售价       float64
#  5   消费金额       float64
#  6   商品名称       object
#  7   此次消费的会员积分  float64
#  8   收银机号       int64
#  9   单据号        object
#  10  柜组编码       float64
#  11  柜组名称       object
# dtypes: float64(4), int64(2), object(6)
# memory usage: 173.4+ MB

# 销售数量全部大于0
# print('销售数量大于0的记录有：{}\t 全部记录有：{}\t 两者是否相等：{}'.format(len(df_sale['销售数量'] > 0),
#                                                      df_sale.shape[0], len(df_sale['销售数量'] > 0) == df_sale.shape[0]))
#销售数量大于0的记录有：1893532	 全部记录有：1893532	 两者是否相等：True

# # 销售金额也全部大于0，说明两者不会对后者特征创造时产生影响
# print('销售金额大于0的记录有：{}\t 全部记录有：{}\t 两者是否相等：{}'.format(len(df_sale['消费金额'] > 0),
#                                                      df_sale.shape[0], len(df_sale['消费金额'] > 0) == df_sale.shape[0]))
# 销售金额大于0的记录有：1893532	 全部记录有：1893532	 两者是否相等：True


# 查看是否存在缺失值
# print(df_sale.isnull().mean())
# 会员卡号         0.537348
# 消费产生的时间      0.000000
# 商品编码         0.000000
# 销售数量         0.000000
# 商品售价         0.000000
# 消费金额         0.000000
# 商品名称         0.000000
# 此次消费的会员积分    0.537348
# 收银机号         0.000000
# 单据号          0.000000
# 柜组编码         0.537348
# 柜组名称         0.547631
# dtype: float64

# 2.1.2.2会员信息表和销售流水表这两张表唯一相关联的字段便是“会员卡号”
# 由于销售流水表中“会员卡号”有将近一半为缺失值，这类数据无法进行填充，且后续需要对会员消费记录进行统计分析和建模，故只能舍弃
df_sale_clearn = df_sale.dropna(subset = ['会员卡号'])
# print(df_sale_clearn.isnull().mean())
# 会员卡号         0.000000
# 消费产生的时间      0.000000
# 商品编码         0.000000
# 销售数量         0.000000
# 商品售价         0.000000
# 消费金额         0.000000
# 商品名称         0.000000
# 此次消费的会员积分    0.000000
# 收银机号         0.000000
# 单据号          0.000000
# 柜组编码         0.000000
# 柜组名称         0.022225
# dtype: float64

# 可以看到，舍弃掉会员卡号缺失值之后，便只有柜组名称存在缺失，下面舍弃掉一些无意义的字段，仅保留对本项目有研究价值的字段信息
# 2.1.2.3舍去无意义字段
df_sale_clearn.drop(['收银机号', '柜组编码', '柜组名称'], axis = 1, inplace = True)
# 2.1.2.4重置索引
df_sale_clearn.index = range(df_sale_clearn.shape[0])
# print(type(df_sale_clearn) == type(df_cum))  # True

# 任务2.2 将会员信息表和销售流水表关联与合并
# 2.2.1重新查看一下各个数据集的长度
# print(f'会员信息表中的记录为{len(df_cum)}\t销售流水表中的记录为{len(df_sale_clearn)}')
# 会员信息表中的记录为182070	销售流水表中的记录为876046

# 2.2.2按照会员卡号将两张表里的信息进行合并，使用左连接合并，获得一个既包含会员信息，又包含非会员信息的数据
df = pd.merge(df_sale_clearn, df_cum, on = '会员卡号', how = 'left')
# print(df)

# 2.2.3这里再次查看“消费金额”>0，“积分”>0，“销售数量”>0
index1 = df['消费金额'] > 0
index2 = df['此次消费的会员积分'] > 0
index3 = df['销售数量'] > 0
df1 = df.loc[index1 & index2 & index3, :]
df1.index = range(df1.shape[0])
# print(df1.shape)  # (738462, 11)

# 2.2.4创造一个特征字段，判断是否为会员，1表示为会员，0表示不为会员
df1['会员'] = 1
df1.loc[df1['性别'].isnull(), '会员'] = 0
# print(df1.head())






# #####任务3 统计分析#####
# 任务3.1 分析会员的年龄构成、男女比例等基本信息
# 3.1.1处理男女比例这一列，女表示0，男表示1
L['性别'] = L['性别'].apply(lambda x: '男' if x == 1 else '女')
sex_sort = L['性别'].value_counts()
# print(sex_sort)
# 女    108283
# 男     29377
# Name: 性别, dtype: int64

# 3.1.2可以将年龄划分为老年（1920-1950）、中年（1960-1990）、青年（1990-2010），再重新绘制一个饼图，
L['年龄段'] = '中年'
L.loc[L['年龄'] <= 1950, '年龄段'] = '老年'
L.loc[L['年龄'] >= 1990, '年龄段'] = '青年'
res = L['年龄段'].value_counts()
# print(res)
# 中年    117227
# 青年     12320
# 老年      8113
# Name: 年龄段, dtype: int64


# 使用上述预处理后的数据集L，包含两个字段，分别是“年龄”和“性别”，先画出年龄的条形图
fig, axs = plt.subplots(1, 2, figsize = (16, 7), dpi = 100)
# 3.1.3绘制条形图
ax = sns.countplot(x = '年龄', data = L, ax = axs[0])
# 设置数字标签
for p in ax.patches:
    height = p.get_height()
    ax.text(x = p.get_x() + (p.get_width() / 2), y = height + 500, s = '{:.0f}'.format(height), ha = 'center')
axs[0].set_title('会员的出生年代')
# 3.1.4绘制饼图
axs[1].pie(sex_sort, labels = sex_sort.index, wedgeprops = {'width': 0.4}, counterclock = False, autopct = '%.2f%%', pctdistance = 0.8)
axs[1].set_title('会员的男女比例')
plt.savefig('./static/data/会员出生年代及男女比例情况.png')

# 3.1.5绘制各个年龄段的饼图
plt.figure(figsize = (8, 6), dpi = 100)
plt.pie(res.values, labels = ['中年', '青年', '老年'], autopct = '%.2f%%', pctdistance = 0.8,
        counterclock = False, wedgeprops = {'width': 0.4})
plt.title('会员的年龄分布')
plt.savefig('./static/data/会员的年龄分布.png')





#任务3.2 分析会员的总订单占比，总消费金额占比等消费情况
#由于相同的单据号可能不是同一笔消费，以“消费产生的时间”为分组依据，我们可以知道有多少个不同的消费时间，即消费的订单数
fig, axs = plt.subplots(1, 2, figsize = (12, 7), dpi = 100)
axs[0].pie([len(df1.loc[df1['会员'] == 1, '消费产生的时间'].unique()), len(df1.loc[df1['会员'] == 0, '消费产生的时间'].unique())],
          labels = ['会员', '非会员'], wedgeprops = {'width': 0.4}, counterclock = False, autopct = '%.2f%%', pctdistance = 0.8)
axs[0].set_title('总订单占比')
axs[1].pie([df1.loc[df1['会员'] == 1, '消费金额'].sum(), df1.loc[df1['会员'] == 0, '消费金额'].sum()],
          labels = ['会员', '非会员'], wedgeprops = {'width': 0.4}, counterclock = False, autopct = '%.2f%%', pctdistance = 0.8)
axs[1].set_title('总消费金额占比')
plt.savefig('./static/data/总订单和总消费占比情况.png')

# 任务3.3 分别以季度和天为单位，分析不同时间段会员的消费时间偏好
# 消费偏好：我觉得会稍微偏向与消费的频次，相当于消费的订单数，因为每笔消费订单其中所包含的消费商品和金额都是不太一样的，有的订单所消费的商品很少，
# 但金额却很大，有的消费的商品很多，但金额却特别少。如果单纯以总金额来衡量的话，会员下次消费时间可能会很长，消费频次估计也会相对变小（因为这次所
# 购买的商品已经足够用了）。所以我会偏向于认为一个用户消费频次（订单数）越多，就越能带来更多的价值，从另一方面上来讲，用户也不可能一直都是消费低
# 端产品，消费频次越多用户的粘性也会相对比较大

# 3.3.1将会员的消费数据另存为另一个数据集
df_vip = df1.dropna()
df_vip.drop(['会员'], axis = 1, inplace = True)
df_vip.index = range(df_vip.shape[0])
# print(df_vip.info())
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 393482 entries, 0 to 393481
# Data columns (total 11 columns):
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
#  10  登记时间       393482 non-null  datetime64[ns]
# dtypes: datetime64[ns](1), float64(4), int64(1), object(5)
# memory usage: 33.0+ MB

# 将“消费产生的时间”转变成日期格式
df_vip['消费产生的时间'] = pd.to_datetime(df_vip['消费产生的时间'])
# 新增四列数据，季度、天、年份和月份的字段
df_vip['年份'] = df_vip['消费产生的时间'].dt.year
df_vip['月份'] = df_vip['消费产生的时间'].dt.month
df_vip['季度'] = df_vip['消费产生的时间'].dt.quarter
df_vip['天'] = df_vip['消费产生的时间'].dt.day
# print(df_vip.head())


# 自定义一个函数来计算2015-2018之间每个季度或月份的消费订单均数
def orders(df, label, div):
    '''
    df: 对应的数据集
    label: 为对应的列标签
    div: 为被除数
    '''
    x_list = np.sort(df[label].unique().tolist())
    order_nums = []
    for i in range(len(x_list)):
        order_nums.append(int(len(df.loc[df[label] == x_list[i], '消费产生的时间'].unique()) / div))
    return x_list, order_nums

# 前提假设：2015-2018年之间，消费者偏好在时间上不会发生太大的变化（均值），消费偏好——>以不同时间的订单数来衡量
quarters_list, quarters_order = orders(df_vip, '季度', 3)
days_list, days_order = orders(df_vip, '天', 36)
time_list = [quarters_list, days_list]
order_list = [quarters_order, days_order]
maxindex_list = [quarters_order.index(max(quarters_order)), days_order.index(max(days_order))]
fig, axs = plt.subplots(1, 2, figsize = (18, 7), dpi = 100)
colors = np.random.choice(['r', 'g', 'b', 'orange', 'y'], replace = False, size = len(axs))
titles = ['季度的均值消费偏好', '天数的均值消费偏好']
labels = ['季度', '天数']
for i in range(len(axs)):
    ax = axs[i]
    ax.plot(time_list[i], order_list[i], linestyle = '-.', c = colors[i], marker = 'o', alpha = 0.85)
    ax.axvline(x = time_list[i][maxindex_list[i]], linestyle = '--', c = 'k', alpha = 0.8)
    ax.set_title(titles[i])
    ax.set_xlabel(labels[i])
    ax.set_ylabel('均值消费订单数')
    print(f'{titles[i]}最优的时间为: {time_list[i][maxindex_list[i]]}\t 对应的均值消费订单数为: {order_list[i][maxindex_list[i]]}')
plt.savefig('./static/data/季度和天数的均值消费偏好情况.png')

# 季度的均值消费偏好最优的时间为: 2	 对应的均值消费订单数为: 19886
# 天数的均值消费偏好最优的时间为: 26	 对应的均值消费订单数为: 350

# 自定义函数来绘制不同年份之间的的季度或天数的消费订单差异
def plot_qd(df, label_y, label_m, nrow, ncol):
    '''
    # df: 为DataFrame的数据集
    # label_y: 为年份的字段标签
    # label_m: 为标签的一个列表
    # n_row: 图的行数
    # n_col: 图的列数
    '''
    # 必须去掉最后一年的数据，只能对2015-2017之间的数据进行分析
    y_list = np.sort(df[label_y].unique().tolist())[:-1]
    colors = np.random.choice(['r', 'g', 'b', 'orange', 'y', 'k', 'c', 'm'], replace = False, size = len(y_list))
    markers = ['o', '^', 'v']
    plt.figure(figsize = (8, 6), dpi = 100)
    fig, axs = plt.subplots(nrow, ncol, figsize = (16, 7), dpi = 100)
    for k in range(len(label_m)):
        m_list = np.sort(df[label_m[k]].unique().tolist())
        for i in range(len(y_list)):
            order_m = []
            index1 = df[label_y] == y_list[i]
            for j in range(len(m_list)):
                index2 = df[label_m[k]] == m_list[j]
                order_m.append(len(df.loc[index1 & index2, '消费产生的时间'].unique()))
            axs[k].plot(m_list, order_m, linestyle ='-.', c = colors[i], alpha = 0.8, marker = markers[i], label = y_list[i], markersize = 4)
        axs[k].set_xlabel(f'{label_m[k]}')
        axs[k].set_ylabel('消费订单数')
        axs[k].set_title(f'2015-2018年会员的{label_m[k]}消费订单差异')
        axs[k].legend()
    plt.savefig(f'./static/data/2015-2018年会员的{"和".join(label_m)}消费订单差异.png')

plot_qd(df_vip, '年份', ['季度', '天'], 1, 2)

# 自定义函数来绘制不同年份之间的月份消费订单差异
def plot_ym(df, label_y, label_m):
    '''
    df: 为DataFrame的数据集
    label_y: 为年份的字段标签
    label_m: 为月份的字段标签
    '''
    # 必须去掉最后一年的数据，只能对2015-2017之间的数据进行分析
    y_list = np.sort(df[label_y].unique().tolist())[:-1]
    m_list = np.sort(df[label_m].unique().tolist())
    colors = np.random.choice(['r', 'g', 'b', 'orange', 'y'], replace = False, size = len(y_list))
    markers = ['o', '^', 'v']
    fig, axs = plt.subplots(1, 2, figsize = (18, 8), dpi = 100)
    for i in range(len(y_list)):
        order_m = []
        money_m = []
        index1 = df[label_y] == y_list[i]
        for j in range(len(m_list)):
            index2 = df[label_m] == m_list[j]
            order_m.append(len(df.loc[index1 & index2, '消费产生的时间'].unique()))
            money_m.append(df.loc[index1 & index2, '消费金额'].sum())
        axs[0].plot(m_list, order_m, linestyle ='-.', c = colors[i], alpha = 0.8, marker = markers[i], label = y_list[i])
        axs[1].plot(m_list, money_m, linestyle ='-.', c = colors[i], alpha = 0.8, marker = markers[i], label = y_list[i])
        axs[0].set_xlabel('月份')
        axs[0].set_ylabel('消费订单数')
        axs[0].set_title('2015-2018年会员的消费订单差异')
        axs[1].set_xlabel('月份')
        axs[1].set_ylabel('消费金额总数')
        axs[1].set_title('2015-2018年会员的消费金额差异')
        axs[0].legend()
        axs[1].legend()
    plt.savefig('./static/data/2015-2018年会员的消费订单和金额差异.png')

# 调用函数
plot_ym(df_vip, '年份', '月份')

# 再来分析下时间上的差差异——消费订单数
df_vip['时间'] = df_vip['消费产生的时间'].dt.hour
x_list, order_nums = orders(df_vip, '时间', 1)

maxindex = order_nums.index(max(order_nums))
plt.figure(figsize = (8, 6), dpi = 100)
plt.plot(x_list, order_nums, linestyle = '-.', marker = 'o', c = 'm', alpha = 0.8)
plt.xlabel('小时')
plt.ylabel('消费订单')
plt.axvline(x = x_list[maxindex], linestyle = '--', c = 'r', alpha = 0.6)
plt.title('2015-2018年各段小时的销售订单数')
plt.savefig('./static/data/2015-2018年各段小时的销售订单数.png')

# 保存数据
df_vip.to_csv('./static/data/vip_info.csv', encoding = 'utf-8', index = None)

"""


