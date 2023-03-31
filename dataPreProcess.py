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
plt.style.use('ggplot')
warnings.filterwarnings('ignore')


# #####任务1.1 项目背景与挖掘目标#####

# 1.1.1查看会员信息表
df_cum = pd.read_excel(".\data\百货商场数据集\\cumcm2018c1.xlsx")
# print(df_cum.head())
#        会员卡号                 出生日期   性别                    登记时间
# 0  c68b20b4  2002-11-02 00:00:00  0.0 2013-05-11 00:00:00.000
# 1  1ca15332                  NaN  0.0 2004-11-04 16:31:52.436
# 2  a37cc182  1967-02-17 00:00:00  0.0 2004-12-31 21:24:34.216
# 3  2ab88539  1982-06-01 00:00:00  0.0 2010-11-19 00:00:00.000
# 4  b4c77269  1964-02-05 00:00:00  0.0 2007-12-14 00:00:00.000

# 1.1.2查看销售流水表
df_sale = pd.read_csv('static/data\百货商场数据集\百货商场数据集\\cumcm2018c2.csv')
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

# 2.1.1.7年龄段的处理
# 可以将年龄划分为老年（1920-1950）、中年（1960-1990）、青年（1990-2010），再重新绘制一个饼图，
L['年龄段'] = '中年'
L.loc[L['年龄'] <= 1950, '年龄段'] = '老年'
L.loc[L['年龄'] >= 1990, '年龄段'] = '青年'
res = L['年龄段'].value_counts()
# print(res)
# 中年    117227
# 青年     12320
# 老年      8113
# Name: 年龄段, dtype: int64

# 2.1.1.8处理男女比例这一列，女表示0，男表示1
L['性别'] = L['性别'].apply(lambda x: '男' if x == 1 else '女')
sex_sort = L['性别'].value_counts()
# print(sex_sort)
# 女    108283
# 男     29377
# Name: 性别, dtype: int64

# L  ['年龄','性别','年龄段']



# 2.1.1.9用于与销售流水表进行合并的数据只取['会员卡号', '性别', '登记时间']这三列，将出生日期这列意义不大的进行删除（这列信息最有可能出错），并重置索引
df_cum.drop('出生日期', axis = 1, inplace = True)
df_cum.index = range(df_cum.shape[0])
# print('数据清洗之后共有{}行记录，{}列字段，字段分别为{}'.format(df_cum.shape[0], df_cum.shape[1], df_cum.columns.tolist()))
# 数据清洗之后共有182070行记录，3列字段，字段分别为['会员卡号', '性别', '登记时间']

# print(df_cum)


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
# 会员卡号	消费产生的时间	商品编码	销售数量	商品售价	消费金额	商品名称	此次消费的会员积分	单据号	性别	登记时间	会员
# 0	1be1e3fe	2015-01-01 00:05:41.593	f09c9303	1	290.0	270.20	兰芝化妆品正价瓶	270.20	25bb	NaN	NaT	0
# 1	1be1e3fe	2015-01-01 00:05:41.593	f09c9303	1	325.0	302.80	兰芝化妆品正价瓶	302.80	25bb	NaN	NaT	0
# 2	1be1e3fe	2015-01-01 00:05:41.593	f09c9303	1	195.0	181.80	兰芝化妆品正价瓶	181.80	25bb	NaN	NaT	0
# 3	1be1e3fe	2015-01-01 00:05:41.593	f09c9303	1	270.0	251.55	兰芝化妆品正价瓶	251.55	25bb	NaN	NaT	0
# 4	1be1e3fe	2015-01-01 00:05:41.593	f09c9303	2	245.0	456.55	兰芝化妆品正价瓶	456.55	25bb	NaN	NaT	0


# # 3.3.1将会员的消费数据另存为另一个数据集
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


df_vip['时间'] = df_vip['消费产生的时间'].dt.hour
# 保存数据
df_vip.to_csv('./data/vip_info.csv', encoding = 'gb18030', index = None)
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
    """
    df: 为DataFrame形式，有列数据，第一列为“会员卡号”，第二列为被减的时间
    end_time: 结束时间
    """
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
"""
凌晨：0-5点
上午：6-10点
中午：11-13点
下午：14-17点
晚上：18-23点
"""
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
df_LRFMPSX.to_csv('./data/LRFMPSX.csv', encoding = 'gb18030', index = None)


# 取DataFrame之后转置取values得到一个列表，再绘制对应的词云，可以自定义一个绘制词云的函数，输入参数为df和会员卡号
"""
L: 入会程度（新用户、中等用户、老用户）
R: 最近购买的时间（月）
F: 消费频数（低频、中频、高频）
M: 消费总金额（高消费、中消费、低消费）
P: 积分（高、中、低）
S: 消费时间偏好（凌晨、上午、中午、下午、晚上）
X：性别
"""
# 读取数据集
df = pd.read_csv('static/data/LRFMPSX.csv', encoding ='gbk')
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
"""
L（入会程度）：3个月以下为新用户，4-12个月为中等用户，13个月以上为老用户
R（最近购买的时间）
F（消费频次）：次数20次以上的为高频消费，6-19次为中频消费，5次以下为低频消费
M（消费金额）：10万以上为高等消费，1万-10万为中等消费，1万以下为低等消费
P（消费积分）：10万以上为高等积分用户，1万-10万为中等积分用户，1万以下为低等积分用户
"""
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
df_profile.to_csv('./data/consumers_profile.csv', encoding = 'gb18030', index = None)






