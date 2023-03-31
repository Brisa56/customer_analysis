import pandas as pd
import numpy as np
from datetime import datetime


# 顾客基本属性数据
#df_cum_info = pd.read_excel("F:\作业\学习\\04大四\毕业设计\\02数据集\百货商场\\cumcm2018c1.xlsx")
# 顾客消费数据
df_cum_cons = pd.read_csv("F:\作业\学习\\04大四\毕业设计\\02数据集\百货商场\\cumcm2018c2.csv")


# 先来对会员信息表进行分析
# print('会员信息表一共有{}行记录，{}列字段'.format(df_cum_info.shape[0], df_cum_info.shape[1]))
# print('数据缺失的情况为：\n{}'.format(df_cum_info.isnull().mean()))
# print('会员卡号（不重复）有{}条记录'.format(len(df_cum_info['会员卡号'].unique())))
"""
会员信息表一共有194760行记录，4列字段
数据缺失的情况为：
会员卡号    0.000000
出生日期    0.175539
性别      0.048444
登记时间    0.065126
dtype: float64
会员卡号（不重复）有194754条记录
"""

# # 1.会员信息表去重
# df_cum_info.drop_duplicates(subset = '会员卡号', inplace = True)
# # print('会员卡号（去重）有{}条记录'.format(len(df_cum_info['会员卡号'].unique())))  # 会员卡号（去重）有194754条记录
#
# # 2.去除登记时间的缺失值，不能直接dropna，因为我们需要保留一定的数据集进行后续的LRFM建模操作
# df_cum_info.dropna(subset = ['登记时间'], inplace = True)
# # 转换日期格式 2013-05-11
# df_cum_info['登记时间'] = df_cum_info['登记时间'].apply(lambda x: x.strftime("%Y-%m-%d"))
# # print('df_cum（去重和去缺失）有{}条记录'.format(df_cum_info.shape[0]))  # df_cum_info（去重和去缺失）有182070条记录
# # print(df_cum_info['登记时间'])
#
#
# # 3.性别上缺失的比例较少，所以下面采用众数填充的方法
# df_cum_info['性别'].fillna(df_cum_info['性别'].mode().values[0], inplace = True)
# # df_cum_info.info()
# """
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
# """
#
#
# # 4.由于出生日期这一列的缺失值过多，且存在较多的异常值，不能贸然删除
# # 故下面另建一个数据集L来保存“出生日期”和“性别”信息，方便下面对会员的性别和年龄信息进行统计
# df_cum_info['出生日期'] = pd.to_datetime(df_cum_info['出生日期'],errors='coerce')
# L = pd.DataFrame(df_cum_info.loc[df_cum_info['出生日期'].notnull(), ['会员卡号','出生日期', '性别','登记时间']])
# # print(type(L['出生日期'][0]))  # <class 'datetime.datetime'>
# # print(L['出生日期'])  # 2002-11-02 00:00:00
# L['出生年代'] = L['出生日期'].astype(str).apply(lambda x: x[:3] + '0')  # 出生年代
# # L.drop('出生日期', axis = 1, inplace = True)
# # print(L['出生年代'].value_counts())   # currentYear = datetime.now().year
# # L.info()
# """
# <class 'pandas.core.frame.DataFrame'>
# Int64Index: 154474 entries, 0 to 194759
# Data columns (total 4 columns):
#  #   Column  Non-Null Count   Dtype
# ---  ------  --------------   -----
#  0   会员卡号    154474 non-null  object
#  1   性别      154474 non-null  float64
#  2   登记时间    154474 non-null  object
#  3   出生年代    154474 non-null  object
# dtypes: float64(1), object(3)
# memory usage: 5.9+ MB
# """
#
# # 5.导出会员基本信息数据
# df_cum_info.to_csv("F:\作业\学习\\04大四\毕业设计\\02数据集\百货商场\\new1_cumcm2018c1.csv")
# L.to_csv("F:\作业\学习\\04大四\毕业设计\\02数据集\百货商场\\new2_cumcm2018c1.csv")


# 1.删除会员卡号为空的
# print(df_cum_cons.info())
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 1048575 entries, 0 to 1048574
# Data columns (total 12 columns):
#  #   Column     Non-Null Count    Dtype
# ---  ------     --------------    -----
#  0   会员卡号       483183 non-null   object
#  1   消费产生的时间    1048575 non-null  object
#  2   商品编码       1048575 non-null  object
#  3   销售数量       1048575 non-null  int64
#  4   商品售价       1048575 non-null  float64
#  5   消费金额       1048575 non-null  float64
#  6   商品名称       1048575 non-null  object
#  7   此次消费的会员积分  483183 non-null   float64
#  8   收银机号       1048575 non-null  int64
#  9   单据号        1048575 non-null  object
#  10  柜组编码       483183 non-null   float64
#  11  柜组名称       475202 non-null   object
# dtypes: float64(4), int64(2), object(6)
# memory usage: 96.0+ MB
