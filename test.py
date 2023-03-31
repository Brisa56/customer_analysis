import json

import pandas as pd
# customerId = 'd8d36575'
# salesName = 'MARYLINGC件'
# salesTime = '2015-01-01'

customerId = ''
salesName = ''
salesTime = ''
df_vip = pd.read_csv('./static/data/vip_info.csv')
if customerId:
    if salesName:
        if salesTime:
            r = df_vip.loc[(df_vip['会员卡号'] == customerId) &
                           (df_vip['商品名称'].str.contains(salesName)) &
                           (df_vip['消费产生的时间'].str.contains(salesTime))]
        else:
            r = df_vip.loc[(df_vip['会员卡号'] == customerId) &
                           (df_vip['商品名称'].str.contains(salesName))]
    else:
        if salesTime:
            r = df_vip.loc[(df_vip['会员卡号'] == customerId) &
                           (df_vip['消费产生的时间'].str.contains(salesTime))]
        else:
            r = df_vip.loc[(df_vip['会员卡号'] == customerId)]
else:
    if salesName:
        if salesTime:
            r = df_vip.loc[(df_vip['商品名称'].str.contains(salesName)) &
                           (df_vip['消费产生的时间'].str.contains(salesTime))]
        else:
            r = df_vip.loc[(df_vip['商品名称'].str.contains(salesName))]
    else:
        if salesTime:
            r = df_vip.loc[(df_vip['消费产生的时间'].str.contains(salesTime))]
        else:
            r = df_vip
# if customerId:
#     if salesName:
#         if salesTime:
#             r = df_vip.loc[(df_vip['会员卡号']==customerId) &
#                            (df_vip['消费产生的时间'].str.contains(salesTime)) &
#                            (df_vip['商品名称'].str.contains(salesName))]
#             print(type(r))
customer_vip = {}
customer_vip['code'] = 0
customer_vip['msg'] = ''
customer_vip['count'] = r.shape[0]
data = json.dumps(r.to_dict(orient="records"), sort_keys=True, indent=4, separators=(', ', ': '), ensure_ascii=False)
customer_vip['data'] = data

print(customer_vip)

# {
#         "会员卡号": "b456b18c",
#         "单据号": "2954",
#         "商品名称": "Juicy Couyure F件",
#         "商品售价": 597.0,
#         "商品编码": "b264e578",
#         "天": 3,
#         "季度": 1,
#         "年份": 2018,
#         "性别": 0.0,
#         "时间": 21,
#         "月份": 1,
#         "此次消费的会员积分": 597.0,
#         "消费产生的时间": "2018-01-03 21:20:24.783",
#         "消费金额": 597.0,
#         "登记时间": "2017-12-05 12:20:53.090",
#         "销售数量": 1
#     },


# 60310000000000003218559648512108778745242856137466315156589943882509006845899071138216111415000778735501893922944355559528693472404468874935296275408324726007401058133881275522362313208216727298785010409755626200739910975613244534598730861144055164425560695882198810624