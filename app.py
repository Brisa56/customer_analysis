import json
import random
from datetime import datetime
import user_portrait
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO

import matplotlib
import warnings
import re
from wordcloud import WordCloud
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# %matplotlib inline
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams.update({'font.size' : 16})
plt.style.use('ggplot')
warnings.filterwarnings('ignore')
from flask import Flask, render_template, request, flash, redirect, session, jsonify

import pymysql
from sqlalchemy import null

app = Flask(__name__)
app.secret_key = "123"


def conn_db():
    ######## 数据库操作
    # 1.链接数据库
    conn = pymysql.connect(
        host="127.0.0.1", port=3306,
        db="customer_analysis", user="root", password="yyqxwhc.",
        charset="utf8"
    )
    return conn


@app.route('/')
def home():  # put application's code here
    return redirect("/page/template/login/login.html")


# **************** 登录  ****************
@app.route('/page/template/login/login.html')
def login():  # put application's code here
    return render_template("page/template/login/login.html")


@app.route('/doLogin', methods=['GET', 'POST'])
def doLogin():
    # 获取姓名 request.get('key')
    name = request.form.get("username")
    pwd = request.form.get("password")
    print("username:", name, "password:", pwd)

    # 1.连接数据库
    conn = conn_db()
    # 2.创建游标对象
    cursor = conn.cursor()
    # 3.sql语句
    sql = "select * from user where username=%s and password=%s"
    row = cursor.execute(sql, [name, pwd])
    # 4.提交
    conn.commit()
    # 5.关闭
    conn.close()

    if row > 0:
        session['username'] = name
        return redirect("/index")
    else:
        flash("密码不正确")
        return redirect("/page/template/login/login.html")


# **************** 注册  ****************
@app.route('/page/template/login/reg.html')
def reg():
    return render_template("page/template/login/reg.html")


@app.route('/doReg', methods=['GET', 'POST'])
def doReg():
    # 获取姓名 request.get('key')
    name = request.form.get("username")
    pwd = request.form.get("password")
    email = request.form.get("email")
    ######## 数据库操作
    # 1.链接数据库
    conn = conn_db()
    # 2.创建游标对象
    cursor = conn.cursor()
    # 3.sql语句
    sql = "insert into user values(null, %s, %s, %s)"
    row = cursor.execute(sql, [name, pwd, email])
    # 4.提交
    conn.commit()
    # 5.关闭
    conn.close()

    if row > 0:
        return redirect("/page/template/login/login.html")
    else:
        flash("注册失败")
        return redirect("/page/template/login/reg.html")


# **************** 忘记密码  ****************
@app.route('/page/template/login/forget.html')
def forget():  # put application's code here
    return render_template("page/template/login/forget.html")


@app.route('/doForget', methods=['GET', 'POST'])
def doForget():
    # 获取姓名 request.get('key')
    email = request.form.get("email")
    pwd = request.form.get("password")

    # 2.创建游标对象
    conn = conn_db()
    cursor = conn.cursor()
    # 3.sql语句
    sql = "update user set password=%s where email=%s "
    row = cursor.execute(sql, [pwd, email])
    # 4.提交
    conn.commit()
    # 5.关闭
    conn.close()

    if row > 0:
        return redirect("/page/template/login/login.html")
    else:
        flash("更改失败")
        return redirect("/page/template/login/forget.html")


# **************** 首页  ****************
@app.route('/index')
def index():
    return render_template('index.html')


# **************** 首页-顶部栏  ****************
@app.route('/page/tpl/tpl-message.html')
def tpl_message():
    return render_template('page/tpl/tpl-message.html')


@app.route('/page/tpl/tpl-lock-screen.html')
def tpl_lock_screen():
    return render_template('page/tpl/tpl-lock-screen.html')


@app.route('/page/tpl/tpl-note.html')
def tpl_note():
    return render_template('page/tpl/tpl-note.html')


@app.route('/page/tpl/tpl-password.html')
def tpl_password():
    return render_template('page/tpl/tpl-password.html')


# 修改密码
@app.route('/page/tpl/tpl-password/change', methods=['GET', 'POST'])
def tpl_password_change():
    username = session.get('username')

    oldPsw = request.form.get('oldPsw')
    newPsw = request.form.get('newPsw')
    # 2.创建游标对象
    conn = conn_db()
    cursor = conn.cursor()
    # 3.sql语句
    sql = "update user set password=%s where username=%s and password=%s"
    row = cursor.execute(sql, [newPsw, username, oldPsw])
    # 4.提交
    conn.commit()
    # 5.关闭
    conn.close()
    if row == 0:
        flash("更改失败")
    return redirect("/index")


@app.route('/page/tpl/tpl-theme.html')
def tpl_theme():
    return render_template('page/tpl/tpl-theme.html')


# **************** 首页-侧边栏-DashBoard  ****************
# 工作台
@app.route("/page/console/workplace.html")
def workPlace():
    return render_template("page/console/workplace.html")


# 控制台
@app.route("/page/console/console.html")
def console():
    return render_template("page/console/console.html")


# 分析页
@app.route("/page/console/dashboard.html")
def dashBoard():
    return render_template("page/console/dashboard.html")


# ****************首页-侧边栏-客户管理  ****************
# 客户管理
@app.route("/page/customer/customer.html")
def customerIndex():
    return render_template("page/customer/customer.html")

def getCustomerJson(result):  # 数据库中的role数据转为json格式用于前端Table输出
    customer_vip = {}
    customer_vip['code'] = 0
    customer_vip['msg'] = ''
    data = []
    count = 0
    for r in result:
        count = count + 1
        dict = {}
        dict["memberId"] = r[0]
        dict["bitrhday"] = r[1]
        dict["sex"] = r[2]
        dict["registerTime"] = r[3]
        data.append(dict)
    customer_vip['count'] = count
    customer_vip['data'] = data
    # print(user)
    return jsonify(customer_vip)

@app.route("/page/customer/customer/jsonData", methods=['GET', 'POST'])
def customerJsonData():  # 展示以及查找数据库中的user数据
    memberId = request.args.get("memberId")
    sex = request.args.get("sex")
    bitrhday = request.args.get("bitrhday")
    # 2.创建游标对象
    conn = conn_db()
    cursor = conn.cursor()
    if memberId:
        if sex:
            if bitrhday:
                sql = "select * from vip_info1 where memberId=%s and sex=%s and bitrhday=%s "
                row = cursor.execute(sql, [memberId, sex, bitrhday])
            else:
                sql = "select * from vip_info1 where memberId=%s and sex=%s "
                row = cursor.execute(sql, [memberId, sex])
        else:
            if bitrhday:
                sql = "select * from vip_info1 where memberId=%s and bitrhday=%s "
                row = cursor.execute(sql, [memberId, bitrhday])
            else:
                sql = "select * from vip_info1 where memberId=%s "
                row = cursor.execute(sql, [memberId])
    else:
        if sex:
            if bitrhday:
                sql = "select * from vip_info1 where  sex=%s and bitrhday=%s "
                row = cursor.execute(sql, [sex, bitrhday])
            else:
                sql = "select * from vip_info1 where  sex=%s "
                row = cursor.execute(sql, [sex])
        else:
            if bitrhday:
                sql = "select * from vip_info1 where  bitrhday=%s "
                row = cursor.execute(sql, [bitrhday])
            else:
                sql = "select * from vip_info1 "
                row = cursor.execute(sql)
    result = cursor.fetchall()
    # # 4.提交
    conn.commit()
    # 5.关闭
    conn.close()
    json = getCustomerJson(result) # 数据库中的数据转为json
    return json


# 总体分析
@app.route("/page/customer/overallAnalysis.html")
def customerBasicAnalysis():
    L = pd.read_csv('./static/data/L.csv', encoding='utf-8')
    df1 =  pd.read_csv('./static/data/df1.csv', encoding='utf-8')
    df_vip = pd.read_csv('./static/data/vip_info.csv', encoding='utf-8')

    # 1.出生年代
    age_sort = L['年龄'].value_counts()
    # 排序
    age_sort.sort_index(inplace=True)
    age_sort_index = age_sort.index.tolist()
    age_sort_values = age_sort.values.tolist()
    # 2.年龄段
    age_group = L['年龄段'].value_counts()
    age_group_index = age_group.index.tolist()
    age_group_values = age_group.values.tolist()
    # 3.性别
    sex_sort = L['性别'].value_counts()
    sex_sort_index = sex_sort.index.tolist()
    sex_sort_values = sex_sort.values.tolist()
    # 4.消费人数及消费情况会员占比
    isVipLabels = ['会员', '非会员']
    orderIsVip = [len(df1.loc[df1['会员'] == 1, '消费产生的时间'].unique()), len(df1.loc[df1['会员'] == 0, '消费产生的时间'].unique())]
    saleIsVip = [df1.loc[df1['会员'] == 1, '消费金额'].sum(), df1.loc[df1['会员'] == 0, '消费金额'].sum()]
    # 5.消费偏好
    # 前提假设：2015-2018年之间，消费者偏好在时间上不会发生太大的变化（均值），消费偏好——>以不同时间的订单数来衡量
    quarters_list, quarters_order = orders(df_vip, '季度', 3)  # [17226, 19886, 15874, 18015]
    # quarters_list, quarters_order = orders(df_vip, '季度', 4)  # [12919, 14914, 11905, 13511]
    days_list, days_order = orders(df_vip, '天', 36)

    return render_template("page/customer/overallAnalysis.html", age_sort_index=age_sort_index, age_sort_values=age_sort_values,
                           age_group_index=age_group_index, age_group_values=age_group_values,
                           sex_sort_index=sex_sort_index,sex_sort_values=sex_sort_values,
                           isVipLabels=isVipLabels,orderIsVip=orderIsVip,saleIsVip=saleIsVip,
                            quarters_order=quarters_order,quarters_list=quarters_list.tolist(),days_order=days_order,days_list=days_list.tolist()
                           )
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

        # order_nums.append(int(len(df.loc[df[label] == x_list[i], '消费产生的时间'].unique()) ))
        # [51678, 59658, 47622, 54046]
    return x_list, order_nums


# 用户画像
@app.route("/page/customer/portrait.html", methods=['GET', 'POST'])
def customerPortrait():
    df_profile = pd.read_csv('./static/data/consumers_profile.csv', encoding='utf-8')
    # src = user_portrait.wc_plot(df_profile)
    src = wc_plot(df_profile)
    randomList = random.sample(range(1, 100), 5)

    return render_template("page/customer/portrait.html", src=src,df_profile=df_profile,
                           randomList=randomList)

@app.route("/page/customer/portraitData", methods=['GET', 'POST'])
def customerPortraitData():
    memberId = request.form.get('memberId')
    df_profile = pd.read_csv('./static/data/consumers_profile.csv', encoding='utf-8')
    # src = user_portrait.wc_plot(df_profile, memberId)
    src = wc_plot(df_profile, memberId)
    randomList = random.sample(range(1, 100), 5)
    return render_template("page/customer/portrait.html", src=src,df_profile=df_profile,
                           randomList=randomList)
# 开始绘制用户词云，封装成一个函数来直接显示词云
def wc_plot(df, id_label = None):
    """
    df: 为DataFrame的数据集
    id_label: 为输入用户的会员卡号，默认为随机取一个会员进行展示
    """
    myfont = './fonts/Simfang.ttf'
    if id_label == None:
        id_label = df.loc[np.random.choice(range(df.shape[0])), '会员卡号']
    # sex = df[df['会员卡号'] == id_label]['性别'].values
    # backgroundImage = plt.imread('./static/data/男.jpg')
    # if '女' in sex:
    #     backgroundImage = plt.imread('./static/data/女.jpg')
    text = df[df['会员卡号'] == id_label].T.iloc[:, 0].values.tolist()
    plt.figure(dpi = 100)
    wc = WordCloud(font_path = myfont, background_color = None,  # 透明背景
                   width = 500, height = 440).generate_from_text(' '.join(text))
    plt.imshow(wc)
    plt.axis('off')
    sio = BytesIO()
    plt.savefig(sio, format='png', bbox_inches='tight', pad_inches=0.0)
    data = base64.encodebytes(sio.getvalue()).decode()
    src = 'data:image/png;base64,' + str(data)
    # 记得关闭，不然画出来的图是重复的
    plt.close()
    return src
# **************** 首页-侧边栏-系统管理  ****************
# 用户管理
@app.route("/page/system/user.html")
def user():
    return render_template("page/system/user.html")

def getUserJson(result):  # 数据库中的role数据转为json格式用于前端Table输出
    user = {}
    user['code'] = 0
    user['msg'] = ''
    data = []
    count = 0
    for r in result:
        count = count + 1
        dict = {}
        dict["userId"] = r[0]
        dict["username"] = r[1]
        dict["password"] = r[2]
        dict["nickName"] = r[3]
        dict["avatar"] = r[4]
        dict["sex"] = r[5]
        dict["phone"] = r[6]
        dict["email"] = r[7]
        dict["emailVerified"] = r[8]
        dict["trueName"] = r[9]
        dict["idCard"] = r[10]
        dict["birthday"] = r[11]
        dict["departmentId"] = r[12]
        dict["state"] = r[13]
        dict["createTime"] = r[14]
        dict["updateTime"] = r[15]
        dict["roleId"] = r[16]
        dict["roleName"] = r[17]
        data.append(dict)
    user['count'] = count
    user['data'] = data
    # print(user)
    return jsonify(user)

@app.route("/page/system/user/jsonData", methods=['GET', 'POST'])
def userJsonData():  # 展示以及查找数据库中的user数据
    username = request.args.get("username")
    nickName = request.args.get("nickName")
    sex = request.args.get("sex")
    # 2.创建游标对象
    conn = conn_db()
    cursor = conn.cursor()
    if username:
        if nickName:
            if sex:
                sql = "select * from user where username=%s and nickName=%s and sex=%s "
                row = cursor.execute(sql, [username, nickName, sex])
            else:
                sql = "select * from user where username=%s and nickName=%s "
                row = cursor.execute(sql, [username, nickName])
        else:
            if sex:
                sql = "select * from user where username=%s and sex=%s "
                row = cursor.execute(sql, [username, sex])
            else:
                sql = "select * from user where username=%s "
                row = cursor.execute(sql, [username])
    else:
        if nickName:
            if sex:
                sql = "select * from user where  nickName=%s and sex=%s "
                row = cursor.execute(sql, [nickName, sex])
            else:
                sql = "select * from user where  nickName=%s "
                row = cursor.execute(sql, [nickName])
        else:
            if sex:
                sql = "select * from user where  sex=%s "
                row = cursor.execute(sql, [sex])
            else:
                sql = "select * from user "
                row = cursor.execute(sql)
    result = cursor.fetchall()
    # # 4.提交
    conn.commit()
    # 5.关闭
    conn.close()
    json = getUserJson(result)  # 数据库中的数据转为json
    return json

@app.route("/page/system/user/editUserData", methods=['GET', 'POST'])
def editUserData():  # 修改及添加数据库中的user数据

   # "GET /static/json/ok.json?userId=1&username=admin&nickName=管理员&sex=男&phone=12345678901&userEditRoleSel=1&roleIds=1 HTTP/1.1"
   #  "GET /static/json/ok.json?userId=&username=11&nickName=111&sex=男&phone=1&userEditRoleSel=3&roleIds=3 HTTP/1.1"
    userId = request.args.get("userId")
    username = request.args.get("username")
    nickName = request.args.get("nickName")
    sex = request.args.get("sex")
    phone = request.args.get("phone")
    roleName_  = request.args.get("userEditRoleSel")
    roleId = request.args.get("roleIds")
    state = request.args.get("state")

    # 2.创建游标对象
    conn = conn_db()
    cursor = conn.cursor()
    if state:
        sql = "update user set state=%s where userId=%s"
        row = cursor.execute(sql, [state, userId])
    else:
        sql = "select roleName from role where roleId=%s"
        row = cursor.execute(sql, [roleId])
        roleName = (cursor.fetchone())[0]

        print(userId, username, nickName, sex, phone, roleName, roleId)

        if userId:
            # # 3.sql语句
            sql = "update user set username=%s, nickName=%s, sex=%s, phone=%s, roleName=%s, roleId=%s, updateTime=%s where userId=%s"
            row = cursor.execute(sql, [username, nickName, sex, phone, roleName, roleId, str(datetime.now()), userId])
        else:
            sql = "insert into user (userId, username, nickName, sex, phone, roleName, roleId, state, createTime, updateTime) values (null, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
            row = cursor.execute(sql, [username, nickName, sex, phone, roleName, roleId, 1, str(datetime.now()), str(datetime.now())])

    # # 4.提交
    conn.commit()
    # 5.关闭
    conn.close()
    ###
    if row > 0:
        return jsonify({"msg": "操作成功", "code": 200})
    else:
        flash("更改失败")
        return jsonify({"msg": "操作失败"})


@app.route("/page/system/user/deleteUserData", methods=['GET', 'POST'])
def deleteUserData():  # 删除数据库中的role数据（将isDelete改为1）
    # GET /page/system/user/deleteUserData?id=10&ids= HTTP/1.1" 200 -
    userId = request.args.get("id")
    userIds = request.args.get("ids")
    # 2.创建游标对象
    conn = conn_db()
    cursor = conn.cursor()
    if userId:
        # # 3.sql语句
        sql = "delete from user where userId=%s"
        row = cursor.execute(sql, [userId])
    if userIds:
        for id in userIds:
            sql = "delete from user where userId=%s"
            row = cursor.execute(sql, [id])
    # 4.提交
    conn.commit()
    # 5.关闭
    conn.close()
    ###
    if row > 0:
        return jsonify({"msg": "操作成功", "code": 200})
    else:
        flash("更改失败")
        return jsonify({"msg": "操作失败"})

@app.route("/page/system/user/resetPasswordUserData", methods=['GET', 'POST'])
def resetPasswordUserData():  # 重置密码
    # GET /page/system/user/deleteUserData?id=10&ids= HTTP/1.1" 200 -
    userId = request.args.get("userId")
    # 2.创建游标对象
    conn = conn_db()
    cursor = conn.cursor()
    if userId:
        # # 3.sql语句
        sql = "update user set password=%s where userId=%s"
        row = cursor.execute(sql, ['123456', userId])
    # 4.提交
    conn.commit()
    # 5.关闭
    conn.close()
    ###
    if row > 0:
        return jsonify({"msg": "操作成功", "code": 200})
    else:
        flash("更改失败")
        return jsonify({"msg": "操作失败"})



# 角色管理
@app.route("/page/system/role.html")
def role():
    return render_template("page/system/role.html")

def getRoleJson(result):  # 数据库中的role数据转为json格式用于前端Table输出
    role = {}
    role['code'] = 0
    role['msg'] = ''
    data = []
    count = 0
    for r in result:
        count = count + 1
        dict = {}
        dict["roleId"] = r[0]
        dict["roleName"] = r[1]
        dict["comments"] = r[2]
        dict["roleCode"] = r[3]
        dict["isDelete"] = r[4]
        dict["createTime"] = r[5]
        dict["updateTime"] = r[6]
        data.append(dict)
    role['count'] = count
    role['data'] = data
    # print(role)
    return jsonify(role)


@app.route("/page/system/role/jsonData", methods=['GET', 'POST'])
def roleJsonData():  # 展示以及查找数据库中的role数据（isDelete为0）
    # limit = request.args.get("limit")
    # page = request.args.get("page")
    roleName = request.args.get("roleName")
    roleCode = request.args.get("roleCode")
    comments = request.args.get("comments")

    # 2.创建游标对象
    conn = conn_db()
    cursor = conn.cursor()
    if roleName:
        if roleCode:
            if comments:
                sql = "select * from role where roleName=%s and roleCode=%s and comments=%s and isDelete=0"
                row = cursor.execute(sql, [roleName, roleCode, comments])
            else:
                sql = "select * from role where roleName=%s and roleCode=%s and isDelete=0"
                row = cursor.execute(sql, [roleName, roleCode])
        else:
            if comments:
                sql = "select * from role where roleName=%s and comments=%s and isDelete=0"
                row = cursor.execute(sql, [roleName, comments])
            else:
                sql = "select * from role where roleName=%s and isDelete=0"
                row = cursor.execute(sql, [roleName])
    else:
        if roleCode:
            if comments:
                sql = "select * from role where  roleCode=%s and comments=%s and isDelete=0"
                row = cursor.execute(sql, [roleCode, comments])
            else:
                sql = "select * from role where  roleCode=%s and isDelete=0"
                row = cursor.execute(sql, [roleCode])
        else:
            if comments:
                sql = "select * from role where  comments=%s and isDelete=0"
                row = cursor.execute(sql, [comments])
            else:
                sql = "select * from role where isDelete=0"
                row = cursor.execute(sql)
    result = cursor.fetchall()
    # # 4.提交
    conn.commit()
    # 5.关闭
    conn.close()
    json = getRoleJson(result)  # 数据库中的数据转为json
    return json


@app.route("/page/system/role/editRoleData", methods=['GET', 'POST'])
def editRoleData():  # 修改及添加数据库中的role数据
    roleId = request.args.get("roleId")
    roleName = request.args.get("roleName")
    roleCode = request.args.get("roleCode")
    comments = request.args.get("comments")
    print(roleId, roleName, roleCode, comments)
    # 2.创建游标对象
    conn = conn_db()
    cursor = conn.cursor()
    if roleId:
        # # 3.sql语句
        sql = "update role set roleName=%s, roleCode=%s, comments=%s,  updateTime=%s where roleId=%s"
        row = cursor.execute(sql, [roleName, roleCode, comments, str(datetime.now()), roleId])
    else:
        sql = "insert into role (roleId, roleName, comments, roleCode, isDelete, createTime, updateTime) values (null, %s, %s, %s, %s, %s, %s)"
        row = cursor.execute(sql, [roleName, comments, roleCode, 0, str(datetime.now()), str(datetime.now())])

    # print(type(roleId), type(roleName), type(roleCode),type(comments))  #str
    # # 4.提交
    conn.commit()
    # 5.关闭
    conn.close()
    ###
    if row > 0:
        return jsonify({"msg": "操作成功", "code": 200})
    else:
        flash("更改失败")
        return jsonify({"msg": "操作失败"})


@app.route("/page/system/role/deleteRoleData", methods=['GET', 'POST'])
def deleteRoleData():  # 删除数据库中的role数据（将isDelete改为1）
    roleId = request.args.get("id")
    roleIds = request.args.get("ids")
    # 2.创建游标对象
    conn = conn_db()
    cursor = conn.cursor()
    if roleId:
        # # 3.sql语句
        sql = "update role set isDelete=1 where roleId=%s"
        row = cursor.execute(sql, [roleId])
    if roleIds:
        for id in roleIds:
            sql = "update role set isDelete=1 where roleId=%s"
            row = cursor.execute(sql, [id])
    # 4.提交
    conn.commit()
    # 5.关闭
    conn.close()
    ###
    if row > 0:
        return jsonify({"msg": "操作成功", "code": 200})
    else:
        flash("更改失败")
        return jsonify({"msg": "操作失败"})

# 角色管理------------权限分配--------未完成
@app.route("/page/system/role/role_auth_tree", methods=['GET', 'POST'])
def roleAuthtree():  # 角色权限树
    # 2.创建游标对象
    conn = conn_db()
    cursor = conn.cursor()
    roleId = request.args.get('roleId=1')
    sql = 'select * from role_auth_tree'
    row = cursor.execute(sql)
    result = cursor.fetchall()
    role_auth_tree = {}
    role_auth_tree['code'] = 200
    role_auth_tree['msg'] = ''
    data = []
    count = 0
    for r in result:
        count = count + 1
        dict = {}
        dict["name"] = r[0]
        dict["checked"] = r[1]
        dict["pId"] = r[2]
        dict["id"] = r[3]
        dict["open"] = r[4]
        data.append(dict)
    role_auth_tree['count'] = count
    role_auth_tree['data'] = data
    # 4.提交
    conn.commit()
    # 5.关闭
    conn.close()
    # print(role)
    return jsonify(role_auth_tree)

# @app.route("/page/system/role/changeAuthorityRoleData", methods=['GET', 'POST'])
# def changeAuthorityRoleData():  # 删除数据库中的role数据（将isDelete改为1）
#     # roleId = 1 & authIds =
#     roleId = request.args.get('roleId')
#     authIds = request.args.get('authIds')
#     # 2.创建游标对象
#     conn = conn_db()
#     cursor = conn.cursor()
#     idNotInAuthIds = []  # 对应index为1则表示对应数字 为空缺，需修改，为0则不用修改
#     if authIds:
#         for id in authIds:
#
#             # sql = "update role_auth set open='true' where roleId=%s and authId=%s"
#             # row = cursor.execute(sql, [roleId, id])
#     # else:
#     #     sql = "update role_auth set open='false' where roleId=%s"
#     #     row = cursor.execute(sql, [roleId])
#     # 4.提交
#     conn.commit()
#     # 5.关闭
#     conn.close()
#     ###
#     if row > 0:
#         return jsonify({"msg": "操作成功", "code": 200})
#     else:
#         flash("更改失败")
#         return jsonify({"msg": "操作失败"})


# 权限管理
@app.route("/page/system/authorities.html")
def authorities():
    return render_template("page/system/authorities.html")


def getAuthoritiesJson(result):  # 数据库中的role数据转为json格式用于前端Table输出
    authorities = {}
    authorities['code'] = 0
    authorities['msg'] = ''
    data = []
    count = 0
    for r in result:
        count = count + 1
        dict = {}
        dict["authorityId"] = r[0]
        dict["authorityName"] = r[1]
        dict["authority"] = r[2]
        dict["menuUrl"] = r[3]
        dict["parentId"] = r[4]
        dict["isMenu"] = r[5]
        dict["orderNumber"] = r[6]
        dict["menuIcon"] = r[7]
        dict["createTime"] = r[8]
        dict["updateTime"] = r[9]
        dict["open"] = r[10]
        data.append(dict)

    authorities['count'] = count
    authorities['data'] = data
    # print(authorities)
    return jsonify(authorities)


@app.route("/page/system/authorities/jsonData", methods=['GET', 'POST'])
def authoritiesJsonData():  # 展示以及查找数据库中的authorities数据（isDelete为0）
    authorityName = request.args.get('authorityName')
    menuUrl = request.args.get('menuUrl')
    authority = request.args.get('authority')
    # 2.创建游标对象
    conn = conn_db()
    cursor = conn.cursor()
    if authorityName:
        if menuUrl:
            if authority:
                sql = "select * from authorities where authorityName=%s and menuUrl=%s and authority=%s "
                row = cursor.execute(sql, [authorityName, menuUrl, authority])
            else:
                sql = "select * from authorities where authorityName=%s and menuUrl=%s "
                row = cursor.execute(sql, [authorityName, menuUrl])
        else:
            if authority:
                sql = "select * from authorities where authorityName=%s and authority=%s "
                row = cursor.execute(sql, [authorityName, authority])
            else:
                sql = "select * from authorities where authorityName=%s "
                row = cursor.execute(sql, [authorityName])
    else:
        if menuUrl:
            if authority:
                sql = "select * from authorities where  menuUrl=%s and authority=%s "
                row = cursor.execute(sql, [menuUrl, authority])
            else:
                sql = "select * from authorities where  menuUrl=%s "
                row = cursor.execute(sql, [menuUrl])
        else:
            if authority:
                sql = "select * from authorities where  authority=%s "
                row = cursor.execute(sql, [authority])
            else:
                sql = "select * from authorities"
                row = cursor.execute(sql)
    result = cursor.fetchall()
    # print(result)
    # # 4.提交
    conn.commit()
    # 5.关闭
    conn.close()
    json = getAuthoritiesJson(result)  # 数据库中的数据转为json
    return json

@app.route("/page/system/authorities/editAuthoritiesData", methods=['GET', 'POST'])
def editAuthoritiesData():  # 修改及添加数据库中的authorities数据
 # "GET /static/json/ok.json?authorityId=2&select=2&authorityName=用户管理&isMenu=0&menuUrl=system%2Fuser&authority=&menuIcon=&orderNumber=2&parentId=2 HTTP/1.1" 200 -
 # "GET /static/json/ok.json?authorityId=&select=&authorityName=1&isMenu=0&menuUrl=&authority=&menuIcon=&orderNumber=1&parentId= HTTP/1.1" 200 -
    authorityId = request.args.get("authorityId")
    # select = request.args.get("select")
    authorityName = request.args.get("authorityName")
    isMenu = request.args.get("isMenu")
    menuUrl = request.args.get("menuUrl")
    authority = request.args.get("authority")
    menuIcon = request.args.get("menuIcon")
    orderNumber = request.args.get("orderNumber")
    parentId = request.args.get("parentId")
# authorityId, select, authorityName, isMenu, menuUrl, authority, menuIcon, orderNumber, parentId
    # 2.创建游标对象
    conn = conn_db()
    cursor = conn.cursor()
    if authorityId:
        # # 3.sql语句
        sql = "update authorities set  authorityName=%s, isMenu=%s, menuUrl=%s, authority=%s, menuIcon=%s, orderNumber=%s,updateTime=%s where authorityId=%s"
        row = cursor.execute(sql, [authorityName, isMenu, menuUrl, authority, menuIcon, orderNumber, str(datetime.now()), authorityId])
    else:
        sql = "insert into authorities (authorityId,  authorityName, isMenu, menuUrl, authority, menuIcon, orderNumber, parentId, createTime, updateTime, open) values (null, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'true')"
        row = cursor.execute(sql, [authorityName, isMenu, menuUrl, authority, menuIcon, orderNumber, parentId, str(datetime.now()), str(datetime.now())])
    # 4.提交
    conn.commit()
    # 5.关闭
    conn.close()
    ###
    if row > 0:
        return jsonify({"msg": "操作成功", "code": 200})
    else:
        flash("更改失败")
        return jsonify({"msg": "操作失败"})


@app.route("/page/system/authorities/deleteAuthoritiesData", methods=['GET', 'POST'])
def deleteAuthoritiesData():  # 删除数据库中的role数据（将isDelete改为1）
# 127.0.0.1 - - [25/Apr/2022 17:02:05] "GET /static/json/ok.json?id%5B%5D=1&id%5B%5D=2&id%5B%5D=3&id%5B%5D=4 HTTP/1.1" 200 -
    authorityId = request.args.get("id")
    authorityIds = request.args.get("ids")
    # 2.创建游标对象
    conn = conn_db()
    cursor = conn.cursor()
    if authorityId:
        # # 3.sql语句
        sql = "delete from authorities where authorityId=%s"
        row = cursor.execute(sql, [authorityId])
    if authorityIds:
        for id in authorityIds:
            sql = "delete from authorities where authorityId=%s"
            row = cursor.execute(sql, [id])
    # 4.提交
    conn.commit()
    # 5.关闭
    conn.close()
    ###
    if row > 0:
        return jsonify({"msg": "操作成功", "code": 200})
    else:
        flash("更改失败")
        return jsonify({"msg": "操作失败"})

# 字典管理
@app.route("/page/system/dictionary.html")
def dictionary():
    return render_template("page/system/dictionary.html")


# 机构管理
@app.route("/page/system/organization.html")
def organization():
    return render_template("page/system/organization.html")


# 登录日志
@app.route("/page/system/login-record.html")
def login_Record():
    return render_template("page/system/login-record.html")


# ****************首页-侧边栏-个人中心  ****************
# 个人中心
@app.route("/page/template/user-info.html")
def userInfo():
    username = session.get('username')

    # 2.创建游标对象
    conn = conn_db()
    cursor = conn.cursor()
    # 3.sql语句
    sql = "select * from user where username=%s"
    row = cursor.execute(sql, username)
    result = cursor.fetchone()  # 元组集合  (1, 'root', '123456', 'root', '我很帅气', '四川省成都市新都区', 0, '111111', '123456@qq.com', '', datetime.datetime(2022, 4, 20, 13, 23, 16), datetime.datetime(2022, 4, 20, 13, 23, 20))
    # 4.提交
    conn.commit()
    # 5.关闭
    conn.close()
    dict = {}
    dict['email'] = result[8]
    dict['name'] = result[3]
    dict['desc'] = result[4]
    dict['address'] = result[5]
    dict['phone2'] = result[7]
    return render_template("page/template/user-info.html", dict=dict)


# 个人中心信息修改
@app.route("/page/template/user-info/change", methods=['GET', 'POST'])
def userInfoChange():
    username = session.get('username')

    email = request.form.get('email')
    nickname = request.form.get('name')
    intro = request.form.get('desc')
    address = request.form.get('address')
    phone = request.form.get('phone2')
    # 2.创建游标对象
    conn = conn_db()
    cursor = conn.cursor()
    # 3.sql语句
    sql = "update user set email=%s, nickname=%s, intro=%s, address=%s, phone=%s where username=%s"
    row = cursor.execute(sql, [email, nickname, intro, address, phone, username])
    # 4.提交
    conn.commit()
    # 5.关闭
    conn.close()
    if row == 0:
        flash("更改失败")
    return redirect("/page/template/user-info.html")

# 加一个东西
@app.route("/page/plugin/other/layui.html")
def layUi():
    return render_template("page/plugin/other/layui.html")

if __name__ == '__main__':
    app.run()
