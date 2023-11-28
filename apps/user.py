import datetime
import os
import pymysql
from flask import request, render_template, jsonify, redirect, url_for, session, Blueprint, send_from_directory
from itsdangerous import json
from Predict import getdata
from LR import LR_predict
from CART import CART_predict
from RF import RF_predict
from datamerge import merge, readfile, read_csv

bp = Blueprint('user', __name__, url_prefix='/user')

# bp.config['UPLOAD_EXTENSIONS'] = ['.csv']
# bp.config['UPLOAD_PATH'] = ['uploads']
from functools import wraps
from flask import session, redirect, url_for


def login_required(role):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if 'role' in session and session['role'] in role:
                # 角色验证通过，执行原函数
                return func(*args, **kwargs)
            else:
                # 角色验证失败，重定向到登录页面
                # return "权限不足"
                return '''<!DOCTYPE html>
                            <html>
                              <head>
                                <title>401 权限不足</title>
                              </head>
                              <body>
                                <h1>401 权限不足</h1>
                                <p>对不起，你没有权限访问该页面。</p>
                              </body>
                            </html>'''

        return wrapper

    return decorator


def func(sql, m='r'):
    # 连接数据库
    db = pymysql.connect(host="localhost", user="root", password="root", database="gdesign",
                         charset='utf8')  # 连接,其中database为数据库名称
    cursor = db.cursor()  # 创建游标对象
    try:
        cursor.execute(sql)  # 执行sql语句
        if m == 'r':  # 如果是读操作
            data = cursor.fetchall()  # 获取所有
        elif m == 'w':  # 写操作
            db.commit()  # 提交
            data = cursor.rowcount  # 返回执行execute()方法后影响的行数。
    except Exception as err:
        print(err)
        data = False
        db.rollback()  # 回滚
    db.close()  # 数据库关闭
    return data


# return render_template('result.html', result_json=json.dumps(result))

@bp.route('/charts', methods=['GET', 'POST'])
@login_required(['user', 'admin', 'superadmin'])
def charts():
    if request.method == 'GET':  # get请求，直接返回页面
        return render_template('charts.html', is_admin=session['role'] == 'admin' or session['role'] == 'superadmin')


@bp.route('/index', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@bp.route('/applying', methods=['GET', 'POST'])
@login_required(['user', 'admin', 'superadmin'])
def applying():
    if request.method == 'GET':
        files = os.listdir('./uploads')
        print(files)
        return render_template('applying.html', files=files,
                               is_admin=session['role'] == 'admin' or session['role'] == 'superadmin')


@bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        # SQL 查询语句
        sql = """ select user_name, role from gd_user where user_name='{}' and user_psw='{}' """.format(username,
                                                                                                        password)
        results = func(sql)
        print(results)

        if results:
            # session['username'] = results[0][0]
            session['username'] = username
            session['role'] = results[0][-1]
            print(results[0][-1])
            session.permanent = True
            return redirect(url_for('user.charts'))

            # return render_template('charts.html', name=session['username'])
        else:
            return render_template('login.html', msg='用户名或者密码错误')
            # return '<script>alert("用户名或者密码错误");location.href="/user/login";</script>'

    return render_template('login.html')


@bp.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        if not validate_username(name):
            return render_template('login.html', msg='用户名长度为5~16，只能使用大小写字母和数字，且必须以字母开头')
        password = request.form['password']
        if not validate_password(password):
            return render_template('login.html', msg='密码长度为6~18，只能使用大小写字母和数字')
        tel = request.form['tel']
        if not validate_phone_number(tel):
            return render_template('login.html', msg='电话号不合法')
        sql = """ insert into gd_user(user_name, user_psw,user_tel,role) values('{}','{}','{}','{}')""".format(name,
                                                                                                               password,
                                                                                                               tel,
                                                                                                               'user')
        results = func(sql, m='w')
        if results:
            return render_template('login.html', msg='注册成功')
        else:
            return '<script>alert("该ID_number已存在");location.href="/user/login";</script>'


@bp.route('/upload', methods=['GET', 'POST'])
@login_required(['user', 'admin', 'superadmin'])
def upload():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        filename = uploaded_file.filename
        print(filename)
        if filename != '':
            file_ext = os.path.splitext(filename)[1]
            print(file_ext)
            if file_ext not in ['.csv']:
                return '<script>alert("请上传正确的文件");location.href="/user/upload";</script>'
            # 上传到可展示文件堆
            uploaded_file.save(os.path.join('./uploads', filename))
            # 上传到处理文件堆
            # uploaded_file.save(os.path.join('./PredictSource', filename))
        return '<script>alert("上传成功");location.href="/user/upload";</script>'

    files = os.listdir('./uploads')
    return render_template('upload.html', files=files,
                           is_admin=session['role'] == 'admin' or session['role'] == 'superadmin')


@bp.route('/rawfile', methods=['GET', 'POST'])
@login_required(['user', 'admin', 'superadmin'])
def rawfile():
    if request.method == 'GET':
        files = os.listdir('./uploads')
        return render_template('rawfile.html', files=files,
                               is_admin=session['role'] == 'admin' or session['role'] == 'superadmin')


# @bp.route('/deal', methods=['GET', 'POST'])
# def deal():
#     if request.method == 'POST':


@bp.route('/dealedfile', methods=['GET', 'POST'])
@login_required(['user', 'admin', 'superadmin'])
def dealedfile():
    if request.method == 'GET':
        files = os.listdir('./dealedfile')
        print(files)
        return render_template('dealedfile.html', files=files,
                               is_admin=session['role'] == 'admin' or session['role'] == 'superadmin')

    if request.method == 'POST':
        filename = request.form['filename']
        if filename:
            print("filename： " + filename)  # predict_raw.csv
            getdata(filename, filename[:-4] + '_del.csv', filename[:-4] + '_dealed.csv')

            return json.dumps({
                'code': 200,
                'msg': 'deal file successfully',
                'data': {}
            })

        return json.dumps({
            'code': 300,
            'msg': 'Loading file error',
            'data': {}
        })


@bp.route('/resultfile', methods=['GET', 'POST'])
@login_required(['user', 'admin', 'superadmin'])
def resultfile():
    if request.method == 'GET':
        files = os.listdir('./resultfile')
        res = []
        print(files)
        for i in range(len(files)):
            if files[i].__contains__('CART'):
                res.append([files[i], 'CART'])
            elif files[i].__contains__('lr'):
                res.append([files[i], '逻辑回归'])
            elif files[i].__contains__('RF'):
                res.append([files[i], '随机森林'])
            else:
                res.append([files[i], '集成三种算法'])

        print(res)
        return render_template('resultfile.html', files=res,
                               is_admin=session['role'] == 'admin' or session['role'] == 'superadmin')


@bp.route('/predict', methods=['POST'])
@login_required(['user', 'admin', 'superadmin'])
def predict():
    filename = request.form['filename']
    option = request.form['option']
    print(option)

    pre = 'D:/pythoncode_2/GD/dealedfile/'
    if filename:
        print("filename： " + filename)  # predict_raw.csv

        # getdata(filename, filename[:-4] + '_del.csv', filename[:-4] + '_dealed.csv')
        if option == '1':
            LR_predict(pre + filename[:], './model/lr_over')
            return json.dumps({
                'code': 200,
                'msg': '逻辑回归预测已完成',
                'data': {}
            })

        if option == '2':
            CART_predict(pre + filename[:], './model/CART_mixed')
            return json.dumps({
                'code': 200,
                'msg': 'CART预测已完成',
                'data': {}
            })

        if option == '3':
            RF_predict(pre + filename[:], './model/RF_mixed')
            return json.dumps({
                'code': 200,
                'msg': '随机森林预测已完成',
                'data': {}
            })
        if option == '4':
            LR_predict(pre + filename[:], './model/lr_over')
            CART_predict(pre + filename[:], './model/CART_mixed')
            RF_predict(pre + filename[:], './model/RF_mixed')
            ppre = 'D:/pythoncode_2/GD/resultfile/' + filename.split('.')[0]
            merge(filename, ppre + '_CART_mixed_submission.csv', ppre + '_lr_over_submission.csv',
                  ppre + '_RF_mixed_submission.csv')
            return json.dumps({
                'code': 200,
                'msg': '集成预测已完成',
                'data': {}
            })

    return json.dumps({
        'code': 300,
        'msg': 'predict file error',
    })


@bp.route('/models', methods=['GET'])
@login_required(['user', 'admin', 'superadmin'])
def showModels():
    return render_template('models.html', is_admin=session['role'] == 'admin' or session['role'] == 'superadmin')


@bp.route('/models_lr', methods=['GET'])
@login_required(['user', 'admin', 'superadmin'])
def showModels_lr():
    return render_template('models_lr.html', is_admin=session['role'] == 'admin' or session['role'] == 'superadmin')


@bp.route('/models_cart', methods=['GET'])
@login_required(['user', 'admin', 'superadmin'])
def showModels_cart():
    return render_template('models_cart.html', is_admin=session['role'] == 'admin' or session['role'] == 'superadmin')


@bp.route('/models_rf', methods=['GET'])
@login_required(['user', 'admin', 'superadmin'])
def showModels_rf():
    return render_template('models_rf.html', is_admin=session['role'] == 'admin' or session['role'] == 'superadmin')


# xxx/result?filename=RF_mixed_submission.csv
@bp.route('/results', methods=['GET'])
@login_required(['user', 'admin', 'superadmin'])
def results():
    filename = request.args.get("filename")
    print(filename)
    # filename = r'D:\pythoncode_2\GD\uploads\predict_raw.csv'
    # submission_file = filename.split('.')[0] + '_submission.csv'
    # return render_template('results.html', files=readfile('D:/pythoncode_2/GD/resultfile/' + filename), is_admin=session['role'] == 'admin' or session['role'] == 'superadmin')
    # df, l = read_csv('D:/pythoncode_2/GD/resultfile/' + filename)

    df, l = read_csv(filename)
    return render_template('results2.html', l=l, cols=df.keys(), df=df,
                           is_admin=session['role'] == 'admin' or session['role'] == 'superadmin')


@bp.route('/download', methods=['GET'])
@login_required(['user', 'admin', 'superadmin'])
def download():
    filename = request.args.get("filename")
    print(filename)
    # submission_file = filename.split('.')[0] + '_submission.csv'
    path, submission_file = os.path.split(filename)
    if not os.path.exists(os.path.join(path, submission_file)):
        return json.dumps({
            'code': 404,
            'msg': 'File Not Found',
        })
    return send_from_directory(path, submission_file, as_attachment=True)


@bp.route('/readme', methods=['GET'])
@login_required(['user', 'admin', 'superadmin'])
def readme():
    return render_template('readme2.html', part=request.args.get("part", 0, type=int),
                           is_admin=session['role'] == 'admin' or session['role'] == 'superadmin')


@bp.route('/backend', methods=['GET'])
@login_required(['admin', 'superadmin'])
def backend():
    sql = """ select user_name, user_tel, role from gd_user where 1=1 """
    users = func(sql)
    return render_template('user_manage.html', part=request.args.get("part", 0, type=int),
                           is_admin=session['role'] == 'admin' or session['role'] == 'superadmin', users=users)


@bp.route('/put_user', methods=['GET'])
@login_required(['admin', 'superadmin'])
def put_user():
    user = request.args.get('user')
    tel = request.args.get('tel')
    role = request.args.get('role')  # 修改后的权限
    o_role = get_role(user)  # 原始权限
    if role not in ['user', 'admin', 'superadmin']:
        return {"code": 400, "msg": "参数错误"}
    if not validate_phone_number(tel):
        return {"code": 400, "msg": "电话号码格式错误"}

    # # 不允许将某用户修改成超级管理员
    # if o_role != role and role == 'superadmin':
    #     return {"code": 401, "msg": "无法修改成超级管理员权限"}

    # 超级管理员无法被修改 and 无法把某用户修改成超级管理员
    if o_role == 'superadmin' or role == 'superadmin':
        return {"code": 401, "msg": "无法修改成超级管理员权限"}

    if session["role"] == 'admin':
        if o_role == 'admin':
            return {"code": 401, "msg": "权限不足"}

    # if role != 'user' and role != o_role and session["role"] != 'superadmin':
    #     return {"code": 401, "msg": "权限不足"}
    # if role != 'user' and user != session['username'] and session["role"] != 'superadmin':
    #     return {"code": 401, "msg": "权限不足"}

    sql = """UPDATE gd_user SET user_tel = '{}', role = '{}' WHERE user_name = '{}'""".format(tel, role, user)
    res = {"code": 200, "msg": "SUCCESS"} if func(sql, 'w') else {"code": 500, "msg": "修改失败"}
    return res


@bp.route('/delete_user', methods=['GET'])
@login_required(['admin', 'superadmin'])
def delete_user():
    user = request.args.get('user')
    role = get_role(user)
    if session["role"] == 'admin' and role == 'admin':
        return {"code": 401, "msg": "管理员无法删除管理员"}
    if role == "superadmin":
        return {"code": 401, "msg": "超级管理员无法被删除"}
    sql = """DELETE FROM gd_user WHERE user_name = '{}'""".format(user)
    res = {"code": 200, "msg": "SUCCESS"} if func(sql, 'w') else {"code": 500, "msg": "删除失败"}
    return res


import os


@bp.route('/delete_file', methods=['GET'])
@login_required(['admin', 'superadmin'])
def delete_file():
    filename = request.args.get('filename')
    path = request.args.get('path')
    print(filename)
    print(path)
    try:
        os.remove(path + filename)
        return {"code": 200, "msg": f"{filename} 已删除."}
    except OSError as error:
        return {"code": 500, "msg": f"Error: {error.filename} - {error.strerror}."}


import re


# 验证用户名
def validate_username(username):
    pattern = r'^[a-zA-Z][a-zA-Z0-9_]{4,15}$'
    return bool(re.match(pattern, username))


# 验证密码
# 6-8位数字
def validate_password(password):
    pattern = r'\w{6,18}$'
    return bool(re.match(pattern, password))


# 验证电话号码
def validate_phone_number(phone_number):
    pattern = r'^1[3456789]\d{9}$'
    return bool(re.match(pattern, phone_number))


def get_role(user):
    sql = """select role from gd_user where user_name = '{}'""".format(user)
    try:
        return func(sql)[0][0]
    except:
        return None
