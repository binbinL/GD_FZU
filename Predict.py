from LR import LR_predict
from CART import *
from GetPredictData import *
from GetPredictData import get_del, get_transform


pre = 'D:/pythoncode_2/GD/Predictsource/'


def getdata(path, des, des2):
    '''

    :param path: 源路径
    :param des: 中间过程文件
    :param des2: 用于模型输入的文件
    :return:
    '''
    # rawfile
    path = 'D:/pythoncode_2/GD/uploads/' + path
    des = 'D:/pythoncode_2/GD/del/' + des
    des2 = 'D:/pythoncode_2/GD/dealedfile/' + des2
    get_del(path, des)
    get_transform(des, des2)


def LR_Res(path, model):
    # model = './model/lr_under'
    path = pre + path
    LR_predict(path, model)

def CART_Res(path, model):
    path = pre + path
    CART_predict(path, model)

def RF_Res(path, model):
    path = pre + path
    CART_predict(path, model)