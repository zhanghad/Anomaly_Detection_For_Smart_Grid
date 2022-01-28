# -*- coding: utf-8 -*-
import pickle
import pandas as pd
import constant
import json
import datetime
import copy
from db_pool import POOL


def save_object(obj, path):
    """
    导出对象
    :param obj:
    :param path:
    :return:
    """
    file = open(path, 'wb')
    out_objcet = pickle.dumps(obj)
    file.write(out_objcet)
    file.close()


def load_object(path):
    """
    导入对象
    :param path:
    :return:
    """
    with open(path, 'rb') as file:
        obj = pickle.loads(file.read())
    return obj


def print_dict(dic):
    """
    打印字典
    :param dic:
    :return:
    """
    data = json.dumps(dic, indent=4, ensure_ascii=False,
                      sort_keys=False, separators=(',', ':'))
    print(data)


def print_obj(obj):
    """
    打印对象信息
    :param obj:
    :return:
    """
    for key, values in obj.__dict__.items():
        print(key, ':', values)


def date_range(start_date, end_date):
    """
    生成连续日期
    :param start_date:
    :param end_date:
    :return:
    """
    result = []
    for n in range(int((end_date - start_date).days)):
        result.append((start_date + datetime.timedelta(n)).strftime('%Y%m%d'))
    return result


def train_data_from_db(device_id, include_gas, start_time_str=None, end_time_str=None):
    """
    从数据库中读取训练数据
    :param device_id:
    :param include_gas:
    :param start_time_str:
    :param end_time_str:
    :return: DataFrame
    """

    # 生成遍历的日期区间
    start = datetime.datetime(2020, 10, 1, 0, 0, 0)
    end = datetime.datetime.today()
    if start_time_str is not None:
        start = datetime.datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S")
    if end_time_str is not None:
        end = datetime.datetime.strptime(end_time_str, "%Y-%m-%d %H:%M:%S")
    table_date = date_range(start, end)

    # 确定是否查询气体字段
    data_name_list = copy.deepcopy(constant.DB_DATA_NAME)
    if include_gas == constant.INCLUDE_GAS:
        data_name_list += constant.DB_DATA_NAME_GAS

    # 生成查询的sql字段
    attr_str = ''
    for a in data_name_list:
        attr_str = attr_str + a + ','
    attr_str = attr_str[:-1]

    # 从连接池获取数据库连接
    conn_hybackup = POOL.connection()
    cursor_hybackup = conn_hybackup.cursor()
    # 遍历每张表
    temp = []
    for t in table_date:
        sql = '''select %s from data_king%s where equipment_id = '%s';''' % (attr_str, t, device_id)
        cursor_hybackup.execute(sql)
        t_data = list(cursor_hybackup.fetchall())
        t_df = pd.DataFrame(data=t_data, columns=data_name_list)
        temp.append(t_df)
    # 释放数据库连接
    cursor_hybackup.close()
    conn_hybackup.close()

    # 数据处理
    result = pd.concat(temp)
    result[constant.DB_DATA_NAME[0]] = result.apply(
        lambda x: str(datetime.datetime.strftime(x[constant.DB_DATA_NAME[0]], "%Y-%m-%d %H:%M:%S")),
        axis=1)
    result.set_index(constant.DB_DATA_NAME[0], inplace=True, drop=True)
    result.sort_index(inplace=True)
    result = result.astype('float')
    result[constant.DB_DATA_NAME[0]] = result.index

    return result


def train_data_from_csv(csv_path, start_time_str=None, end_time_str=None):
    """
    从csv文件中读取训练数据
    :param csv_path:
    :param start_time_str:
    :param end_time_str:
    :return:返回值为 pandas.Series 类型，index为datetime的字符串，data是数值
    """

    device_data = pd.read_csv(csv_path)
    device_data.index = device_data['create_date']

    if start_time_str is not None and end_time_str is not None:
        data = device_data.loc[start_time_str:end_time_str, constant.DB_DATA_NAME]
    elif start_time_str is not None:
        data = device_data.loc[start_time_str:, constant.DB_DATA_NAME]
    elif end_time_str is not None:
        data = device_data.loc[:end_time_str, constant.DB_DATA_NAME]
    else:
        data = device_data.loc[:, constant.DB_DATA_NAME]

    return data


def data_check(data):
    """
    数据检查
    """
    flag = True
    fail_reason = ''

    data = data.dropna()
    data = data[data >= 0]

    if len(data.values) < constant.DATA_NUM_MIN:
        flag = False
        fail_reason = 'useful data num is %d, less than %d.' \
                      % (len(data.values), constant.DATA_NUM_MIN)

    return flag, fail_reason
