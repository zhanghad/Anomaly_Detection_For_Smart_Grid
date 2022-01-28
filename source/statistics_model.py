# -*- coding: utf-8 -*-
import constant
from utils import data_check

'''
电压、温度、气体
'''


def remove_outliers(ds):
    """
    根据四分位数去除数据中的异常点
    (针对电压、气体、温度)零值较少
    :param ds:原始数据(ds>=0)
    :return:清洗后数据(ds>=0)
    """
    q1 = ds.quantile(q=0.25)
    q3 = ds.quantile(q=0.75)
    outlier_step = 1.5 * (q3 - q1)
    ds = ds[(ds >= q1 - outlier_step) & (ds <= q3 + outlier_step)]
    return ds


def train_statistics_model(data):
    """
    统计特征模型计算
    :param data:原始数据,类型为 pd.Series,index为datetime，data为数据（可以为电压、温度、气体）
    :return:flag:训练是否成功的标志; fail_reason:失败原因; model:模型
    """

    # 返回值
    model = {}

    # 数据检查
    flag, fail_reason = data_check(data)

    # 数据存在问题
    if not flag:
        model[constant.MODEL_STATUS] = constant.MODEL_DISABLED
        return flag, fail_reason, model

    # 去除异常值
    data = remove_outliers(data)

    # 统计特征
    model[constant.MODEL_STATUS] = constant.MODEL_ENABLED_0
    model[constant.STAT_MEAN] = data.mean()
    model[constant.STAT_MAX] = data.max()
    model[constant.STAT_MIN] = data.min()
    model[constant.STAT_STD] = data.std()

    return flag, fail_reason, model
