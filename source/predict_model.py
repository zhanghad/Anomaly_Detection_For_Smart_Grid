# -*- coding: utf-8 -*-

# 基础
import datetime
import math
import copy
import pandas as pd
import numpy as np

# 机器学习相关
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.metrics import pairwise
from sklearn import ensemble

# 自定义
import constant
from utils import data_check

'''
电流（功率），漏电值
'''


def remove_outliers(ds):
    """
    根据四分位数去除数据中的异常点
    (针对漏电、电流)零值较多
    :param ds:原始数据(ds>=0)
    :return:清洗后数据(ds>=0)
    """
    # 计算四分位数时只考虑非零值
    ds_0 = ds[ds == 0]
    ds_1 = ds[ds > 0]
    q1 = ds_1.quantile(q=0.25)
    q3 = ds_1.quantile(q=0.75)
    outlier_step = 1.5 * (q3 - q1)
    ds_1 = ds_1[(ds_1 <= q3 + outlier_step)]
    return pd.concat([ds_0, ds_1]).sort_index()


def data_segment(data, data_normal):
    """
    数据分割，将数据分为每天一段
    :param data:未归一化的数据
    :param data_normal:归一化的数据
    :return:
        data_seg:未归一化的数据的分割;
        data_normal_seg:归一化的数据的分割;
    """

    data_normal_seg = []  # 归一化分段数据
    data_seg = []  # 原始分段数据

    last_date = data.index[0][:10]
    temp_day_normal = []
    temp_day = []
    for i in range(0, len(data)):
        if data.index[i][:10] == last_date:
            # 归一化数据
            temp_data_normal = pd.Series(index=[data_normal.index[i]], data=[data_normal.values[i]])
            temp_day_normal.append(temp_data_normal)
            # 原始数据
            temp_data = pd.Series(index=[data.index[i]], data=[data.values[i]])
            temp_day.append(temp_data)
        elif len(temp_day_normal) != 0:
            # 归一化数据
            data_normal_seg.append(pd.concat(temp_day_normal))
            temp_day_normal = []
            # 原始数据
            data_seg.append(pd.concat(temp_day))
            temp_day = []
        last_date = data.index[i][:10]

    return data_seg, data_normal_seg


def features_create(feature_n, origin_seg, normal_seg):
    """
    特征创建，每天数据的区间平均值，相等间隔时间段
    :param feature_n:特征数量
    :param origin_seg:分段的原始数据
    :param normal_seg:分段的归一化数据
    :return:
        days_features:特征;
        normal_seg_clean:原始数据的分割（与特征对齐）;
        origin_seg_clean:归一化数据的分割（与特征对齐）;
    """

    # 返回值
    days_features = []
    normal_seg_clean = []
    origin_seg_clean = []

    # 对每天进行特征创建
    for d in range(len(normal_seg)):
        # 对每天数据按 feature_n 进行分段
        temp_sp = []
        for i in range(feature_n):
            temp_sp.append([])
        for i in range(len(normal_seg[d])):
            index = int(normal_seg[d].index[i][11:13]) // (24 // feature_n)
            temp_sp[index].append(normal_seg[d][i])

        # 数据缺失判断
        exist_null = False
        for i in range(feature_n):
            if len(temp_sp[i]) == 0:
                exist_null = True
                break
        # 直接去掉缺失值
        if not exist_null:
            # 特征创建(这里取一段时间内的平均值)
            temp_f = []
            for i in range(feature_n):
                temp_f.append(np.mean(temp_sp[i]))

            # 峰谷比作为额外特征
            temp_f.append(normal_seg[d].max() - normal_seg[d].min())

            # 存储结果
            days_features.append(temp_f)
            origin_seg_clean.append(origin_seg[d])
            normal_seg_clean.append(normal_seg[d])

    return days_features, normal_seg_clean, origin_seg_clean


def features_create_custom(origin_seg, normal_seg):
    """
    特征创建，每天数据的区间平均值，自定义间隔时间段
    :param origin_seg:分段的原始数据
    :param normal_seg:分段的归一化数据
    :return:
        days_features:特征;
        normal_seg_clean:原始数据的分割（与特征对齐）;
        origin_seg_clean:归一化数据的分割（与特征对齐）;
    """

    # 返回值
    days_features = []
    normal_seg_clean = []
    origin_seg_clean = []

    for d in range(len(normal_seg)):
        f1 = []
        f2 = []
        f3 = []
        f4 = []
        for i in range(len(normal_seg[d])):
            if int(normal_seg[d].index[i][11:13]) >= 0 and int(normal_seg[d].index[i][11:13]) <= 6:
                f1.append(normal_seg[d][i])
            elif int(normal_seg[d].index[i][11:13]) >= 7 and int(normal_seg[d].index[i][11:13]) <= 11:
                f2.append(normal_seg[d][i])
            elif int(normal_seg[d].index[i][11:13]) >= 12 and int(normal_seg[d].index[i][11:13]) <= 17:
                f3.append(normal_seg[d][i])
            else:
                f4.append(normal_seg[d][i])

        # 处理特征中的缺失值,直接将该天去掉
        if len(f1) == 0 or len(f2) == 0 or len(f3) == 0 or len(f4) == 0:
            pass
        else:
            # 特征选择
            days_features.append(
                [np.mean(f1), np.mean(f2), np.mean(f3), np.mean(f4), normal_seg[d].max() - normal_seg[d].min()])
            origin_seg_clean.append(origin_seg[d])
            normal_seg_clean.append(normal_seg[d])

    return days_features, normal_seg_clean, origin_seg_clean


def mean_of_KNN(dataset, k):
    """
    k平均最近邻的距离
    :param dataset:列表数据
    :param k:第k近邻
    :return:dbscan的参数;
    """
    temp = 0
    distMatrix = pairwise.euclidean_distances(dataset)
    for i in range(len(distMatrix)):
        distMatrix[i].sort()
        temp += distMatrix[i][k + 1]
    return temp / len(dataset)


def dbscan_arg_select(days_features):
    """
    dbscan参数的选择
    :param days_features:特征
    :return:
        min_points:dbscan的参数;
        eps_arg:dbscan的参数;
    """

    min_points = len(days_features[0]) + 1
    eps_arg = mean_of_KNN(days_features, min_points)
    return min_points, eps_arg


def dbscan_clean(eps_arg, min_points, days_features, data_seg_clean, data_normal_seg_clean):
    """
    使用DBSCAN去除异常模式
    :param eps_arg:dbscan的参数
    :param min_points:dbscan的参数
    :param days_features:特征
    :param data_seg_clean:原始数据
    :param data_normal_seg_clean:归一化数据
    :return:
        days_features_dbscan:特征;
        origin_seg_clean_dbscan:原始数据;
        normal_seg_clean_dbscan:归一化数据;
    """

    days_features_dbscan = []
    origin_seg_clean_dbscan = []
    normal_seg_clean_dbscan = []

    # 使用DBSCAN去除异常模式
    dbscan = DBSCAN(eps=eps_arg, min_samples=min_points).fit(days_features)

    for i in range(0, len(dbscan.labels_)):
        # if optics.labels_[i] == -1: # 备选
        if dbscan.labels_[i] == -1:
            pass
        else:
            days_features_dbscan.append(days_features[i])
            origin_seg_clean_dbscan.append(data_seg_clean[i])
            normal_seg_clean_dbscan.append(data_normal_seg_clean[i])

    return days_features_dbscan, origin_seg_clean_dbscan, normal_seg_clean_dbscan


def kmeans_arg_select(days_features_dbscan):
    """
    k-means 参数的选择
    :param days_features_dbscan:特征
    :return:
        clusters_num:k-means聚类数量;
    """

    # k-means 参数的确定
    num_max = 20
    num_min = 1
    inertias = []
    for i in range(num_min, num_max):
        temp = KMeans(n_clusters=i, init='k-means++').fit(days_features_dbscan)
        inertias.append(temp.inertia_)

    # 1阶变化率
    inertias_1_diff = []
    for i in range(1, len(inertias) - 1):
        inertias_1_diff.append(inertias[i - 1] - inertias[i + 1])

    # 2阶变化率
    inertias_2_diff = []
    for i in range(1, len(inertias_1_diff) - 1):
        inertias_2_diff.append(inertias_1_diff[i - 1] - inertias_1_diff[i + 1])

    clusters_num = 0
    for i in range(len(inertias_2_diff)):
        if inertias_2_diff[i] <= 0.5:
            clusters_num = i
            break

    return max(clusters_num, 2)


def first_phase_features_create(data_normal_seg_clean_dbscan):
    """
    第一阶段特征准备
    :param data_normal_seg_clean_dbscan:特征
    :return:
         classification_features:分类特征;
    """

    classification_features = []
    # 选择特征,目前主要选择时间特征
    for i in range(0, len(data_normal_seg_clean_dbscan)):
        d = datetime.datetime.strptime(data_normal_seg_clean_dbscan[i].index[0], "%Y-%m-%d %H:%M:%S")

        weekday = int(d.weekday())  # 星期 0~6
        # month = int(d.month)  # 月份 1~12
        # classification_features.append([weekday, month])
        classification_features.append([weekday])
    return classification_features


def first_phase_train(features, labels):
    """
    第一阶段特模型训练
    :param features:特征
    :param labels:标签
    :return:
        clf:分类模型;
    """

    # 备选模型
    clfs = []

    # 训练第一阶段分类模型
    clfs.append(DecisionTreeClassifier().fit(features, labels))
    clfs.append(svm.SVC().fit(features, labels))
    clfs.append(ensemble.RandomForestClassifier(n_estimators=20).fit(features, labels))
    clfs.append(ensemble.AdaBoostClassifier(n_estimators=20).fit(features, labels))

    # 模型评价
    scores = []
    for i in range(len(clfs)):
        scores.append(clfs[i].score(features, labels))

    # 模型选择
    index = scores.index(max(scores, key=lambda x: x))
    return clfs[index]


def second_phase_features_labels_create(clusters_num, data_seg_clean_dbscan, classification_labels):
    """
    第二阶段特征标签准备
    :param clusters_num:聚类的数目
    :param data_seg_clean_dbscan:原始数据，用于拟合回归模型
    :param classification_labels:分类标签
    :return:
        regression_Xs:回归特征;
        regression_Ys:回归标签;
        regression_Ys_origin_std:标准差统计量;
    """

    r_windows = constant.R_WINDOWS  # 窗口宽度(分钟)
    n_windows = int(1440 // r_windows)  # 窗口数量

    regression_Xs = []
    regression_Xs_labels = []
    regression_Ys = []  # 平滑后数据
    regression_Ys_origin = []  # 未平滑的原始数据
    regression_Ys_origin_std = []  # 用于后续预测的标准差统计量

    for i in range(0, clusters_num):
        regression_Xs.append([])
        regression_Xs_labels.append([])
        regression_Ys.append([])
        regression_Ys_origin.append([])

    # 换一种组建回归训练集的方法
    # 把同一类模式的数据全部压到1天内，然后进行平滑处理
    temps = []
    temps_merge = []  # 合并掉相同时间的点
    for i in range(0, clusters_num):
        temps.append([])
        temps_merge.append([])  # 初始化固定长度的列表，长度为每天的窗口数

    # 对每天的数据
    for i in range(0, len(data_seg_clean_dbscan)):
        # 对当天的每条数据
        for j in range(0, len(data_seg_clean_dbscan[i])):
            # 将当天时间化为分钟
            time = data_seg_clean_dbscan[i].index[j]
            x = int(time[11:13]) * 60 + int(time[14:16])
            # 元组顺序：时间x(分钟),漏电真实数据, 将同一类模式的数据合并至1个列表内
            temps[classification_labels[i]].append((x, data_seg_clean_dbscan[i][j]))

    # 对每个类别
    for i in range(0, clusters_num):
        # 对每个时间点的数据
        for j in range(0, 1440):
            t_l = [x[1] for x in temps[i] if x[0] == j]
            if len(t_l) != 0:
                temps_merge[i].append((j, max(t_l)))

    # 数据存储
    for i in range(0, clusters_num):
        for d in temps_merge[i]:
            regression_Xs[i].append([d[0]])
            regression_Xs_labels[i].append(d[0])
            regression_Ys_origin[i].append(d[1])
    # 引用
    x_lable = regression_Xs_labels
    y_origin = regression_Ys_origin

    # 数据平滑(滑动平均法)
    smooth_window = 60  # 滑动平均窗口宽度(分钟)，时间上的窗口
    wide = int((1440 / smooth_window) // 2)
    for i in range(0, clusters_num):
        for j in range(len(regression_Ys_origin[i])):
            t_list = [y_origin[i][m] for m in range(len(y_origin[i]))
                      if x_lable[i][j] - wide <= x_lable[i][m] <= x_lable[i][j] + wide]
            y_smooth = np.mean(t_list)
            regression_Ys[i].append(y_smooth)

    # 计算数据各窗口（时间）的统计量
    for i in range(clusters_num):
        # 计算窗口标准差 if n_windows*60 <= x_lable[i][m] <= (n_windows+1)*60
        temp_stds = []
        for w in range(n_windows):
            t_list = [y_origin[i][m] for m in range(len(y_origin[i]))
                      if w * 60 <= x_lable[i][m] <= (w + 1) * 60]
            t_std = np.std(t_list, ddof=1)
            temp_stds.append(t_std)
        # 保存各模型各窗口的标准差
        t_std_mean = np.nanmean(temp_stds)
        temp_stds = [t_std_mean if math.isnan(x) else x for x in temp_stds]
        regression_Ys_origin_std.append(temp_stds)

    return regression_Xs, regression_Ys, regression_Ys_origin_std


def second_phase_train(clusters_num, features, labels):
    """
    第二阶段特模型训练
    :param clusters_num:聚类数量
    :param features:特征
    :param labels:标签
    :return:
        regression_models:回归模型;
    """

    # 对每个模式构建对应的回归模型
    regression_models = {}
    # 暂定 adaboost
    # regression_models['extratree'] = []
    regression_models['adaboost'] = []

    # 模型训练
    for i in range(0, clusters_num):
        # extratree = tree.ExtraTreeRegressor().fit(features[i], labels[i])
        adaboost = ensemble.AdaBoostRegressor(n_estimators=100).fit(features[i], labels[i])
        regression_models['adaboost'].append(adaboost)

    return regression_models['adaboost']


def train_predict_model(data):
    """
    正常模式模型训练
    :param data:原始数据,类型为 pd.Series,index为datetime，data为数据（可以为漏电值，电流，功率...）
    :return:
        flag:训练是否成功的标志;
        fail_reason:失败原因;
        model:模型;
    """

    # 返回值
    model = {}

    # 数据检查
    flag, fail_reason = data_check(data)

    # 数据存在问题
    if not flag:
        model[constant.MODEL_STATUS] = constant.MODEL_DISABLED
        return flag, fail_reason, model

    # 去除原始异常数据
    data = remove_outliers(data)

    # 若数据本身比较稳定，可以直接用统计分析
    if data.std() < constant.EASY_MODEL_STD:
        model[constant.MODEL_STATUS] = constant.MODEL_ENABLED_0
        model[constant.STAT_MEAN] = data.mean()
        model[constant.STAT_MAX] = data.max()
        model[constant.STAT_MIN] = data.min()
        model[constant.STAT_STD] = data.std()
        return flag, fail_reason, model

    # 数据归一化
    data_normal = copy.deepcopy(data)
    data_normal = data_normal.apply(lambda x: x / data.max())

    # 数据分割
    data_seg, data_normal_seg = data_segment(data, data_normal)

    # 特征创建
    # days_features, data_normal_seg_clean, data_seg_clean = \
    #     features_create(4, data_seg, data_normal_seg)
    days_features, data_normal_seg_clean, data_seg_clean = \
        features_create_custom(data_seg, data_normal_seg)

    if len(days_features) < constant.FEATURES_NUM_MIN:
        model[constant.MODEL_STATUS] = constant.MODEL_DISABLED
        fail_reason = 'useful day num is %d, lower than %d'\
                      %(len(days_features), constant.FEATURES_NUM_MIN)
        return flag, fail_reason, model

    # DBSCAN 参数的确定
    min_points, eps_arg = dbscan_arg_select(days_features)

    # 使用dbscan去除异常点
    days_features_dbscan, data_seg_clean_dbscan, data_normal_seg_clean_dbscan = \
        dbscan_clean(eps_arg, min_points, days_features, data_seg_clean, data_normal_seg_clean)

    # k-means 参数选择
    clusters_num = kmeans_arg_select(days_features_dbscan)

    # k-means 聚类分析
    kmeans = KMeans(n_clusters=clusters_num, init='k-means++').fit(days_features_dbscan)

    # 存在只有单个类的情况
    if max(kmeans.labels_) == 0:
        flag = False
        fail_reason = 'data is not enough.'
        model[constant.MODEL_STATUS] = constant.MODEL_DISABLED
        return flag, fail_reason, model

    # 准备第一阶段分类模型的特征及标签
    classification_features = first_phase_features_create(data_normal_seg_clean_dbscan)
    classification_labels = kmeans.labels_.tolist()

    # 第一阶段模型的训练以及选择
    clf = first_phase_train(classification_features, classification_labels)
    model[constant.FIRST_PHASE_MODEL] = clf

    # 准备第二阶段回归模型的特征及标签
    regression_Xs, regression_Ys, regression_Ys_origin_std = \
        second_phase_features_labels_create(clusters_num, data_seg_clean_dbscan, classification_labels)
    model[constant.SECOND_PHASE_STD] = regression_Ys_origin_std

    # 第二阶段模型的训练
    model[constant.SECOND_PHASE_MODEL] = \
        second_phase_train(clusters_num, regression_Xs, regression_Ys)
    # 模型可用
    model[constant.MODEL_STATUS] = constant.MODEL_ENABLED_1

    return flag, fail_reason, model