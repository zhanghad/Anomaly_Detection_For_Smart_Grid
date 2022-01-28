# -*- coding: utf-8 -*-

# 基础
import datetime
import pandas as pd
import copy

# 自定义
import constant as C
from predict_model import train_predict_model
from statistics_model import train_statistics_model
from load_classify import single_phase_classify
from gas_classify import gas_classify


class Equipment:
    """
    对设备建模，用于异常检测
    每个对象对应一个设备
    """

    def __init__(self, device_id, include_gas):
        """
        初始化
        :param device_id:
        :param include_gas:
        """

        self.gas_enable = False                 # 是否开启气体检测功能
        if include_gas == C.INCLUDE_GAS:
            self.gas_enable = True
        self.device_id = device_id              # 设备标识符
        self.VALVE = copy.deepcopy(C.VALVE)     # 硬件阈值
        self.STD_CO = copy.deepcopy(C.STD_CO)   # 标准差系数
        self.OFFSET = copy.deepcopy(C.OFFSET)   # 阈值偏移
        self.models = {}                        # 存储各任务的模型
        self.train_log = {}                     # 训练日志
        self.train_time = ''                    # 训练开始时间
        self.data_start_time = ''               # 训练数据最早时间
        self.data_end_time = ''                 # 训练数据最晚时间

    def get_dict(self):
        """
        获取属性字典，必须是在训练完成后
        :return:
        """
        ret = {}
        ret['device_id'] = self.device_id
        ret['gas_enable'] = self.gas_enable
        ret['VALVE'] = self.VALVE
        ret['STD_CO'] = self.STD_CO
        ret['OFFSET'] = self.OFFSET
        ret['train_time'] = self.train_time
        ret['data_start_time'] = self.data_start_time
        ret['data_end_time'] = self.data_end_time
        ret['train_log'] = self.train_log
        ret['models'] = self.models

        # 将无法直接转json的机器学习模型转为名字的字符串
        for sub_dict in ret['models'].values():
            for key in sub_dict.keys():
                if key == C.FIRST_PHASE_MODEL:
                    sub_dict[key] = str(sub_dict[key])
                elif key == C.SECOND_PHASE_MODEL:
                    temp = []
                    for t in sub_dict[key]:
                        temp.append(str(t))
                    sub_dict[key] = temp
        return ret

    def train(self, data):
        """
        训练异常检测模型
        :param data:
        :return:
        """

        # 训练相关信息
        train_result = {}
        # 数据校验
        if data is None or data.empty:
            train_result[C.TRAIN_STATUS] = C.DATA_NULL
            return train_result

        dt = datetime.datetime.now()
        self.train_time = dt.strftime("%Y-%m-%d %H:%M:%S")
        self.data_start_time = data['create_date'].tolist()[0]
        self.data_end_time = data['create_date'].tolist()[-1]

        # 训练漏电检测模型
        leakage_data = pd.Series(index=data['create_date'].tolist(), data=data['leakage'].tolist())
        flag, fail_reason, model = train_predict_model(leakage_data)
        # 为避免内外编码混淆，这里用 TASK_TYPE
        train_result[C.TASK_TYPE['leakage']] = {C.FLAG: flag, C.REASON: fail_reason}
        self.models[C.MODEL_TYPE['leakage']] = model

        # 训练过流检测模型
        temp_task = ['currentA', 'currentB', 'currentC']
        temp_data = ['current_a', 'current_b', 'current_c']
        for i in range(len(temp_task)):
            temp_d = pd.Series(index=data['create_date'].tolist(), data=data[temp_data[i]].tolist())
            flag, fail_reason, model = train_predict_model(temp_d)
            train_result[C.TASK_TYPE[temp_task[i]]] = {C.FLAG: flag, C.REASON: fail_reason}
            self.models[C.MODEL_TYPE[temp_task[i]]] = model

        # 训练电压、温度异常检测模型
        v_task = ['voltageA', 'voltageB', 'voltageC']
        temp_task = ['voltageA', 'voltageB', 'voltageC',
                     'temperatureA', 'temperatureB', 'temperatureC', 'temperatureN',
                     ]
        temp_data = ['voltage_a', 'voltage_b', 'voltage_c',
                     'temperature_a', 'temperature_b', 'temperature_c', 'temperature_n',
                     'air_pollution', 'pm25']
        for i in range(len(temp_task)):
            temp_d = pd.Series(index=data['create_date'].tolist(),
                               data=data[temp_data[i]].tolist())
            flag, fail_reason, model = train_statistics_model(temp_d)
            train_result[C.TASK_TYPE[temp_task[i]]] = \
                {C.FLAG: flag, C.REASON: fail_reason}

            self.models[C.MODEL_TYPE[temp_task[i]]] = model

        # 训练气体异常检测模型
        temp_task = ['airPollution', 'pm25']
        temp_data = ['air_pollution', 'pm25']
        if self.gas_enable:
            # 开启气体检测功能
            for i in range(len(temp_task)):
                temp_d = pd.Series(index=data['create_date'].tolist(),
                                   data=data[temp_data[i]].tolist())
                flag, fail_reason, model = train_statistics_model(temp_d)
                train_result[C.TASK_TYPE[temp_task[i]]] = \
                    {C.FLAG: flag, C.REASON: fail_reason}
                self.models[C.MODEL_TYPE[temp_task[i]]] = model
        else:
            # 关闭气体检测功能
            for i in range(len(temp_task)):
                train_result[C.TASK_TYPE[temp_task[i]]] = \
                    {C.FLAG: False, C.REASON: C.MODEL_DISABLED}
            train_result[C.TASK_TYPE['gas']] = \
                {C.FLAG: False, C.REASON: C.MODEL_DISABLED}

        # 训练日志
        train_result[C.TRAIN_STATUS] = C.TRAIN_SUCCESS
        self.train_log = train_result

        return train_result

    def predict_normal_leakage(self, datetime_str):
        """
        预测正常漏电值
        :param datetime_str:
        :return:
        """

        task_type = C.MODEL_TYPE['leakage']

        d = datetime.datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
        weekday = int(d.weekday())  # 星期（0~6）
        time = int(datetime_str[11:13]) * 60 + int(datetime_str[14:16])  # 每天时间（0~1440）

        # 第一阶段预测
        first_phase_X = [[weekday]]
        first_phase_Y = self.models[task_type][C.FIRST_PHASE_MODEL].predict(first_phase_X)
        # print(first_phase_Y.type)
        # 第二阶段预测
        second_phase_X = [[time]]
        second_phase_Y = self.models[task_type][C.SECOND_PHASE_MODEL][first_phase_Y[0]].predict(second_phase_X)

        return second_phase_Y[0]

    def predict_normal_current(self, datetime_str, task_type):
        """
        预测正常电流值
        :param datetime_str:
        :param task_type:
        :return:
        """

        d = datetime.datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
        weekday = int(d.weekday())  # 星期（0~6）
        time = int(datetime_str[11:13]) * 60 + int(datetime_str[14:16])  # 每天时间（0~1440）

        # 第一阶段预测
        first_phase_X = [[weekday]]
        first_phase_Y = self.models[task_type][C.FIRST_PHASE_MODEL].predict(first_phase_X)
        # print(first_phase_Y.type)
        # 第二阶段预测
        second_phase_X = [[time]]
        second_phase_Y = self.models[task_type][C.SECOND_PHASE_MODEL][first_phase_Y[0]].predict(
            second_phase_X)

        return second_phase_Y[0]

    def is_above_valve(self, data: float, task: str):
        """
        数据>=硬件阈值?
        :param data:
        :param task:
        :return:
        """
        if task in ['voltageA', 'voltageB', 'voltageC']:
            return float(data) >= self.VALVE[task][1]
        else:
            return float(data) >= self.VALVE[task]

    def is_under_valve(self, data: float, task: str):
        """
        数据<=硬件阈值?
        只针对电压，其余均为False
        :param data:
        :param task:
        :return:
        """

        if task in ['voltageA', 'voltageB', 'voltageC']:
            return float(data) <= self.VALVE[task][0]
        else:
            return False

    def is_above_stat_model(self, model, data: float, task: str):
        """
        model_valve, data>=model_valve?
        :param task:
        :param model:
        :param data:
        :return:
        """
        #  stat + STD_CO*std + OFFSET
        if task in ['voltageA', 'voltageB', 'voltageC']:
            model_valve = model[C.STAT_MAX] + \
                          self.STD_CO[task] * model[C.STAT_STD] + \
                          self.OFFSET[task][1]

        else:
            model_valve = model[C.STAT_MAX] \
                          + self.STD_CO[task] * model[C.STAT_STD] + \
                          self.OFFSET[task]

        # flag = float(data) >= model_valve
        flag = float(data) > model_valve

        return model_valve, flag

    def is_under_stat_model(self, model, data: float, task: str):
        """
        model_valve, data<=model_valve?
        只针对电压，其余均为False
        :param model:
        :param data:
        :param task:
        :return:
        """
        #  stat + STD_CO*std + OFFSET
        if task in ['voltageA', 'voltageB', 'voltageC']:
            # 欠压
            model_valve = model[C.STAT_MIN] - \
                          self.STD_CO[task] * model[C.STAT_STD] + \
                          self.OFFSET[task][0]
            flag = float(data) <= model_valve
            return model_valve, flag
        else:
            return -1.0, False

    def is_above_predict_model(self, model, data: float, task: str, datetime_str: str):
        """
        model_valve, data>=model_valve?
        :param datetime_str:
        :param model:
        :param data:
        :param task:
        :return:
        """

        # 时间特征提取
        d = datetime.datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
        weekday = int(d.weekday())  # 星期 0~6
        minutes = int(datetime_str[11:13]) * 60 + int(datetime_str[14:16])

        pattern_id = model[C.FIRST_PHASE_MODEL].predict([[weekday]]).tolist()[0]
        time_win_id = int(minutes // C.R_WINDOWS)
        predict = self.predict_normal_current(datetime_str=datetime_str, task_type=C.TASK_TYPE[task])

        model_valve = predict + \
                      self.STD_CO[task] * model[C.SECOND_PHASE_STD][pattern_id][time_win_id] + \
                      self.OFFSET[task]

        flag = float(data) >= model_valve

        return model_valve, flag

    def anomaly_detect(self, args):
        """
        异常检测
        """

        """
        初始化异常检测返回信息（字典第1层）
        """
        result = {}
        result[C.EQUIPMENT_ID] = self.device_id  # 设备编号
        result[C.CREATE_DATE] = args[C.ARGS['createDate']]  # 数据时间
        result[C.STATUS] = C.ABNORMAL_TYPE['normal']  # 整体异常状态
        result[C.DETAIL] = {}  # 详细信息
        # 初始化异常状态(初始化为正常)
        for values in C.TASK_TYPE.values():
            result[C.DETAIL][values] = {C.STATUS: C.ABNORMAL_TYPE['normal']}
        # 初始化负载分类任务返回值(单独列出)
        for l in C.TASK_LOAD:
            result[C.DETAIL][l] = [-1.0, -1.0, -1.0]

        """
        负载分类
        """
        load_code = ['loadA', 'loadB', 'loadC']
        current_s = [args[C.ARGS['currentA']], args[C.ARGS['currentB']], args[C.ARGS['currentC']]]
        voltage_angle_s = [args[C.ARGS['voltageAngleA']], args[C.ARGS['voltageAngleB']], args[C.ARGS['voltageAngleC']]]
        current_angle_s = [args[C.ARGS['currentAngleA']], args[C.ARGS['currentAngleB']], args[C.ARGS['currentAngleC']]]
        power_factor_s = [args[C.ARGS['powerFactorA']], args[C.ARGS['powerFactorB']], args[C.ARGS['powerFactorC']]]
        for i in range(0, 3):
            load_type = single_phase_classify(current=current_s[i], voltage_angle=voltage_angle_s[i],
                                              current_angle=current_angle_s[i], power_factor=power_factor_s[i])
            result[C.DETAIL][load_code[i]] = load_type

        """
        异常检测
        """
        for task_key in C.MODEL_TYPE.keys():
            if task_key == C.MODEL_TYPE['gas']:
                # 气体分类任务后置
                continue
            # 当前任务数据
            t_data = args[C.ARGS[task_key]]
            # 当前任务结果
            t_result = result[C.DETAIL][C.TASK_TYPE[task_key]]

            if (float(t_data)) < 0 or \
                    (task_key in ['airPollution', 'pm25'] and not self.gas_enable):
                # 设备未开启气体检测功能
                t_result[C.STATUS] = C.DEVICE_DISABLED
            else:
                # 设备开启该功能
                # 当前任务模型
                t_model = self.models[C.MODEL_TYPE[task_key]]
                if task_key in ['voltageA', 'voltageB', 'voltageC']:
                    # 针对电压
                    # 使用硬件阈值进行判断
                    flag_under_valve = self.is_under_valve(t_data, task_key)
                    flag_above_valve = self.is_above_valve(t_data, task_key)
                    if t_model[C.MODEL_STATUS] == C.MODEL_ENABLED_0:
                        # 统计模型
                        valve_stat_under, flag_under_stat = \
                            self.is_under_stat_model(model=t_model, data=t_data, task=task_key)
                        valve_stat_upper, flag_above_stat = \
                            self.is_above_stat_model(model=t_model, data=t_data, task=task_key)

                        voltage_threshold = [max(self.VALVE[task_key][0], valve_stat_under),
                                             min(self.VALVE[task_key][1], valve_stat_upper)]
                        flag_voltage_under = flag_under_valve or flag_under_stat
                        flag_voltage_above = flag_above_valve or flag_above_stat
                    else:
                        # 模型失效
                        voltage_threshold = self.VALVE[task_key]
                        flag_voltage_under = flag_under_valve
                        flag_voltage_above = flag_above_valve

                    # 向 result 中保存阈值
                    t_result[C.THRESHOLD] = voltage_threshold
                    # 向 result 中保存异常检测结果
                    if flag_voltage_under:
                        # 欠压
                        t_result[C.STATUS] = C.ABNORMAL_TYPE[C.MODEL_TO_ABNORMAL[task_key][0]]
                        result[C.STATUS] = C.ABNORMAL_TYPE['abnormal']
                    elif flag_voltage_above:
                        # 过压
                        t_result[C.STATUS] = C.ABNORMAL_TYPE[C.MODEL_TO_ABNORMAL[task_key][1]]
                        result[C.STATUS] = C.ABNORMAL_TYPE['abnormal']
                else:
                    # 针对其他属性
                    # 异常检测标志
                    flag_above_valve = self.is_above_valve(t_data, task_key)    # 硬件阈值标志
                    flag_above_stat = False                                     # 统计模型阈值标志
                    flag_above_predict = False                                  # 预测模型阈值标志
                    valve_stat = 0.0                                            # 统计模型阈值
                    valve_predict = 0.0                                         # 预测模型阈值

                    if t_model[C.MODEL_STATUS] == C.MODEL_ENABLED_0:
                        # 统计模型
                        valve_stat, flag_above_stat = \
                            self.is_above_stat_model(model=t_model, data=t_data, task=task_key)
                        flag_other_above = flag_above_stat or flag_above_valve
                        other_threshold = min(valve_stat, self.VALVE[task_key])
                    elif t_model[C.MODEL_STATUS] == C.MODEL_ENABLED_1:
                        # 预测模型
                        valve_predict, flag_above_predict = \
                            self.is_above_predict_model(model=t_model, data=t_data,
                                                        task=task_key, datetime_str=args[C.ARGS['createDate']])
                        flag_other_above = flag_above_predict or flag_above_valve
                        other_threshold = min(valve_predict, self.VALVE[task_key])
                    else:
                        # 模型失效
                        flag_other_above = flag_above_valve
                        other_threshold = self.VALVE[task_key]

                    # 向 result 中保存阈值
                    t_result[C.THRESHOLD] = other_threshold
                    # 向 result 中保存异常检测结果
                    if flag_other_above:
                        # 超过异常检测阈值
                        if task_key in ['currentA', 'currentB', 'currentC']:
                            # 电流分为超过硬件阈值和行为异常两类
                            if (flag_above_stat or flag_above_predict) and (not flag_above_valve):
                                # 这里检测电流的行为异常
                                t_result[C.STATUS] = C.ABNORMAL_TYPE['current_behavior']
                            else:
                                # 这里检测电流的超出硬件阈值异常
                                t_result[C.STATUS] = C.ABNORMAL_TYPE[C.MODEL_TO_ABNORMAL[task_key]]
                                t_result[C.THRESHOLD] = self.VALVE[task_key]
                        else:
                            t_result[C.STATUS] = C.ABNORMAL_TYPE[C.MODEL_TO_ABNORMAL[task_key]]
                        result[C.STATUS] = C.ABNORMAL_TYPE['abnormal']

        if self.gas_enable:
            # 开启气体检测功能
            if (result[C.DETAIL][C.TASK_TYPE['airPollution']][C.STATUS] == C.ABNORMAL_TYPE['airPollution_over']
                    or result[C.DETAIL][C.TASK_TYPE['pm25']][C.STATUS] == C.ABNORMAL_TYPE['pm25_over']):
                # 气体异常
                # 异常气体分类

                t_max = max([args[C.ARGS['temperatureA']], args[C.ARGS['temperatureB']],
                             args[C.ARGS['temperatureC']], args[C.ARGS['temperatureN']]])
                index = gas_classify(temp=float(t_max),
                                     ap=float(args[C.ARGS['airPollution']]),
                                     pm25=float(args[C.ARGS['pm25']]))
                gas_type = C.GAS_CLASS[index[0]]
                result[C.DETAIL][C.TASK_TYPE['gas']][C.STATUS] = C.GAS_TYPE[gas_type]
        else:
            # 关闭气体检测功能
            result[C.DETAIL][C.TASK_TYPE['gas']][C.STATUS] = C.DEVICE_DISABLED

        return result
