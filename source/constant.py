# -*- coding: utf-8 -*-

"""
接口参数字典编码
"""

# 设备相关参数名称的编码 equipment_attributes
E_ATTR = \
    {'equipmentId': 'equipmentId', 'equipmentState': 'equipmentState', 'equipmentRssi': 'equipmentRssi',
     'createDate': 'createDate', 'leakage': 'leakage', 'temperatureA': 'temperatureA', 'temperatureB': 'temperatureB',
     'temperatureC': 'temperatureC', 'temperatureN': 'temperatureN', 'currentA': 'currentA', 'currentB': 'currentB',
     'currentC': 'currentC', 'voltageA': 'voltageA', 'voltageB': 'voltageB', 'voltageC': 'voltageC', 'rateA': 'rateA',
     'rateB': 'rateB', 'rateC': 'rateC', 'voltageAngleA': 'voltageAngleA', 'voltageAngleB': 'voltageAngleB',
     'voltageAngleC': 'voltageAngleC', 'currentAngleA': 'currentAngleA', 'currentAngleB': 'currentAngleB',
     'currentAngleC': 'currentAngleC', 'powerFactorA': 'powerFactorA', 'powerFactorB': 'powerFactorB',
     'powerFactorC': 'powerFactorC', 'powerFactorSum': 'powerFactorSum', 'powerUsefulA': 'powerUsefulA',
     'powerUsefulB': 'powerUsefulB', 'powerUsefulC': 'powerUsefulC', 'powerUsefulSum': 'powerUsefulSum',
     'powerUselessA': 'powerUselessA', 'powerUselessB': 'powerUselessB', 'powerUselessC': 'powerUselessC',
     'powerUselessSum': 'powerUselessSum', 'powerApparentA': 'powerApparentA', 'powerApparentB': 'powerApparentB',
     'powerApparentC': 'powerApparentC', 'powerApparentSum': 'powerApparentSum', 'energyUsefulA': 'energyUsefulA',
     'energyUsefulB': 'energyUsefulB', 'energyUsefulC': 'energyUsefulC', 'energyUsefulSum': 'energyUsefulSum',
     'energyUselessA': 'energyUselessA', 'energyUselessB': 'energyUselessB', 'energyUselessC': 'energyUselessC',
     'energyUselessSum': 'energyUselessSum', 'energyApparentA': 'energyApparentA', 'energyApparentB': 'energyApparentB',
     'energyApparentC': 'energyApparentC', 'energyApparentSum': 'energyApparentSum', 'balanceCurrent': 'balanceCurrent',
     'balanceVoltage': 'balanceVoltage', 'airPollution': 'airPollution', 'pm25': 'pm25'}
# 其他参数编码 other_attributes
O_ATTR = \
    {'datetimeStart': 'datetimeStart', 'datetimeEnd': 'datetimeEnd', 'includeGas': 'includeGas',
     'batchData': 'batchData'}

# 电压阈值参数编码 valve_attributes
V_ATTR = \
    {'voltageAUpper': 'voltageAUpper', 'voltageALower': 'voltageALower', 'voltageBUpper': 'voltageBUpper',
     'voltageBLower': 'voltageBLower', 'voltageCUpper': 'voltageCUpper', 'voltageCLower': 'voltageCLower'}

# 字符串类型参数(这里只是字典键名)
STR_ARGS = ['equipmentId', 'equipmentState', 'createDate',
            'datetimeStart', 'datetimeEnd', 'includeGas', 'batchData']
# 对象类型参数(这里只是字典键名)
OBJECT_ARGS = []

# 全部的参数
ARGS = {**E_ATTR, **O_ATTR, **V_ATTR}


"""
接口返回值字典编码
"""

'''返回值字典编码'''
EQUIPMENT_ID = 'equipmentId'                    # 设备编号
CREATE_DATE = 'createDate'                      # 数据产生时间
STATUS = 'status'                               # 异常检测状态(正常、异常、未知错误)
UNKNOWN_ERROR = 'unknown_error'                 # 未知错误
DETAIL = 'detail'                               # 异常检测详细信息
THRESHOLD = 'threshold'                         # 异常检测的阈值
DATA = 'data'                                   # 数据
FLAG = 'flag'                                   # 训练是否成功
REASON = 'reason'                               # 训练失败原因
MODEL_READY = 'ready'                           # 模型可以进行异常检测
MODEL_IS_TRAINING = 'model_is_training'         # 模型正在训练
MODEL_NOT_EXIST = 'model_not_exist'             # 模型不存在
MODEL_START_TRAINING = 'model_start_training'   # 模型开始训练
SUCCESS = 'success'                             # 通用成功
FAIL = 'fail'                                   # 通用失败
INCLUDE_GAS = '1'                               # 开启气体检测功能
EXCLUDE_GAS = '0'                               # 关闭气体检测功能

'''异常检测任务编码(对应 DETAIL 中的key)'''
TASK_TYPE = \
    {
        'leakage': 'leakage',               # 漏电
        'temperatureA': 'temperatureA',     # A相温度
        'temperatureB': 'temperatureB',     # B相温度
        'temperatureC': 'temperatureC',     # C相温度
        'temperatureN': 'temperatureN',     # N相温度
        'currentA': 'currentA',             # A相电流
        'currentB': 'currentB',             # B相电流
        'currentC': 'currentC',             # C相电流
        'voltageA': 'voltageA',             # A相电压
        'voltageB': 'voltageB',             # B相电压
        'voltageC': 'voltageC',             # C相电压
        'gas': 'gasType',                   # 气体类别
        'airPollution': 'airPollution',     # 有机气体
        'pm25': 'pm25'                      # pm2.5
    }
'''负载分类任务编码(对应 detail 字典的key)'''
TASK_LOAD = \
    {
        'loadA': 'loadA',   # A相负载
        'loadB': 'loadB',   # B相负载
        'loadC': 'loadC',   # C相负载
    }

'''气体类型编码(对应 detail->gasType 字典的value)'''
GAS_TYPE = \
    {
        'alcohol': 'alcohol',       # 酒精
        'incense': 'incense',       # 燃香
        'lampblack': 'lampblack',   # 油烟
        'heat': 'heat',             # 加热线缆
        'wood': 'wood',             # 燃烧木屑
        'cigarette': 'cigarette',   # 香烟
        'normal': 'normal'          # 正常
    }

'''气体分类编码(内部编码, 和GAS_TYPE的key对应)'''
# GAS_CLASS = \
#     [
#         'alcohol',
#         'incense',
#         'lampblack',
#         'heat',
#         'normal'
#     ]
GAS_CLASS = \
    [
        'alcohol',      # 酒精
        'incense',      # 燃香
        'lampblack',    # 油烟
        'heat',         # 加热线缆
        'wood',         # 燃烧木屑
        'cigarette'     # 香烟
    ]

'''异常类型编码(对应 detail->**->status 的 value )'''
ABNORMAL_TYPE = \
    {
        'normal': 'normal',                         # 通用正常
        'abnormal': 'abnormal',                     # 通用异常
        'leak': 'leak',                             # 漏电异常
        'voltage_over': 'voltage_over',             # 过压异常
        'voltage_under': 'voltage_under',           # 欠压异常
        'current_over': 'current_over',             # 过流异常（超过硬件阈值）
        'current_behavior': 'current_behavior',     # 过流异常（用电行为异常）
        'temp_over': 'temp_over',                   # 过温异常
        'gas_abnormal': 'gas_abnormal',             # 气体异常
        'airPollution_over': 'airPollution_over',   # 有机气体异常
        'pm25_over': 'pm25_over'                    # PM2.5异常
    }
DEVICE_DISABLED = 'device_disabled'     # 设备功能未开启(数值为-1.0)

'''模型类型(key)到异常类型(key)的关系'''
MODEL_TO_ABNORMAL = \
    {
        'leakage': 'leak',
        'temperatureA': 'temp_over',
        'temperatureB': 'temp_over',
        'temperatureC': 'temp_over',
        'temperatureN': 'temp_over',
        'currentA': 'current_over',     # 过流异常（超过硬件阈值）
        'currentB': 'current_over',     # 过流异常（超过硬件阈值）
        'currentC': 'current_over',     # 过流异常（超过硬件阈值）
        'voltageA': ['voltage_under', 'voltage_over'],
        'voltageB': ['voltage_under', 'voltage_over'],
        'voltageC': ['voltage_under', 'voltage_over'],
        'gas': 'gas_abnormal',
        'airPollution': 'airPollution_over',
        'pm25': 'pm25_over'
    }

"""
模型内部字典编码
"""

'''预测模型字典编码'''
FIRST_PHASE_MODEL = 'first'     # 第一阶段模型
SECOND_PHASE_MODEL = 'second'   # 第二阶段模型
SECOND_PHASE_STD = 'std1'       # 第二阶段模型的标准差

'''统计模型字典编码'''
STAT_MEAN = 'mean'              # 均值
STAT_STD = 'std2'               # 标准差
STAT_MAX = 'max'                # 最大值
STAT_MIN = 'min'                # 最小值

'''模型状态字典编码'''
MODEL_STATUS = 'model_status'   # 模型状态（可用(不同类别)/不可用）
MODEL_ENABLED_0 = 'enabled_0'         # 统计模型
MODEL_ENABLED_1 = 'enabled_1'         # 预测模型
MODEL_DISABLED = 'disabled'           # 模型不可用，或者模型被禁用

'''训练结果字典编码'''
TRAIN_STATUS = 'train_status'       # 训练结果
TRAIN_SUCCESS = 'train_success'     # 训练成功
DATA_NULL = 'data_null'             # 数据为空导致模型训练失败

'''模型类别字典编码'''
MODEL_TYPE = \
    {
        'leakage': 'leakage',
        'temperatureA': 'temperatureA',
        'temperatureB': 'temperatureB',
        'temperatureC': 'temperatureC',
        'temperatureN': 'temperatureN',
        'currentA': 'currentA',
        'currentB': 'currentB',
        'currentC': 'currentC',
        'voltageA': 'voltageA',
        'voltageB': 'voltageB',
        'voltageC': 'voltageC',
        'gas': 'gas',
        'airPollution': 'airPollution',
        'pm25': 'pm25'
    }

'''训练模型所需数据库中字段名称'''
DB_DATA_NAME = \
    [
        'create_date', 'leakage',
        'temperature_a', 'temperature_b', 'temperature_c', 'temperature_n',
        'current_a', 'current_b', 'current_c',
        'voltage_a', 'voltage_b', 'voltage_c',
    ]
'''训练气体模型所需数据库中字段名称'''
DB_DATA_NAME_GAS = \
    [
        'air_pollution', 'pm25'
    ]

'''全局变量'''
R_WINDOWS = 60              # 第二阶段标准差的窗口，单位为分钟
EASY_MODEL_STD = 1.0        # 标准差小于该值直接使用简单模型
DATA_NUM_MIN = 1440 * 1     # 训练数据需要的最小数量
FEATURES_NUM_MIN = 30       # 特征的最小数量(天数)
MIN_STD = 0.1               # 简单模型最小的方差

'''阈值初始化值'''
LEAKAGE_MAX = 5000.0        # 最大漏电值
v_under = 187.0             # 电压上阈值
v_upper = 253.0             # 电压下阈值
VALVE = \
    {
        'leakage': LEAKAGE_MAX,
        'temperatureA': 75.0,
        'temperatureB': 75.0,
        'temperatureC': 75.0,
        'temperatureN': 75.0,
        'currentA': 150.0,
        'currentB': 150.0,
        'currentC': 150.0,
        'voltageA': [v_under, v_upper],
        'voltageB': [v_under, v_upper],
        'voltageC': [v_under, v_upper],
        'airPollution': 200.0,
        'pm25': 400.0
    }

'''标准差系数初始化值，用于和统计特征一同计算阈值'''
STD_CO_INIT = 5.0
STD_CO = \
    {
        'leakage': STD_CO_INIT,
        'temperatureA': STD_CO_INIT,
        'temperatureB': STD_CO_INIT,
        'temperatureC': STD_CO_INIT,
        'temperatureN': STD_CO_INIT,
        'currentA': STD_CO_INIT,
        'currentB': STD_CO_INIT,
        'currentC': STD_CO_INIT,
        'voltageA': STD_CO_INIT,
        'voltageB': STD_CO_INIT,
        'voltageC': STD_CO_INIT,
        'airPollution': STD_CO_INIT,
        'pm25': STD_CO_INIT
    }


OFFSET_INIT = 0.0               # OFFSET的初始值
# 对阈值的调整值(最终异常检测阈值=THRESHOLD+OFFSET)
OFFSET = \
    {
        'leakage': OFFSET_INIT,
        'temperatureA': OFFSET_INIT,
        'temperatureB': OFFSET_INIT,
        'temperatureC': OFFSET_INIT,
        'temperatureN': OFFSET_INIT,
        'currentA': OFFSET_INIT,
        'currentB': OFFSET_INIT,
        'currentC': OFFSET_INIT,
        'voltageA': [OFFSET_INIT, OFFSET_INIT],
        'voltageB': [OFFSET_INIT, OFFSET_INIT],
        'voltageC': [OFFSET_INIT, OFFSET_INIT],
        'airPollution': OFFSET_INIT,
        'pm25': OFFSET_INIT
    }
