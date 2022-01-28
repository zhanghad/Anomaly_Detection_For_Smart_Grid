# -*- coding: utf-8 -*-
# 高并发相关
from gevent import monkey
monkey.patch_all()
from gevent.pywsgi import WSGIServer
from gevent.lock import BoundedSemaphore
from threading import Thread,Lock

# 系统库
import time
import os
import copy
import traceback
import json
import argparse

# 自定义库
from anomaly_detection import Equipment
import constant as C
import config
from utils import save_object, load_object, train_data_from_db
from config import doc_dir as DD

# 基础网络库
from flask import Flask, jsonify
from flask_restful import reqparse, abort, Api, Resource
from flasgger import Swagger,swag_from

start_time = time.time()

"""
协程信号量
"""
model_sem = BoundedSemaphore(1)
train_sem = BoundedSemaphore(1)

"""
线程锁
"""
mutex = Lock()


"""
命令行参数解析
"""
cmd_parser = argparse.ArgumentParser(description='anomaly detect server.')
cmd_parser.add_argument('--host', type=str, default=config.king_host)
cmd_parser.add_argument('--ip', type=str, default=config.ip)
cmd_parser.add_argument('--port', type=int, default=config.king_port)
cmd_parser.add_argument('--dev', type=bool, default=config.DEV)
cmd_args = cmd_parser.parse_args()

'''
初始化flask
'''
# app = Flask(__name__, template_folder=config.template_dir)
app = Flask(__name__)
api = Api(app)
print('flask init success!')

'''
初始化swagger
'''
config.template['host'] = cmd_args.ip + ':' + str(cmd_args.port)
swagger = Swagger(app, template=config.template)
print('swagger init success!')

'''
加载模型
'''
model_dir = config.model_dir
MODELS = []
MODELS_ids = []
TRAINING_ids = []  # 正在更新的模型id
files = os.listdir(model_dir)
for f in files:
    if f.split('.')[-1] == 'pkl':
        temp_equipment = load_object(model_dir + f)
        MODELS_ids.append(temp_equipment.device_id)
        MODELS.append(temp_equipment)
print('model load success!')


'''
参数解析器+参数校验
'''
parser = reqparse.RequestParser()
for index, values in C.ARGS.items():
    if index in C.STR_ARGS:
        parser.add_argument(values, type=str)
    elif index in C.OBJECT_ARGS:
        parser.add_argument(values)
    else:
        parser.add_argument(values, type=float)


def abort_if_model_not_exist(equipment_id):
    if equipment_id not in MODELS_ids:
        abort(404, message="model {} doesn't exist!".format(equipment_id))


def abort_if_model_is_training(equipment_id):
    if equipment_id in TRAINING_ids:
        abort(401, message="model {} is training!".format(equipment_id))


def train_worker(args):

    # 初始化模型
    equipment_temp = Equipment(device_id=args[C.ARGS['equipmentId']],
                               include_gas=args[C.ARGS['includeGas']])
    # 获取训练数据
    train_data = train_data_from_db(device_id=args[C.ARGS['equipmentId']],
                                    include_gas=args[C.ARGS['includeGas']],
                                    start_time_str=args[C.ARGS['datetimeStart']],
                                    end_time_str=args[C.ARGS['datetimeEnd']])
    # 训练
    try:
        train_result = equipment_temp.train(data=train_data)
    except:
        traceback.print_exc()
        train_result = {C.TRAIN_STATUS: C.UNKNOWN_ERROR}

    result = {'train_result': train_result,
              'equipment_temp': equipment_temp,
              'equipment_id': args[C.ARGS['equipmentId']]}

    train_call_back(result)


def train_call_back(res):
    """
    多进程训练回调函数
    """

    # 标明全局变量
    global MODELS
    global MODELS_ids
    global TRAINING_ids

    train_result = res['train_result']
    equipment_temp = res['equipment_temp']
    equipment_id = res['equipment_id']

    # 训练成功
    if train_result[C.TRAIN_STATUS] == C.TRAIN_SUCCESS:
        # 保存模型
        save_object(equipment_temp, model_dir + '%s.pkl' % (equipment_temp.device_id))
        if equipment_temp.device_id in MODELS_ids:
            mutex.acquire()
            model_sem.acquire()
            # 修改全局变量
            MODELS[MODELS_ids.index(equipment_temp.device_id)] = equipment_temp
            model_sem.release()
            mutex.release()
        else:
            mutex.acquire()
            model_sem.acquire()
            # 修改全局变量
            MODELS.append(copy.deepcopy(equipment_temp))
            MODELS_ids.append(equipment_temp.device_id)
            model_sem.release()
            mutex.release()

    # 训练结束
    mutex.acquire()
    train_sem.acquire()
    TRAINING_ids.remove(equipment_id)
    train_sem.release()
    mutex.release()


class TrainBack(Resource):
    """
    训练模型(后台处理)
    """

    @swag_from(DD + 'trainback_put.yml')
    def put(self):
        """
        请求训练模型
        """
        args = parser.parse_args(strict=True)
        # 若正在训练则报错
        abort_if_model_is_training(args[C.ARGS['equipmentId']])
        # 标记为正在训练的模型
        train_sem.acquire()
        TRAINING_ids.append(args[C.ARGS['equipmentId']])
        train_sem.release()

        # 删除旧的训练日志缓存文件
        '''
        logfile_name = '%s%s.json' % (config.train_log_dir, args[C.ARGS['equipmentId']])
        if os.path.exists(logfile_name):
            os.remove(logfile_name)
        '''

        # 开启训练线程
        tt = Thread(target=train_worker,args=(args,))
        tt.start()

        # 返回模型开始训练的信息
        train_result = {C.TRAIN_STATUS: C.MODEL_START_TRAINING}
        response = jsonify(train_result)
        response.status_code = 201
        return response


class Train(Resource):
    """
    训练模型(保持http连接)
    """

    @swag_from(DD+'train_put.yml')
    def put(self):
        # 标明全局变量
        global MODELS
        global MODELS_ids
        global TRAINING_ids
        # 对未定义参数报错
        args = parser.parse_args(strict=True)
        # 若正在训练则报错
        abort_if_model_is_training(args[C.ARGS['equipmentId']])
        # 标记为正在训练的模型
        train_sem.acquire()
        TRAINING_ids.append(args[C.ARGS['equipmentId']])
        train_sem.release()
        # 初始化模型
        equipment_temp = Equipment(device_id=args[C.ARGS['equipmentId']],
                                   include_gas=args[C.ARGS['includeGas']])
        # 获取训练数据
        train_data = train_data_from_db(device_id=args[C.ARGS['equipmentId']],
                                        include_gas=args[C.ARGS['includeGas']],
                                        start_time_str=args[C.ARGS['datetimeStart']],
                                        end_time_str=args[C.ARGS['datetimeEnd']])
        # 训练
        try:
            train_result = equipment_temp.train(data=train_data)
        except:
            traceback.print_exc()
            train_sem.acquire()
            TRAINING_ids.remove(args[C.ARGS['equipmentId']])
            train_sem.release()
            train_result = {C.TRAIN_STATUS: C.UNKNOWN_ERROR}
            response = jsonify(train_result)
            response.status_code = 500
            return response

        # 训练成功
        if train_result[C.TRAIN_STATUS] == C.TRAIN_SUCCESS:
            s_code = 201
            # 保存模型
            save_object(equipment_temp, model_dir + '%s.pkl' % (equipment_temp.device_id))
            if equipment_temp.device_id in MODELS_ids:
                model_sem.acquire()
                # 修改全局变量
                MODELS[MODELS_ids.index(equipment_temp.device_id)] = equipment_temp
                model_sem.release()
            else:
                model_sem.acquire()
                # 修改全局变量
                MODELS.append(copy.deepcopy(equipment_temp))
                MODELS_ids.append(equipment_temp.device_id)
                model_sem.release()

        else:
            s_code = 500

        # 训练结束
        train_sem.acquire()
        TRAINING_ids.remove(args[C.ARGS['equipmentId']])
        train_sem.release()
        response = jsonify(train_result)
        response.status_code = s_code
        return response


class Delete(Resource):
    """
    删除模型
    """
    @swag_from(DD+'delete_put.yml')
    def put(self):
        # 标明全局变量
        global MODELS
        global MODELS_ids
        global TRAINING_ids
        args = parser.parse_args(strict=True)

        # 模型正在训练
        abort_if_model_is_training(args[C.ARGS['equipmentId']])
        # 模型不存在
        abort_if_model_not_exist(args[C.ARGS['equipmentId']])

        # 清除内存中的模型
        i = MODELS_ids.index(args[C.ARGS['equipmentId']])
        model_sem.acquire()
        MODELS_ids[i] = ''
        MODELS[i] = ''
        model_sem.release()

        # 删除模型文件
        os.remove(model_dir+args[C.ARGS['equipmentId']]+'.pkl')
        ret = {C.STATUS: C.SUCCESS}

        response = jsonify(ret)
        response.status_code = 200
        return response


class Trained(Resource):
    """
    获取所有可使用模型的 equipment_id
    """
    @swag_from(DD+'trained_get.yml')
    def get(self):
        # 标明全局变量
        global MODELS
        global MODELS_ids
        global TRAINING_ids
        ids = []
        for i in MODELS_ids:
            if i not in TRAINING_ids:
                ids.append(i)

        response = jsonify({C.DATA: ids, C.STATUS: C.SUCCESS})
        response.status_code = 200
        return response


class CheckModel(Resource):
    """
    检查设备的模型状态
    """

    @swag_from(DD+'check_get.yml')
    def get(self):
        # 标明全局变量
        global MODELS
        global MODELS_ids
        global TRAINING_ids
        args = parser.parse_args(strict=True)

        ret = {}
        if args[C.ARGS['equipmentId']] in TRAINING_ids:
            ret[C.STATUS] = C.MODEL_IS_TRAINING
        elif args[C.ARGS['equipmentId']] not in MODELS_ids:
            ret[C.STATUS] = C.MODEL_NOT_EXIST
        elif (args[C.ARGS['equipmentId']] in MODELS_ids) and (args[C.ARGS['equipmentId']] not in TRAINING_ids):
            ret[C.STATUS] = C.MODEL_READY
        response = jsonify(ret)
        response.status_code = 200

        return response


class ModelDetail(Resource):
    """
    获取设备模型的详细信息
    """

    @swag_from(DD+'model_get.yml')
    def get(self):
        # 标明全局变量
        global MODELS
        global MODELS_ids
        global TRAINING_ids
        args = parser.parse_args(strict=True)

        # 模型正在训练
        abort_if_model_is_training(args[C.ARGS['equipmentId']])
        # 模型不存在
        abort_if_model_not_exist(args[C.ARGS['equipmentId']])

        obj = MODELS[MODELS_ids.index(args[C.ARGS['equipmentId']])]

        response = jsonify(obj.get_dict())
        response.status_code = 200
        return response


class Detect(Resource):
    """
    异常检测
    """

    @swag_from(DD+'detect_post.yml')
    def post(self):
        # 标明全局变量
        global MODELS
        global MODELS_ids
        global TRAINING_ids

        args = parser.parse_args(strict=True)
        # 模型正在训练
        abort_if_model_is_training(args[C.ARGS['equipmentId']])
        # 模型不存在
        abort_if_model_not_exist(args[C.ARGS['equipmentId']])

        try:
            result = MODELS[MODELS_ids.index(args[C.ARGS['equipmentId']])].\
                anomaly_detect(copy.deepcopy(args))

        except:
            traceback.print_exc()
            result = {}
            result[C.EQUIPMENT_ID] = args[C.ARGS['equipmentId']]
            result[C.CREATE_DATE] = args[C.ARGS['createDate']]
            result[C.STATUS] = C.UNKNOWN_ERROR
            response = jsonify(result)
            response.status_code = 500
            return response

        response = jsonify(result)
        response.status_code = 200
        
        return response


class BatchDetect(Resource):
    """
    批量异常检测
    """

    @swag_from(DD+'batchdetect_post.yml')
    def post(self):
        # 标明全局变量
        global MODELS
        global MODELS_ids
        global TRAINING_ids
        args = parser.parse_args(strict=True)

        # 传递过来为json字符串
        datas = args[C.ARGS['batchData']]
        datas_dict = json.loads(datas)
        data_list = datas_dict[C.DATA]
        results = {C.DATA: []}

        for data in data_list:
            try:
                result = MODELS[MODELS_ids.index(data[C.ARGS['equipmentId']])]. \
                    anomaly_detect(copy.deepcopy(data))
                results[C.DATA].append(result)
            except:
                traceback.print_exc()
                result = {}
                result[C.EQUIPMENT_ID] = args[C.ARGS['equipmentId']]
                result[C.CREATE_DATE] = args[C.ARGS['createDate']]
                result[C.STATUS] = C.UNKNOWN_ERROR
                results[C.DATA].append(result)

        response = jsonify(results)
        response.status_code = 200
        return response


class ModelValve(Resource):
    """
    查询和修改模型的边界值
    """

    @swag_from(DD+'valve_get.yml')
    def get(self):
        # 标明全局变量
        global MODELS
        global MODELS_ids
        global TRAINING_ids
        args = parser.parse_args(strict=True)
        # 模型正在训练
        abort_if_model_is_training(args[C.ARGS['equipmentId']])
        # 模型不存在
        abort_if_model_not_exist(args[C.ARGS['equipmentId']])

        result = MODELS[MODELS_ids.index(args[C.ARGS['equipmentId']])].VALVE

        response = jsonify(result)
        response.status_code = 200
        return response

    @swag_from(DD+'valve_post.yml')
    def post(self):
        # 标明全局变量
        global MODELS
        global MODELS_ids
        global TRAINING_ids
        args = parser.parse_args(strict=True)
        # 模型不存在
        abort_if_model_not_exist(args[C.ARGS['equipmentId']])
        # 模型正在训练
        abort_if_model_is_training(args[C.ARGS['equipmentId']])

        t_valve = {C.MODEL_TYPE['leakage']: args[C.ARGS['leakage']],
                   C.MODEL_TYPE['temperatureA']: args[C.ARGS['temperatureA']],
                   C.MODEL_TYPE['temperatureB']: args[C.ARGS['temperatureB']],
                   C.MODEL_TYPE['temperatureC']: args[C.ARGS['temperatureC']],
                   C.MODEL_TYPE['temperatureN']: args[C.ARGS['temperatureN']],
                   C.MODEL_TYPE['currentA']: args[C.ARGS['currentA']],
                   C.MODEL_TYPE['currentB']: args[C.ARGS['currentB']],
                   C.MODEL_TYPE['currentC']: args[C.ARGS['currentC']],
                   C.MODEL_TYPE['voltageA']: [args[C.ARGS['voltageALower']], args[C.ARGS['voltageAUpper']]],
                   C.MODEL_TYPE['voltageB']: [args[C.ARGS['voltageBLower']], args[C.ARGS['voltageBUpper']]],
                   C.MODEL_TYPE['voltageC']: [args[C.ARGS['voltageCLower']], args[C.ARGS['voltageCUpper']]],
                   C.MODEL_TYPE['airPollution']: args[C.ARGS['airPollution']],
                   C.MODEL_TYPE['pm25']: args[C.ARGS['pm25']]}

        # 引用
        model = MODELS[MODELS_ids.index(args[C.ARGS['equipmentId']])]
        # 更新
        model_sem.acquire()
        model.VALVE = copy.deepcopy(t_valve)
        model_sem.release()
        # 保存
        save_object(model, model_dir + '%s.pkl' % (model.device_id))

        response = jsonify({C.STATUS: C.SUCCESS})
        response.status_code = 200
        return response


class ModelOffset(Resource):
    """
    查询和修改模型的阈值偏移
    """

    @swag_from(DD+'offset_get.yml')
    def get(self):
        # 标明全局变量
        global MODELS
        global MODELS_ids
        global TRAINING_ids
        args = parser.parse_args(strict=True)
        # 模型正在训练
        abort_if_model_is_training(args[C.ARGS['equipmentId']])
        # 模型不存在
        abort_if_model_not_exist(args[C.ARGS['equipmentId']])

        result = MODELS[MODELS_ids.index(args[C.ARGS['equipmentId']])].OFFSET

        response = jsonify(result)
        response.status_code = 200
        return response

    @swag_from(DD+'offset_post.yml')
    def post(self):
        # 标明全局变量
        global MODELS
        global MODELS_ids
        global TRAINING_ids
        args = parser.parse_args(strict=True)
        # 模型不存在
        abort_if_model_not_exist(args[C.ARGS['equipmentId']])
        # 模型正在训练
        abort_if_model_is_training(args[C.ARGS['equipmentId']])

        t_offset = {C.MODEL_TYPE['leakage']: args[C.ARGS['leakage']],
                    C.MODEL_TYPE['temperatureA']: args[C.ARGS['temperatureA']],
                    C.MODEL_TYPE['temperatureB']: args[C.ARGS['temperatureB']],
                    C.MODEL_TYPE['temperatureC']: args[C.ARGS['temperatureC']],
                    C.MODEL_TYPE['temperatureN']: args[C.ARGS['temperatureN']],
                    C.MODEL_TYPE['currentA']: args[C.ARGS['currentA']],
                    C.MODEL_TYPE['currentB']: args[C.ARGS['currentB']],
                    C.MODEL_TYPE['currentC']: args[C.ARGS['currentC']],
                    C.MODEL_TYPE['voltageA']: [args[C.ARGS['voltageALower']], args[C.ARGS['voltageAUpper']]],
                    C.MODEL_TYPE['voltageB']: [args[C.ARGS['voltageBLower']], args[C.ARGS['voltageBUpper']]],
                    C.MODEL_TYPE['voltageC']: [args[C.ARGS['voltageCLower']], args[C.ARGS['voltageCUpper']]],
                    C.MODEL_TYPE['airPollution']: args[C.ARGS['airPollution']],
                    C.MODEL_TYPE['pm25']: args[C.ARGS['pm25']]}

        # 引用
        model = MODELS[MODELS_ids.index(args[C.ARGS['equipmentId']])]
        # 更新
        model_sem.acquire()
        model.OFFSET = copy.deepcopy(t_offset)
        model_sem.release()
        # 保存
        save_object(model, model_dir + '%s.pkl' % (model.device_id))

        response = jsonify({C.STATUS: C.SUCCESS})
        response.status_code = 200
        return response


class ModelStdCO(Resource):
    """
    查询和修改模型的标准差系数
    """

    @swag_from(DD+'stdco_get.yml')
    def get(self):
        # 标明全局变量
        global MODELS
        global MODELS_ids
        global TRAINING_ids
        args = parser.parse_args(strict=True)
        # 模型正在训练
        abort_if_model_is_training(args[C.ARGS['equipmentId']])
        # 模型不存在
        abort_if_model_not_exist(args[C.ARGS['equipmentId']])

        result = MODELS[MODELS_ids.index(args[C.ARGS['equipmentId']])].STD_CO

        response = jsonify(result)
        response.status_code = 200
        return response

    @swag_from(DD+'stdco_post.yml')
    def post(self):
        # 标明全局变量
        global MODELS
        global MODELS_ids
        global TRAINING_ids
        args = parser.parse_args(strict=True)
        # 模型不存在
        abort_if_model_not_exist(args['equipmentId'])
        # 模型正在训练
        abort_if_model_is_training(args['equipmentId'])

        t_stdco = {C.MODEL_TYPE['leakage']: args[C.ARGS['leakage']],
                    C.MODEL_TYPE['temperatureA']: args[C.ARGS['temperatureA']],
                    C.MODEL_TYPE['temperatureB']: args[C.ARGS['temperatureB']],
                    C.MODEL_TYPE['temperatureC']: args[C.ARGS['temperatureC']],
                    C.MODEL_TYPE['temperatureN']: args[C.ARGS['temperatureN']],
                    C.MODEL_TYPE['currentA']: args[C.ARGS['currentA']],
                    C.MODEL_TYPE['currentB']: args[C.ARGS['currentB']],
                    C.MODEL_TYPE['currentC']: args[C.ARGS['currentC']],
                    C.MODEL_TYPE['voltageA']: args[C.ARGS['voltageA']],
                    C.MODEL_TYPE['voltageB']: args[C.ARGS['voltageB']],
                    C.MODEL_TYPE['voltageC']: args[C.ARGS['voltageC']],
                    C.MODEL_TYPE['airPollution']: args[C.ARGS['airPollution']],
                    C.MODEL_TYPE['pm25']: args[C.ARGS['pm25']]}

        # 引用
        model = MODELS[MODELS_ids.index(args[C.ARGS['equipmentId']])]
        # 更新
        model_sem.acquire()
        model.STD_CO = copy.deepcopy(t_stdco)
        model_sem.release()
        # 保存
        save_object(model, model_dir + '%s.pkl' % (model.device_id))

        response = jsonify({C.STATUS: C.SUCCESS})
        response.status_code = 200
        return response


# 映射地址
api.add_resource(CheckModel, '/check/')
api.add_resource(Trained, '/trained/')
api.add_resource(Train, '/train/')
api.add_resource(TrainBack, '/trainback/')
api.add_resource(Delete, '/delete/')
api.add_resource(Detect, '/detect/')
api.add_resource(BatchDetect, '/batchdetect/')
api.add_resource(ModelValve, '/valve/')
api.add_resource(ModelOffset, '/offset/')
api.add_resource(ModelStdCO, '/stdco/')
api.add_resource(ModelDetail, '/model/')

if __name__ == '__main__':

    if cmd_args.dev:
        print('development server start at %s:%d!'%(cmd_args.host, cmd_args.port))
        end_time = time.time()
        print('time cost: %.4f s'%(end_time-start_time))
        app.run(host=cmd_args.host, port=cmd_args.port, debug=False, threaded=False)  # 开发环境
    else:
        # 单进程+多协程
        print('production server start at %s:%d!'%(cmd_args.host, cmd_args.port))
        end_time = time.time()
        print('time cost: %.4f s' % (end_time - start_time))
        WSGIServer((cmd_args.host, cmd_args.port), app).serve_forever()  # 生产环境




