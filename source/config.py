"""
配置域名端口
"""

# 测试环境
king_host = '127.0.0.1'
ip = king_host

# 服务器部署
# ip = ''
# king_host = '0.0.0.0'

# 端口
king_port = 5001

# 是否是开发环境
DEV = False

"""
配置 swagger
"""
template = {
    "SWAGGER": {
        "openapi": "3.3"
    },
    "info": {
        "title": "异常监测算法API",
        "description": "异常监测算法的接口文档",
        "version": "3.3"
    },
    "host": ip + ':' + str(king_port),
    "basePath": "/",
}

"""
模型加载配置
"""
model_dir = '../model/'

"""
气体模型
"""
gas_model_1 = '../gas_model/gas_rf.pkl'
gas_model_2 = '../gas_model/gas_rf_2.pkl'

"""
文档目录
"""
doc_dir = '../doc/'

"""
训练日志缓存目录
"""
train_log_dir = '../train_log/'


"""
网页资源目录
"""
template_dir = '../templates/'
