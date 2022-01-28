from utils import load_object
import config

gas_model = load_object(config.gas_model_2)


def gas_classify(temp, ap, pm25):
    return gas_model.predict([[temp, ap, pm25]])
