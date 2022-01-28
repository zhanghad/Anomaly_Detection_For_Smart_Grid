
def single_phase_classify(current, voltage_angle, current_angle, power_factor):
    """
    判断单相负载
    @current：当前电流
    @voltage_angle：电压角
    @current_angle：电流角
    @power_factor：功率因数

    返回值为列表[arg1, arg2, arg3]
    @arg1：负载种类：-1-当前未使用、0-阻性、1-容性、2-感性、3-有源负载，012三类负载类型同时需根据arg2和arg3综合判断
    @arg2：阻性负载比重，当其值为100时为纯阻性电路，否则根据arg1判断为阻感电路或阻容电路
    @arg3：感/容性负载比重，种类由arg1判断，若为100则为纯感/容性电路
    """
    current = float(current)
    voltage_angle = float(voltage_angle)
    current_angle = float(current_angle)
    power_factor = float(power_factor)

    result = [0, 0, 0]

    if current > 0:
        sub = current_angle - voltage_angle

        if sub > 90:
            sub = sub - 360
        elif sub < -90:
            sub = sub + 360

        if sub > 90 or sub < -90:
            result = [3, 0, 0]
        else:
            if power_factor >= 99:
                result = [0, 100, 0]
            else:
                if 1 > sub > -1:
                    result[0] = 0
                elif sub >= 1:
                    result[0] = 1
                else:
                    result[0] = 2

                if 90 >= sub > 89:
                    result[2] = 100
                elif -89 > sub >= -90:
                    result[2] = 100
                elif 1 > sub > -1:
                    result[1] = 100
                else:
                    possibility = round(abs(sub) / 90, 2) * 100
                    result[1] = 100 - possibility
                    result[2] = 100 - result[1]
    else:
        result = [-1, 0, 0]

    return result

