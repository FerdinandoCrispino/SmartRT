class Prodist:
    @classmethod
    def faixa_tensao(cls, data_voltages):
        data_voltages_completed = list()

        for item in data_voltages:
            voltage_pu = item["V_mag"] / item["V_base"]

            if item["V_base"] > 230000:
                nivel_tensao = _superior_230kv(voltage_pu)

            elif 69000 < item["V_base"] <= 230000:
                nivel_tensao = _69kv_230kv(voltage_pu)

            elif 2300 < item["V_base"] <= 69000:
                nivel_tensao = _2p3kv_69kv(voltage_pu)

            elif abs(item["V_base"] - 220) < 1:
                nivel_tensao = _220v(item["V_mag"])

            elif abs(item["V_base"] - 127) < 1:
                nivel_tensao = _127v(item["V_mag"])

            elif abs(item["V_base"] - 120) < 1:
                nivel_tensao = _120v(item["V_mag"])

            elif abs(item["V_base"] - 115) < 1:
                nivel_tensao = _115v(item["V_mag"])

            elif abs(item["V_base"] - 110) < 1:
                nivel_tensao = _110v(item["V_mag"])

            data_voltages_completed.append({
                "Node": item["Node"],
                "V_mag": item["V_mag"],
                "V_pu": voltage_pu,
                "Level": nivel_tensao
            })

        return data_voltages_completed


def _superior_230kv(voltage_pu):
    if 0.95 <= voltage_pu <= 1.05:
        nivel_tensao = 'adequada'

    elif 0.93 <= voltage_pu < 0.95:
        nivel_tensao = 'sub_precaria'

    elif 1.05 < voltage_pu <= 1.07:
        nivel_tensao = 'sobre_precaria'

    elif 0.93 > voltage_pu:
        nivel_tensao = 'sub_critica'

    elif voltage_pu > 1.07:
        nivel_tensao = 'sobre_critica'

    return nivel_tensao

def _69kv_230kv(voltage_pu):
    if 0.95 <= voltage_pu <= 1.05:
        nivel_tensao = 'adequada'

    elif 0.9 <= voltage_pu < 0.95:
        nivel_tensao = 'sub_precaria'

    elif 1.05 < voltage_pu <= 1.07:
        nivel_tensao = 'sobre_precaria'

    elif 0.9 > voltage_pu:
        nivel_tensao = 'sub_critica'

    elif voltage_pu > 1.07:
        nivel_tensao = 'sobre_critica'

    return nivel_tensao

def _2p3kv_69kv(voltage_pu):
    if 0.93 <= voltage_pu <= 1.05:
        nivel_tensao = 'adequada'

    elif 0.9 <= voltage_pu < 0.93:
        nivel_tensao = 'sub_precaria'

    elif 0.9 > voltage_pu:
        nivel_tensao = 'sub_critica'

    elif voltage_pu > 1.05:
        nivel_tensao = 'sobre_critica'

    return nivel_tensao

def _220v(voltage_mag):
    if 202 <= voltage_mag <= 231:
        nivel_tensao = 'adequada'

    elif 191 <= voltage_mag < 202:
        nivel_tensao = 'sub_precaria'

    elif 231 < voltage_mag <= 233:
        nivel_tensao = 'sobre_precaria'

    elif 191 > voltage_mag:
        nivel_tensao = 'sub_critica'

    elif voltage_mag > 233:
        nivel_tensao = 'sobre_critica'

    return nivel_tensao

def _127v(voltage_mag):
    if 117 <= voltage_mag <= 133:
        nivel_tensao = 'adequada'

    elif 110 <= voltage_mag < 117:
        nivel_tensao = 'sub_precaria'

    elif 133 < voltage_mag <= 135:
        nivel_tensao = 'sobre_precaria'

    elif 110 > voltage_mag:
        nivel_tensao = 'sub_critica'

    elif voltage_mag > 135:
        nivel_tensao = 'sobre_critica'

    return nivel_tensao

def _120v(voltage_mag):
    if 110 <= voltage_mag <= 126:
        nivel_tensao = 'adequada'

    elif 104 <= voltage_mag < 110:
        nivel_tensao = 'sub_precaria'

    elif 126 < voltage_mag <= 127:
        nivel_tensao = 'sobre_precaria'

    elif 104 > voltage_mag:
        nivel_tensao = 'sub_critica'

    elif voltage_mag > 127:
        nivel_tensao = 'sobre_critica'

    return nivel_tensao

def _115v(voltage_mag):
    if 106 <= voltage_mag <= 121:
        nivel_tensao = 'adequada'

    elif 100 <= voltage_mag < 106:
        nivel_tensao = 'sub_precaria'

    elif 121 < voltage_mag <= 122:
        nivel_tensao = 'sobre_precaria'

    elif 100 > voltage_mag:
        nivel_tensao = 'sub_critica'

    elif voltage_mag > 122:
        nivel_tensao = 'sobre_critica'

    return nivel_tensao

def _110v(voltage_mag):
    if 101 <= voltage_mag <= 116:
        nivel_tensao = 'adequada'

    elif 96 <= voltage_mag < 101:
        nivel_tensao = 'sub_precaria'

    elif 116 < voltage_mag <= 117:
        nivel_tensao = 'sobre_precaria'

    elif 96 > voltage_mag:
        nivel_tensao = 'sub_critica'

    elif voltage_mag > 117:
        nivel_tensao = 'sobre_critica'

    return nivel_tensao
