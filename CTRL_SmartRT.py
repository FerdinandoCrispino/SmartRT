
from setup_dinamico import (
    setup_dinamico_TSEA_iniciar,
    setup_dinamico_TSEA_configurar,
    setup_dinamico_TSEA_atualizar_pesos,
    setup_dinamico_TSEA_prever
)

import time
from py_dss_interface import DSS
import os
import pandas as pd
import numpy as np
import cmath
from dataclasses import dataclass, asdict, fields

def convert2polar(real, imag):
    z = complex(real, imag)
    return cmath.polar(z)

def safe_divide(numerator, denominator):
    return numerator / denominator if denominator != 0 else 0


@dataclass
class Pesos:
    voltage_list: list
    tap: int
    reg_voltage: float
    vreg: float
    ptratio: float
    v_base: float
    v_reg_pu = float
    patamar: int = 0

    def __post_init__(self):
        self.v_reg_pu = (self.vreg * self.ptratio) / self.v_base

class SmartRT:
    def __init__(self, circuit, dss_file, bus_medicao, regcontrolname, num_patamatares=17280, patamar_ini=1, patamar_fim=17280):
        self.circuit = circuit
        self.dss_file = dss_file
        self.total_patamar = num_patamatares
        self.patamar_ini = patamar_ini
        self.paramar_fim = patamar_fim
        self.bus_medicao = [item.lower() for item in bus_medicao]
        self.num_bus_medicao = len(bus_medicao)
        self.regControlName = regcontrolname
        self.pesos_list = []
        self.dss = self._read_dss_file()


    def _save_results(self):

        path_result_pesos = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados", self.circuit, "pesos.csv")
        path_result_bus = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados", self.circuit, "voltage_bus.csv")

        # save pesos results
        data = [asdict(pesos) for pesos in self.pesos_list]
        df = pd.DataFrame(data)
        df.to_csv(path_result_pesos, index=False)

        # save voltage_bus_results
        self.all_bus_kv.to_csv(path_result_bus, index=False)

    def _read_dss_file(self) -> DSS:
        """
        Leitura do arquivo 'master' sem executar o 'solve' e com os medidores desabilitados.
        :return: DSS
        """
        dss = DSS()
        dss.dssinterface.clear_all()
        #os.path.dirname(os.path.abspath(__file__))
        dss.text(f"set Datapath = '{os.path.dirname(self.dss_file)}'")
        with open(os.path.join(self.dss_file), 'r') as file:
            for line_dss in file:
                if not (line_dss.startswith('!') or line_dss.startswith('\n') or line_dss.lower().startswith(
                        'clear')):
                    dss.text(line_dss.strip('\n'))
                if 'calc' in line_dss:
                    break
        # remove meters if present in dss files
        # for name in dss.meters.names:
        #    dss.text(f"disable energymeter.{name}")

        dss.text("set mode = daily")
        dss.text("set controlmode = time")  # Todo avaliar resultado para Static
        dss.text("set tolerance = 0.0001")
        dss.text("set maxcontroliter = 100")
        dss.text("set maxiterations = 100")
        #dss.text("Set stepsize = 10m")
        dss.text(f"Set stepsize = {86400 / self.total_patamar}s")
        dss.text("set number = 1")

        segundos_totais = int(self.patamar_ini * 86400 / self.total_patamar)
        # divmod retorna (quociente, resto)
        minutos, segundos = divmod(segundos_totais, 60)
        horas, minutos = divmod(minutos, 60)
        total_sec = minutos * 60 + segundos

        dss.text(f"set time = ({horas}, {total_sec})")


        """
        first_elem = self.__first_element(dss)
        # self.dss.text(f"new monitor.{first_elem}_P element={first_elem} terminal=1 mode=1 ppolar=no")
        dss.text(f"new 'monitor.{first_elem}_i' element='{first_elem}' terminal=1 mode=0 ppolar=no")
        self.current_monitor = f'{first_elem}_i'

        dss.text(f"new 'Energymeter.{first_elem}_m' element='{first_elem}' terminal=1")
        self.current_medidor = f'{first_elem}_m'
        """
        return dss

    def configure(self):
        num_entradas = self.num_bus_medicao
        algoritmo_de_controle = setup_dinamico_TSEA_iniciar(num_entradas=num_entradas)

        setup_dinamico_TSEA_configurar(limite_inf_tensao_saida=0.93,
                                       limite_sup_tensao_saida=1.05,
                                       limites_inf_pontos=[0.93] * num_entradas,
                                       limites_sup_pontos=[1.05] * num_entradas)
        return algoritmo_de_controle


    def _set_pesos(self, bus_voltage, patamar):

        df_patamar_voltage = pd.DataFrame(bus_voltage)
        df_bus_medicao = df_patamar_voltage[
            df_patamar_voltage['bus'].isin([medicao.split('.')[0] for medicao in self.bus_medicao]) &
            df_patamar_voltage['nodes'].isin([medicao.split('.')[1] for medicao in self.bus_medicao]) &
            (df_patamar_voltage['patamar'] == patamar) ].copy()

        if df_bus_medicao.shape[0] < len(self.bus_medicao):
            print(f"Barra não encontrada! Verificar a lista de barras fornecida.")
            exit()

        self.dss.regcontrols.name = self.regControlName
        if self.dss.regcontrols.name == self.regControlName:
            tap_reg = self.dss.regcontrols.tap_number
            rreg = self.dss.regcontrols.reverse_vreg
            fvreg = self.dss.regcontrols.forward_vreg
            pt_ratio_reg = self.dss.regcontrols.pt_ratio
            self.dss.transformers.name = self.dss.regcontrols.transformer
            bus_reg_trafo = self.dss.cktelement.bus_names[1].split('.')[0]
            node_reg_trafo = self.dss.cktelement.bus_names[1].split('.')[1]
            self.dss.circuit.set_active_bus(bus_reg_trafo)
            v_base = self.dss.bus.kv_base * 1000

            volt_bus_reg = df_patamar_voltage.loc[
                (df_patamar_voltage['bus'] == bus_reg_trafo) & (df_patamar_voltage['nodes'] == node_reg_trafo) &
            (df_patamar_voltage['patamar'] == patamar)]

            # garantir a ordem das barras igual a lista de entrada das barras de medicao
            list_bus_medicao = [word for item in self.bus_medicao for word in item.split('.')]
            bus_order_map = {val: i for i, val in enumerate(list_bus_medicao)}
            df_bus_medicao.loc[:, 'bus_sort'] = df_bus_medicao['bus'].map(bus_order_map)
            df_bus_medicao = df_bus_medicao.sort_values('bus_sort').drop(columns='bus_sort')

            pesos = Pesos(voltage_list=df_bus_medicao['vln_pu'].tolist(), tap=tap_reg, patamar=patamar,
                  reg_voltage=volt_bus_reg['vln_pu'].values[0], vreg=fvreg, ptratio=pt_ratio_reg, v_base=v_base)
            print('Determinacao dos pesos ok. ')
            return pesos
        else:
            print(f'Regulador nao encontrado!')



    def solve_circuit(self):
        total_number = self.total_patamar

        voltage_bus_list_all = []
        voltage_bus_list = []


        for number in range(1, total_number + 1):
            self.dss.solution.solve()
            hour =  self.dss.solution.hour
            sec = self.dss.solution.seconds
            print(f"Patamar:{number}, hour: {hour}, seconds: {sec}")
            status = self.dss.solution.converged
            if status == 0:
                print(f'OpenDSS: File {self.dss_file} not solved to time {number}!')
                # TODO Alterar potencia e tentar a convergencia novamente.
                # executar o mesmo patamar alterando levemente a potencia
                self.dss.text(f"set number = {number}")
                self.dss.text(f"set loadmult=1.01")
                self.dss.solution.solve()
                status = self.dss.solution.converged
                if status == 0:
                    print(f'OpenDSS: File {self.dss_file} alter loadMult 1.01 and not solved to time {number}!')
                    # return False
                    continue
                else:
                    print(f'OpenDSS: File {self.dss_file} alter loadMult 1.01 and solved to time {number}!')

            all_v_mag = self.dss.circuit.buses_vmag_pu  # Tensões de fase
            all_bus_name = self.dss.circuit.nodes_names

            my_dict = dict(zip(all_bus_name, all_v_mag))
            voltage_bus_list.append(my_dict.copy())

            vll_list = []
            for bus_name in self.dss.circuit.nodes_names:
                active_bus, bus_node = bus_name.split('.', 1)
                self.dss.circuit.set_active_bus(active_bus)
                nodes = self.dss.bus.nodes

                # print(bus_name)
                if bus_node == '4':  # para desconsiderar tensao de neutro
                    continue
                # if self.dss.bus.kv_base < 1:  # para desconsiderar a baixa tensao
                #    continue
                num_nodes = len(self.dss.bus.vll) // 2
                # num_nodes = self.dss.bus.num_nodes
                # Nao existe valores de tensao de linha para barras monofosicas
                if num_nodes == 1:
                    pos = 0
                    vll_1 = 0
                    vll_pu_1 = 0
                else:
                    pos = nodes.index(int(bus_node))

                    vll_1 = round(convert2polar(self.dss.bus.vll[pos * 2],
                                                self.dss.bus.vll[(pos * 2) + 1])[0], 5)
                    vll_1 = np.float32(vll_1)
                    vll_pu_1 = round(
                        convert2polar(self.dss.bus.pu_vll[pos * 2], self.dss.bus.pu_vll[(pos * 2) + 1])[0], 5)
                    vll_pu_1 = np.float32(vll_pu_1)

                # tensoes de fase
                # print(self.dss.bus.kv_base)
                # print(self.dss.bus.vmag_angle)
                vln_1 = round(convert2polar(self.dss.bus.voltages[pos * 2],
                                            self.dss.bus.voltages[(pos * 2) + 1])[0], 5)
                vln_1 = np.float32(vln_1)
                vln_pu_1 = round(convert2polar(self.dss.bus.pu_voltages[pos * 2],
                                               self.dss.bus.pu_voltages[(pos * 2) + 1])[0], 5)
                vln_pu_1 = np.float32(vln_pu_1)

                vll_list.append([f"{bus_name.split('.')[0]}", bus_node, vll_1, vll_pu_1, vln_1, vln_pu_1,
                                 int(self.dss.bus.kv_base * 1000) ])

            for bus, nodes, vll, vll_pu, vln, vln_pu, kv_base in vll_list:
                voltage_bus_list_all.append({"patamar": number, "bus": bus, "nodes": nodes, "vll": vll, "vln": vln,
                                             "vll_pu": vll_pu, "vln_pu": vln_pu,
                                             "kv_base": kv_base})

            # Atualizacao dos pesos
            if (number-1) % 3 == 0:  # patamar multiplo de 15 segundos
                # obtem os pesos para o setup dinamico
                set_pesos = self._set_pesos(voltage_bus_list_all, number)
                self.pesos_list.append(set_pesos)
                print(set_pesos)
                # atualizar pesos
                #voltage_list = [str(x) for x in pesos.voltage_list]
                setup_dinamico_TSEA_atualizar_pesos(tensao_saida=set_pesos.reg_voltage, tenssoes_pontos=set_pesos.voltage_list,
                                                    tap_atual=set_pesos.tap)

            # obtem a previsao do setup dinamico para o proximo patamar
            if number % 48 == 0: # patamar multiplo de 4 min - 240 segundos
                setpoint = set_pesos.v_reg_pu
                result_set_point = setup_dinamico_TSEA_prever(tensao_saida=set_pesos.reg_voltage, entradas=set_pesos.voltage_list,
                                           setpoint_atual=setpoint )

                new_vreg = result_set_point * set_pesos.v_base / set_pesos.ptratio
                self.dss.regcontrols.name = self.regControlName
                self.dss.regcontrols.forward_vreg = new_vreg
                print(f'Setpoint: {result_set_point} -- {new_vreg}')


        # Save bus voltage results
        self.all_bus_kv = pd.DataFrame(voltage_bus_list_all)

        # Save result
        self._save_results()

        #self.all_bus_kv['tr_vln'] = pd.to_numeric(self.all_bus_kv['tr_vln'], errors='coerce')
        #self.all_bus_kv = self.all_bus_kv.sort_values(['patamar', 'tr_vln', 'kv_base'])

        # para transformadores fase-fase obter o valor da tensao de linha
        self.all_bus_kv.loc[self.all_bus_kv['vln_pu'] == 0, 'vln_pu'] = self.all_bus_kv['vll'] / self.all_bus_kv[
            'kv_base'] / 1000
        self.all_bus_kv['v_base'] = (self.all_bus_kv['kv_base'] * 1000).astype(int)


if __name__ == '__main__':

    dss_file = r'C:\pastaD\TSEA\SmartRT\cenarios\RMTQ1302_TSEA\DU_7_Master_391_MTQ_RMTQ1302_17280_TSEA.dss'
    circuito = 'RMTQ1302'
    pontos_de_medicao = ['mt4339274745933283mt02.1', 'mt4291205645697419mt02.1', 'mt4294449845693038mt02.1',
                         'mt4283709245476469mt02.1', 'BT430501424549936MT02.1' ]

                         # 'bt4295442945257362mt02.1'] #    , 'mt4279615845183301mt02.1']

    regcontrol = 'creg_295rt000020129c' # Atencao: node 1!

    #dss_file = r'C:\pastaD\TSEA\SmartRT\cenarios\RBOI302_TSEA\DU_7_Master_391_BOI_RBOI1302_17280.dss'
    #circuito = 'RBOI1302'
    #pontos_de_medicao = ['mt4409888436905484bo02', 'mt4419742636825865bo02', 'mt4429122636883188bo02',
    #                     'mt4433062136860652bo02', 'bt443247023686327bo02']
    #regcontrol = ''  # Atencao: node 1!

    num_patamatares = 17280             # numero total de patamares da simulação
    patamar_ini = 0                 # 2520   # numero de patamares - converter a hora de inicio da simulação em patamares
    patamar_fim = 17280             # 5000   # converter a hora de fim da simulação em patamares

    proc_time_ini = time.time()

    simul = SmartRT(circuit=circuito,
                    dss_file=dss_file,
                    bus_medicao=pontos_de_medicao,
                    num_patamatares=num_patamatares,
                    regcontrolname= regcontrol,
                    patamar_ini=patamar_ini,
                    patamar_fim=patamar_fim)

    ctr = simul.configure()

    simul.solve_circuit()

    print(f"Processo concluído em {time.time() - proc_time_ini}")
