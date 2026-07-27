import enum
import math
import time
from datetime import datetime
from py_dss_interface import DSS
import os
import pandas as pd
import numpy as np
import cmath
import yaml
from collections import Counter
from dataclasses import dataclass, asdict
import logging

from sqlalchemy.sql.coercions import expect

from create_result_db import insert_data_analise, insert_data, check_cenario_exist, nome_tabelas

logging.basicConfig(filename='CTRL_SmartRT_new.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d,%H:%M:%S')

class CapControl(enum.Enum):
    CURRENTCONTROL = 0
    VOLTAGECONTROL = 1
    KVARCONTROL = 2
    TIMECONTROL = 3
    PFCONTROL = 4

def load_config_cenario(config_path="config_smartCAP.yml"):
    application_path = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(application_path, config_path), 'r') as file:
        config = yaml.load(file, Loader=yaml.BaseLoader)

    #config = config.get("databases", {}).get(circuito)
    # config = config.get(cenario)
    if not config:
        raise ValueError(f"Configurações para não foram encontradas.")
    return config


def convert2polar(real, imag):
    z = complex(real, imag)
    return cmath.polar(z)


def safe_divide(numerator, denominator):
    return numerator / denominator if denominator != 0 else 0


def gerador_positivo_negativo():
    numero = 1
    while True:
        yield numero
        yield -numero
        numero += 1



class SmartRT:
    def __init__(self, cenario, cenario_id, circuito, dist, sub, dss_file, num_patamatares=17280, patamar_ini=1, patamar_fim=17280):
        self.cenario = cenario
        self.cenario_id = cenario_id
        self.circuito = circuito
        self.dist = dist
        self.sub = sub
        self.dss_file = dss_file
        self.total_patamar = num_patamatares
        self.patamar_ini = patamar_ini
        self.patamar_fim = patamar_fim

        # ensure DSS is ready
        self.dss = self._read_dss_file()

        # Verifica a correta atribuição das tensões de base para os transformadores não trifásicos
        print(f'Check kV_basse')
        self.__check_kv_base()

        # add monitor at first element
        first_elem = self.__first_element()
        self.monitor = f'{first_elem}_powers'
        self.dss.text(f"new monitor.{first_elem}_powers element={first_elem} terminal=1 mode=1 ppolar=no")

        # Verifica o cenario e as modificações do DSS necessarias
        self.__edit_elements()

        # verifica se já existe o cenario no banco de dados e apaga os dados para serem substituidos
        check_cenario_exist(nome_tabelas(), circuito, cenario_id)

        # salva os dados de configuração do cenario
        self.__save_cenario()

    def __edit_elements(self):
        """ Implementa modificações no OpenDSS para a criação do cenario"""


        if self.cenario_id == '14':
            for ctrl_cap in self.dss.capcontrols.names:
                self.dss.capcontrols.name = ctrl_cap  # ativa o controle do apacitor
                self.dss.capcontrols.mode = 4         # PF
                self.dss.capcontrols.on_setting = 0.90
                self.dss.capcontrols.off_setting = 0.95
            print(f'Capacitores alterados para PFControl!')

        elif self.cenario_id == '12':
            for ctrl_cap in self.dss.capcontrols.names:
                self.dss.capcontrols.name = ctrl_cap  # ativa o controle do apacitor
                self.dss.capcontrols.mode = 2         # KVARCONTROL
                self.dss.capcontrols.on_setting = 300
                self.dss.capcontrols.off_setting = 100
            print(f'Capacitores alterados para PFControl!')

        elif self.cenario_id == '11':
            # remover capacitores
            capacitor_names = self.dss.capacitors.names
            for capacitor_name in capacitor_names:
                self.dss.text(f"disable capacitor.{capacitor_name}")
            print(f'Capacitores desabilitados!')

        elif self.cenario_id == '10':
            print(f'Cenário: {self.cenario_id} - Sem alterações no DSS.')

        else:
            print(f'Cenário não definido {self.cenario_id}')

    def __first_element(self):
        """ Retorna o primeiro bus do circuito
            Navega pela topologia da rede de um bus qualquer ate o inicio do circuito
        """
        self.dss.topology.first()
        self.dss.topology.forward_branch()
        while True:
            index_branch = self.dss.topology.backward_branch()
            if index_branch:  # chegou no inicio do alimentador (Vsource)
                self.dss.topology.forward_branch()  # avançar para obter o primeiro elemento
                # print(self.dss.topology.branch_name)
                return self.dss.topology.branch_name

    def _read_dss_file(self) -> DSS:
        """
        Leitura do arquivo 'master' sem executar o 'solve' e com os medidores desabilitados.
        :return: DSS
        """
        dss = DSS()
        dss.dssinterface.clear_all()
        dss.text(f"set Datapath = '{os.path.dirname(self.dss_file)}'")
        with open(os.path.join(self.dss_file), 'r') as file:
            for line_dss in file:
                if not (line_dss.startswith('!') or line_dss.startswith('\n') or line_dss.lower().startswith(
                        'clear')):
                    dss.text(line_dss.strip('\n'))
                if 'calc' in line_dss:
                    break

        dss.text("set mode = daily")
        dss.text("set controlmode = time")  # Todo avaliar resultado para Static
        dss.text("set tolerance = 0.0001")
        dss.text("set maxcontroliter = 100")
        dss.text("set maxiterations = 100")
        dss.text(f"Set stepsize = {86400 / self.total_patamar}s")
        dss.text("set number = 1")

        segundos_totais = int(self.patamar_ini * 86400 / self.total_patamar)
        minutos, segundos = divmod(segundos_totais, 60)
        horas, minutos = divmod(minutos, 60)
        total_sec = minutos * 60 + segundos

        dss.text(f"set time = ({horas}, {total_sec})")

        return dss

    def list_trafo_delta_aberto(self):
        """
        Identifica grupo de transformadores monofasico em delta aberto atraves da
        contagem das ligações das barras dos transformadores
        :return:
        """
        dss = self.dss
        tr_map = {}
        trafos = dss.transformers.names

        dss.transformers.first()
        for _ in range(dss.transformers.count):
            dss.circuit.set_active_element(f"transformer.{dss.transformers.name}")
            if dss.cktelement.num_phases == 1:
                nome_barra_sec = dss.cktelement._bus_names()[1].split('.')[0].lower()
                nos_barra_sec = dss.cktelement._bus_names()[1].split('.', 1)[1]
                tr_map[dss.transformers.name] = (nome_barra_sec)
            dss.transformers.next()

        return Counter(tr_map.values())

    def __check_kv_base(self):
        """
        Verifica a tensão de base definida pelo openDSS para as todas as barras conectadas
        no secundario dos transformadores.
        São obtidas as tensões de fase para a barra do secundario do TR e comparada com a informada pelo openDSS
        Em caso de diferença são localizadas todas barras conectadas no secundario do transformador e set o kv_base
        de todas as barras com o valor obtido da avaliação das conecções do transformador.
        :return:
        """
        # identifica a tensão de linha e de fase para cada transformador
        dss = self.dss
        tr_map = {}

        vln = vll = None

        nome_barra_trafo_delta = self.list_trafo_delta_aberto()

        dss.transformers.first()
        for _ in range(dss.transformers.count):
            dss.circuit.set_active_element(f"transformer.{dss.transformers.name}")

            nome_barra_sec = dss.cktelement.bus_names[1].split('.')[0].lower()
            num_trafo_nomofasicos = nome_barra_trafo_delta[nome_barra_sec]

            tr_ph = dss.cktelement.num_phases
            if (tr_ph == 3) or (num_trafo_nomofasicos == 3):
                dss.transformers.wdg = 2
                vll = dss.transformers.kv
                vln = dss.transformers.kv / np.sqrt(3)
            elif tr_ph == 1:
                num_wdg = dss.transformers.num_windings
                if num_wdg == 2:
                    dss.transformers.wdg = 2  # monofasico
                    if dss.transformers.is_delta:
                        vll = dss.transformers.kv
                        vln = vll / 2
                    else:
                        vln = dss.transformers.kv
                        vll = vln * 2
                elif num_wdg == 3:  # monofasico MRT
                    dss.transformers.wdg = 2
                    vln = dss.transformers.kv
                    vll = 2 * vln

            tr_map[dss.transformers.name] = (round(vll, 3), round(vln, 3))

            bus_name = dss.cktelement.bus_names
            element_name = dss.cktelement.name
            dss.circuit.set_active_bus(bus_name[1])
            bus_name1 = dss.bus.name
            kv_base = dss.bus.kv_base
            # Verifica se ha diferença entre o calculado e o descrito pelo opnDSS
            if round(vln, 3) != round(kv_base, 3):
                print(f'{element_name}: {bus_name1}: {kv_base}: {vln}')
                # todo testar para ver se setar a tensão de linha e a tensão de fase fazem diferença !!!!
                # dss.text(f'SetkVBase Bus={bus_name1} kVLL={vll}')
                dss.text(f'SetkVBase Bus={bus_name1} kVLN={vln}')
                print(f'Valor alterado: {dss.cktelement.bus_names[1]} - kvbase:{dss.bus.kv_base}')

                # Localozar o transformador que foi alterado o valor de kvbase atraves da topologia
                dss.topology.first()
                while True:
                    indx = dss.topology.active_branch
                    indx_level = dss.topology.active_level
                    branch_name = dss.topology.branch_name
                    if branch_name == element_name:
                        dss.circuit.set_active_element(element_name)
                        dss.circuit.set_active_bus(dss.cktelement.bus_names[1])
                        # encontrou o transformador que foi alterado com setkvbase
                        break
                    index_branch = dss.topology.forward_branch()

                # busca os ramais conectados neste transformador
                while True:
                    index_branch_2 = dss.topology.next()
                    indx_level_2 = dss.topology.active_level
                    branch_name_2 = dss.topology.branch_name
                    if not dss.topology.branch_name.startswith(('Line.sbt', 'Line.rbt')):
                        print('\n Proximo transformador !!! \n')
                        break
                    # sekvbase aqui
                    dss.circuit.set_active_element(branch_name_2)
                    dss.circuit.set_active_bus(dss.cktelement.bus_names[1])
                    bus_line_name = dss.bus.name
                    kv_base_2 = dss.bus.kv_base
                    print(f'{branch_name_2}: {dss.cktelement.bus_names}: {kv_base_2}')
                    #dss.text(f'SetkVBase Bus={bus_name1} kVLL={vll}')
                    dss.text(f'SetkVBase Bus={bus_line_name} kVLN={vln}')
                    print(f'Valor alterado: {dss.cktelement.bus_names[1]} - kvbase:{dss.bus.kv_base}')

            dss.transformers.next()

        self._transformer_kv_map = tr_map

    def __csv2db(self, path_csv, patamar):
        tabela = 'linha'
        dados = pd.read_csv(path_csv)
        dados = dados[['Name', ' Distance1',' puV1',' Distance2',' puV2',' Color',' Linetype']]
        dados.columns = ['linha', 'distancia1', 'v1', 'distancia2', 'v2', 'node', 'tipo']
        dados['cenario_id'] = self.cenario_id
        dados['patamar'] = patamar
        dados['circuito'] = self.circuito
        dados = dados.to_dict('records')
        try:
            insert_data(tabela, dados)
        except:
            print('Error: linha -  insert database...')

    def __save_results_db(self, tabela, dados):
        try:
            insert_data(tabela, dados)
        except:
            print('Error: insert database...')

    def __save_cenario(self):
        data = {'cenario': self.cenario, 'cenario_id': self.cenario_id, 'empresa': self.dist, 'sub': self.sub,
                'circuito': self.circuito, 'patamar_ini': self.patamar_ini, 'patamar_fim': self.patamar_fim,
                'data': datetime.today().strftime('%Y-%m-%d')}
        try:
            insert_data('analise', [data])
        except:
            print('Error: insert database...')

    def _read_power(self, number):
        powers_rows = []

        perdas = self.dss.circuit.losses

        self.dss.monitors.name = self.monitor
        # Typical mode=65 mapping: Channel 1 (Ph1), Channel 3 (Ph2), Channel 5 (Ph3)
        p_phase1 = np.array(self.dss.monitors.channel(1))[number]
        p_phase2 = np.array(self.dss.monitors.channel(3))[number]
        p_phase3 = np.array(self.dss.monitors.channel(5))[number]

        q_phase1 = np.array(self.dss.monitors.channel(2))[number]
        q_phase2 = np.array(self.dss.monitors.channel(4))[number]
        q_phase3 = np.array(self.dss.monitors.channel(6))[number]

        powers_rows.append({
            "cenario_id": self.cenario_id,
            "circuito": self.circuito,
            "patamar": number,
            "p1": p_phase1,
            "p2": p_phase2,
            "p3": p_phase3,
            "q1": q_phase1,
            "q2": q_phase2,
            "q3": q_phase3,
            "p_losses": perdas[0]/1000,
            "q_losses": perdas[1]/1000,

        })

        self.__save_results_db("circuito", powers_rows)

    def _read_voltage(self, number):
        # Faz a leitura dos dados das tensões das barras
        current_voltage_rows = []

        for bus_name in self.dss.circuit.nodes_names:
            active_bus, bus_node = bus_name.split('.', 1)
            self.dss.circuit.set_active_bus(active_bus)
            nodes = self.dss.bus.nodes

            if bus_node == '4':
                continue

            num_nodes = len(self.dss.bus.vll) // 2
            if num_nodes == 1:
                pos = 0
                vll_1 = 0
                vll_pu_1 = 0
            else:
                pos = nodes.index(int(bus_node))
                vll_1 = round(convert2polar(self.dss.bus.vll[pos * 2], self.dss.bus.vll[(pos * 2) + 1])[0], 5)
                # vll_1 = np.float32(vll_1)
                vll_pu_1 = round(convert2polar(self.dss.bus.pu_vll[pos * 2], self.dss.bus.pu_vll[(pos * 2) + 1])[0],
                                 5)
                # vll_pu_1 = np.float32(vll_pu_1)

            vln_1 = round(convert2polar(self.dss.bus.voltages[pos * 2], self.dss.bus.voltages[(pos * 2) + 1])[0], 5)
            # vln_1 = np.float32(vln_1)
            vln_pu_1 = round(
                convert2polar(self.dss.bus.pu_voltages[pos * 2], self.dss.bus.pu_voltages[(pos * 2) + 1])[0], 5)
            # vln_pu_1 = np.float32(vln_pu_1)

            # para transformadores fase-fase não existe tensão de fase, usar o valor da tensão de linha em pu
            if math.isnan(vln_pu_1) or vln_pu_1 == 0:
                vln_pu_1 = vll_pu_1
                vln = vll_1 / 2

            kv_base = int(self.dss.bus.kv_base * 1000)

            bus_tipo = 1
            if 2300 > kv_base <= 69000:
                bus_tipo = 2
            elif kv_base > 69000:
                bus_tipo = 3

            distancia = self.dss.bus.distance

            # O OpenDSS retorna um array: [P1, Q1, P2, Q2, ...] para o Terminal 1 (De)
            # seguido de [P1, Q1, P2, Q2, ...] para o Terminal 2 (Para)
            potencias = self.dss.cktelement.powers


            # Terminal 1 (Bus1)
            p_ind1 = pos * 2
            q_ind1 = p_ind1 + 1
            p1 = potencias[p_ind1]
            q1 = potencias[q_ind1]

            # Calcular a potência aparente S = sqrt(P^2 + Q^2)
            s = math.sqrt(p1 ** 2 + q1 ** 2)
            fp = 1
            if s > 0.01:
                fp = p1 / s

            current_voltage_rows.append({
                "cenario_id": self.cenario_id,
                "circuito": self.circuito,
                "patamar": number,
                "bus": f"{bus_name.split('.')[0]}".lower(),
                "node": bus_node,
                "tipo": bus_tipo,
                #"vln": vln_1,
                "vln_pu": vln_pu_1,
                "kv_base": kv_base,
                "fp": fp,
                "distancia": distancia
            })



        self.__save_results_db("barra", current_voltage_rows)

    def __read_cap(self, number):
        cap_rows = []
        cap_dados_rows = []
        vmag = []
        cap = kvar = kv_base = ctrl_mode = current_steps = available_steps = 0
        pt_ratio = ct_ratio = ctrl_on = ctrl_off = 0
        delay = delay_off = dead_time = 0

        for ctrl_cap in self.dss.capcontrols.names:
            self.dss.capcontrols.name = ctrl_cap        # ativa controle do apacitor

            delay = self.dss.capcontrols.delay
            delay_off = self.dss.capcontrols.delay_off
            dead_time = self.dss.capcontrols.dead_time

            ctrl_cap = self.dss.capcontrols.controlled_capacitor
            ctrl_mode = self.dss.capcontrols.mode
            if ctrl_mode == 1:
                pt_ratio = self.dss.capcontrols.pt_ratio
                ctrl_on = self.dss.capcontrols.on_setting
                ctrl_off = self.dss.capcontrols.off_setting


            for cap in self.dss.capacitors.names:
                if ctrl_cap == cap:
                    nome_cap = cap
                    self.dss.capacitors.name = nome_cap      # ativa o capacitor

                    ctrl_bus = self.dss.cktelement.bus_names
                    current_steps = self.dss.capacitors.states[0]
                    available_steps = self.dss.capacitors.available_steps
                    kvar = self.dss.capacitors.kvar

                    self.dss.circuit.set_active_bus(ctrl_bus[0])            # ativa o bus
                    kv_base = self.dss.bus.kv_base
                    for fase in range(self.dss.bus.num_nodes):
                        try:
                            vmag.append(self.dss.bus.vmag_angle_pu[fase * 2])
                        except IndexError:
                            print('')

                    cap_dados_rows.append({"cenario_id": self.cenario_id, "circuito": self.circuito, "nome": nome_cap,
                                           "patamar": number, "step": current_steps,
                                           "vmag_1": vmag[0], "vmag_2": vmag[1], "vmag_3": vmag[2],
                                           "available_steps": available_steps,})

            cap_rows.append({"cenario_id": self.cenario_id, "circuito": self.circuito, "nome": nome_cap, "kvar": kvar,
                         "ctrl_mode": ctrl_mode, "pt_ratio": pt_ratio, "ct_ratio": ct_ratio, "ctrl_on": ctrl_on,
                         "ctrl_off": ctrl_off, "delay": delay, "delay_off": delay_off, "dead_time": dead_time,
                         "kv_base": kv_base})

        if number == 0:
            self.__save_results_db("equipamento", cap_rows)

        self.__save_results_db("capacitor", cap_dados_rows)

    def solve_circuit(self):
        total_number = self.total_patamar
        ini_tentativa = 1  # valor inicial para o loadmult
        max_tentativa = 6  # numero de tentativas apos não covergência
        patamar_ini = self.patamar_ini
        patamar_fim = self.patamar_fim

        self.loadmult_ini = self.dss.solution.load_mult

        for number in range(patamar_ini, patamar_fim ):
            hour = self.dss.solution.hour
            sec = self.dss.solution.seconds

            self.dss.solution.solve()
            status = self.dss.solution.converged
            if status == 0:
                print(f'OpenDSS: File {self.dss_file} not solved to time {number}!')
                logging.info(f'OpenDSS: File {self.dss_file} NOT SOLVED. '
                             f'Set number: {number}, hour: {hour}, seconds: {sec}, event: {self.dss.solution.event_log}')

                # Instancia o gerador
                sequencia = gerador_positivo_negativo()
                # tentar novamente com loadmult
                for tentativa in range(ini_tentativa, max_tentativa + ini_tentativa):
                    #new_load_mult = self.loadmult_ini + tentativa / 100
                    new_load_mult = self.loadmult_ini + next(sequencia) / 100
                    self.dss.text(f"set loadmult={new_load_mult}")
                    self.dss.text(f"set time = ({hour}, {sec})")
                    print(f"Patamar:{number}, hour: {hour}, seconds: {sec}")

                    self.dss.solution.solve()
                    status = self.dss.solution.converged
                    if status == 0:
                        print(
                            f'OpenDSS: File {self.dss_file} alter loadMult {new_load_mult} and not solved to time {number}!')
                        logging.info(
                            f'OpenDSS: File {self.dss_file} NOT solved! - loadmult={new_load_mult} '
                            f'Set number: {number}, hour: {hour}, seconds: {sec}, event: {self.dss.solution.event_log}')
                    else:
                        print(
                            f'OpenDSS: File {self.dss_file} alter loadMult {new_load_mult} and solved to time {number}!')
                        logging.info(f'OpenDSS: File {self.dss_file} SOLVED alter loadMult {new_load_mult} '
                                     f'Set number: {number}, hour: {hour}, seconds: {sec}, event: {self.dss.solution.event_log}')

                        self.dss.text(f"set loadmult={self.loadmult_ini}")
                        self.__check_kv_base()
                        break

            print(f"Patamar:{number}, hour: {hour}, seconds: {sec}")
            if hour in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23) and sec == 0:
                path_dss = os.path.dirname(self.dss_file)
                file_exp = os.path.join(path_dss, f'{self.circuito}_{self.cenario.replace(" ", "_")}_EXP_Profile_time_{hour}.CSV')
                self.dss.text(f"Export Profile Phases=All {file_exp}")
                self.__csv2db(file_exp, hour)

            self._read_voltage(number)

            self._read_power(number)

            self.__read_cap(number)


        self.dss.text("Export Monitors all")


def main():
    # application_path = os.path.dirname(os.path.abspath(__file__))


    conf = load_config_cenario()
    cenarios = conf['cenario_id']
    cenario_desc = conf['cenario_desc']
    feeder_path = conf['feeder_path']
    feeder = conf['feeder']
    month = conf['month']
    type_day = conf['type_day']
    dist = conf['dist']
    sub = conf['sub']

    num_patamares = int(conf['num_patamares'])
    patamar_ini = int(conf['patamar_ini'])
    patamar_fim = int(conf['patamar_fim'])

    proc_time_ini = time.time()

    for index, circuito in enumerate(feeder):
        # if index ==1:
        #     break
        path_base = feeder_path[0]
        substation = sub[index]
        cenario_id = cenarios[index]
        cenario_descricao = cenario_desc[index]
        dss_file = os.path.join(path_base, substation, circuito,
                                fr'{type_day}_{month}_Master_{dist}_{substation}_{circuito}.dss')
        simul = SmartRT(cenario=cenario_descricao,
                        cenario_id = cenario_id,
                        circuito=circuito,
                        dist=dist,
                        sub=substation,
                        dss_file=dss_file,
                        num_patamatares=num_patamares,
                        patamar_ini=patamar_ini,
                        patamar_fim=patamar_fim,
                        )

        #simul.regcontrol_tsea_init()
        print(f'Solve: {circuito} - {cenario_desc}')
        simul.solve_circuit()

    print(f"Processo concluído em {time.time() - proc_time_ini}")


if __name__ == '__main__':
    main()
