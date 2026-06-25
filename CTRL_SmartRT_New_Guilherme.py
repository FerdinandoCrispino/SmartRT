import math

from regcontrol.regcontrol_TSEA import (LadoForteLadoFracoControl)
from setup_dinamico.setup_dinamico_TSEA import (setup_dinamico_TSEA_calcular)

import time
from py_dss_interface import DSS
from pathlib import Path
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cmath
import yaml
import multiprocessing
import sys
import ast
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple
from prodist_fase import Prodist
import logging

logging.basicConfig(filename='CTRL_SmartRT_new.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d,%H:%M:%S')


def convert2polar(real, imag):
    z = complex(real, imag)
    return cmath.polar(z)

def safe_divide(numerator, denominator):
    return numerator / denominator if denominator != 0 else 0


@dataclass
class Pesos:
    voltage_list_faseA: list
    voltage_list_faseB: list
    voltage_list_faseC: list
    tap_faseA: int
    tap_faseB: int
    tap_faseC: int
    reg_voltage_faseA: float
    reg_voltage_faseB: float
    reg_voltage_faseC: float
    vreg: float
    ptratio: float
    v_base: float
    v_reg_pu = float
    patamar: int = 0

    def __post_init__(self):
        self.v_reg_pu = (self.vreg * self.ptratio) / self.v_base


class SmartRT:
    def __init__(self, feeder, dss_file, bus_medicao_faseA, bus_medicao_faseB, bus_medicao_faseC, regcontrolname,
                 num_patamares, patamar_ini, patamar_fim, record_only_violations, usar_setup_dinamico):

        self.feeder = feeder
        self.dss_file = dss_file
        self.total_patamar = num_patamares
        self.patamar_ini = patamar_ini
        self.patamar_fim = patamar_fim
        self.bus_medicao_faseA = list(bus_medicao_faseA)
        self.bus_medicao_faseB = list(bus_medicao_faseB)
        self.bus_medicao_faseC = list(bus_medicao_faseC)

        self.record_only_violations = record_only_violations
        self.setup_dinamico = usar_setup_dinamico
        self.regControlName = regcontrolname
        self.reg_manual = []  # Lista de objetos - Inicia regcontrol_TSEA
        self.set_point = None # valor a ser atualizado pelo setup dimanico
        self.set_point_ideal = None  # valor definido nos parametros do regulador para ser seguido quando não ha violações

        # pre-computes to speed up lookups
        self.bus_medicao_keys_faseA = [item.split('.') for item in self.bus_medicao_faseA]
        self.bus_medicao_lookup_faseA = {f"{bus.lower()}.{node}" for bus, node in self.bus_medicao_keys_faseA}
        self.bus_medicao_faseA = {f"{bus.lower()}" for bus, node in self.bus_medicao_keys_faseA}
        self.bus_medicao_order_map_faseA = {f"{bus.lower()}.{node}": i for i, (bus, node) in enumerate(self.bus_medicao_keys_faseA)}

        self.bus_medicao_keys_faseB = [item.split('.') for item in self.bus_medicao_faseB]
        self.bus_medicao_lookup_faseB = {f"{bus.lower()}.{node}" for bus, node in self.bus_medicao_keys_faseB}
        self.bus_medicao_faseB = {f"{bus.lower()}" for bus, node in self.bus_medicao_keys_faseB}
        self.bus_medicao_order_map_faseB = {f"{bus.lower()}.{node}": i for i, (bus, node) in enumerate(self.bus_medicao_keys_faseB)}

        self.bus_medicao_keys_faseC = [item.split('.') for item in self.bus_medicao_faseC]
        self.bus_medicao_lookup_faseC = {f"{bus.lower()}.{node}" for bus, node in self.bus_medicao_keys_faseC}
        self.bus_medicao_faseC = {f"{bus.lower()}" for bus, node in self.bus_medicao_keys_faseC}
        self.bus_medicao_order_map_faseC = {f"{bus.lower()}.{node}": i for i, (bus, node) in enumerate(self.bus_medicao_keys_faseC)}

        self.bus_medicao = list(set().union(self.bus_medicao_faseA, self.bus_medicao_faseB, self.bus_medicao_faseC))

        # incremental output configuration
        self.result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados", self.feeder)
        # self.result_dir = os.path.join(f"E:/SmartRT/resultados/{self.feeder}") #TODO - SALVAR NO HD
        self.path_result_element = os.path.join(self.result_dir, f"voltage_element.csv")
        self.path_result_bus = os.path.join(self.result_dir, f"voltage_bus.csv")
        self.path_result_measurement = os.path.join(self.result_dir, f"voltage_measurement.csv")
        self.path_result_tap = os.path.join(self.result_dir, f"taps.csv")
        self.path_result_pesos = os.path.join(self.result_dir, f"pesos.csv")
        self._element_buffer = []
        self._bus_buffer = []
        self._measurement_buffer = []
        self._tap_buffer = []
        self._flush_interval = 100  # flush to disk every 1000 patamares
        self._pesos_buffer = []

        # ensure DSS is ready
        self.dss = self._read_dss_file()

        # função imprime o transformado da barra bt de medição para avaliar as suas fases correspondentes na MT
        # self._localiza_transformer()

        # Check kv_base
        self.__check_kv_base()

    def regcontrol_tsea_init(self):
        dss = self.dss
        vn = 7967  # Todo verificar necessidade de alterar para 13.8/sqrt(3)
        list_regcontrols = ast.literal_eval(self.regControlName)
        for reg_name in list_regcontrols:
            dss.regcontrols.name = reg_name
            tranformer = dss.regcontrols.transformer
            vreg = dss.regcontrols.forward_vreg
            revvreg = dss.regcontrols.reverse_vreg
            band = dss.regcontrols.forward_band
            revband = dss.regcontrols.reverse_band
            pt_ratio = dss.regcontrols.pt_ratio
            delay = dss.regcontrols.delay
            tap_delay = dss.regcontrols.tap_delay
            v_base = round(vn / pt_ratio, 2)
            self.set_point = (vreg * pt_ratio) / vn  # valor inicial do vreg_pu para o LadoForteLadoFraco
            self.set_point_ideal = (vreg * pt_ratio) / vn
            # Desabilita os RegControl do Master
            dss.text(f"Edit RegControl.{reg_name} enabled=no")

            reg_manual = LadoForteLadoFracoControl(dss, tranformer, vreg, band,
                                                   pt_ratio, revvreg, revband, delay,
                                                   tap_delay, v_base, ativar_depuracao=True)

            self.reg_manual.append(reg_manual)

    def _localiza_transformer(self):
        dss = self.dss
        dss.transformers.first()
        pontos_med_keys = [item.split('.')[0] for item in self.bus_medicao_faseA]
        pontos_med = [bus.lower() for bus in pontos_med_keys]

        for _ in range(dss.transformers.count):
            if 'reg' in dss.transformers.name:
                dss.transformers.next()
                continue
            dss.circuit.set_active_element(f"transformer.{dss.transformers.name}")
            bus_name = dss.cktelement.bus_names
            element_name = dss.cktelement.name
            dss.circuit.set_active_bus(bus_name[1])
            bus_name1 = dss.bus.name
            if bus_name1 in pontos_med:
                print('trasformador localizado')
                dss.circuit.set_active_element(element_name)
                print(f'bus:{dss.bus.name}, Nodes:{dss.bus.nodes}')
                print(f'{element_name}:{dss.cktelement.node_order}')
                print(f'-' * 50)

            dss.topology.first()
            while True:
                indx = dss.topology.active_branch
                indx_level = dss.topology.active_level
                branch_name = dss.topology.branch_name
                if branch_name == element_name:
                    dss.circuit.set_active_element(element_name)
                    dss.circuit.set_active_bus(dss.cktelement.bus_names[1])
                    # encontrou o transformador na topologia
                    break
                index_branch = dss.topology.forward_branch()

            # busca os ramais conectados neste transformador
            while True:
                index_branch_2 = dss.topology.next()
                indx_level_2 = dss.topology.active_level
                branch_name_2 = dss.topology.branch_name
                if not dss.topology.branch_name.startswith(('Line.sbt', 'Line.rbt')):
                    # print('Proximo transformador!')
                    break

                dss.circuit.set_active_element(branch_name_2)
                dss.circuit.set_active_bus(dss.cktelement.bus_names[1])
                if dss.bus.name in pontos_med:
                    print('trasformador localizado')
                    print(f'Linha:{dss.cktelement.name}, bus:{dss.bus.name}, Nodes:{dss.bus.nodes}')
                    print(f'{element_name}:{dss.cktelement.node_order}')
                    print(f'-' * 50)
            dss.transformers.next()
        print('....')

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
                if not (line_dss.startswith('!') or line_dss.startswith('\n') or line_dss.lower().startswith('clear')):
                    dss.text(line_dss.strip('\n'))
                if 'calc' in line_dss:
                    break

        dss.text("set mode = daily")
        dss.text("set controlmode = time")  # Todo avaliar resultado para Static
        dss.text("set tolerance = 0.0001")
        dss.text("set maxcontroliter = 100")
        dss.text("set maxiterations = 100")

        if self.total_patamar == 144:
            dss.text(f"set stepsize = 10m")
        else:
            dss.text(f"set stepsize = {86400 / self.total_patamar}s")

        dss.text("set number = 1")

        segundos_totais = int(self.patamar_ini * 86400 / self.total_patamar)
        minutos, segundos = divmod(segundos_totais, 60)
        horas, minutos = divmod(minutos, 60)
        total_sec = minutos * 60 + segundos

        dss.text(f"set time = ({horas}, {total_sec})")

        return dss

    def __check_kv_base(self):
        """
        Verifica a tensão de base definida pelo openDSS para as todas as barras conectadas
        no secundario dos transformadores.
        São obtidas as tensões de fase para a barra do secundario do TR e comparada com a informada pelo openDSS
        Em caso de diferença são localizadas todas barras conectadas no secundario do transformador e set o kv_base
        de todas as barras com o valor obtido da avaliação das conecções do transformador.
        :return:
        """
        dss = self.dss
        n = 0
        count_tr = 0
        dss.transformers.first()
        for _ in range(dss.transformers.count):
            count_tr += 1
            transformer_name = dss.transformers.name
            if transformer_name.lower().startswith("reg"):
                dss.transformers.next()
                continue

            dss.circuit.set_active_element(f"transformer.{transformer_name}")
            tr_ph = dss.cktelement.num_phases
            if tr_ph == 3:
                vll = dss.transformers.kv
                vln = vll / math.sqrt(3)
            elif tr_ph == 1:
                num_wdg = dss.transformers.num_windings
                if num_wdg == 2:
                    if dss.transformers.is_delta:
                        vll = dss.transformers.kv
                        vln = vll / 2
                    else:
                        vln = dss.transformers.kv
                        vll = vln * 2
                elif num_wdg == 3:
                    dss.transformers.wdg = 2
                    vln = dss.transformers.kv
                    vll = 2 * vln

            dss.circuit.set_active_bus(dss.cktelement.bus_names[1])
            bus_transformer_name = dss.bus.name
            kv_base = dss.bus.kv_base
            if round(vln, 3) != round(kv_base, 3):
                if n == 0:
                    n += 1
                    print(f'VERIFICAÇÃO DAS TENSÕES DE BASE - {self.feeder}')

                dss.text(f'SetkVBase Bus={bus_transformer_name} kVLN={vln}')
                dss.topology.first()
                while True:
                    indx = dss.topology.active_branch
                    indx_level = dss.topology.active_level
                    branch_name = dss.topology.branch_name
                    if branch_name == f"Transformer.{transformer_name}":
                        dss.circuit.set_active_element(f"transformer.{transformer_name}")
                        dss.circuit.set_active_bus(bus_transformer_name)
                        break
                    index_branch = dss.topology.forward_branch()

                while True:
                    index_branch_2 = dss.topology.next()
                    indx_level_2 = dss.topology.active_level
                    branch_name_2 = dss.topology.branch_name
                    if not dss.topology.branch_name.lower().startswith(('line.sbt', 'line.rbt')):
                        break
                    dss.circuit.set_active_element(branch_name_2)
                    dss.circuit.set_active_bus(dss.cktelement.bus_names[1])
                    bus_line_name = dss.bus.name
                    kv_base_2 = dss.bus.kv_base
                    dss.text(f'SetkVBase Bus={bus_line_name} kVLN={vln}')

            dss.transformers.next()

    def _set_pesos(self, patamar_rows):
        # patamar_rows: lista de dicts contendo as tensões do patamar atual
        if isinstance(patamar_rows, pd.DataFrame):
            df_patamar_voltage = patamar_rows.copy()
        else:
            df_patamar_voltage = pd.DataFrame(patamar_rows)

        if df_patamar_voltage.empty:
            print("_set_pesos: patamar vazio.")
            return None

        # normaliza colunas
        df_patamar_voltage.loc[:, 'bus'] = df_patamar_voltage['bus'].astype(str).str.lower()
        df_patamar_voltage.loc[:, '_bus_node'] = df_patamar_voltage['bus'] + '.' + df_patamar_voltage['nodes'].astype(str)

        # tensões nos barramentos selecionados
        df_bus_medicao_faseA = df_patamar_voltage[df_patamar_voltage['_bus_node'].isin(self.bus_medicao_lookup_faseA)].copy()

        df_bus_medicao_faseB = df_patamar_voltage[df_patamar_voltage['_bus_node'].isin(self.bus_medicao_lookup_faseB)].copy()

        df_bus_medicao_faseC = df_patamar_voltage[df_patamar_voltage['_bus_node'].isin(self.bus_medicao_lookup_faseC)].copy()

        if df_bus_medicao_faseA.shape[0] < len(self.bus_medicao_faseA):
            print(f"Barra não encontrada na fase A! Verificar a lista de barras fornecida.")
            exit()
        if df_bus_medicao_faseB.shape[0] < len(self.bus_medicao_faseB):
            print(f"Barra não encontrada na fase B! Verificar a lista de barras fornecida.")
            exit()
        if df_bus_medicao_faseC.shape[0] < len(self.bus_medicao_faseC):
            print(f"Barra não encontrada na fase C! Verificar a lista de barras fornecida.")
            exit()

        volt_bus_reg = []
        tap_reg = []
        fvreg = 0  # igual para todas as fases
        pt_ratio_reg = 0.0
        v_base = 0
        for index, reg_name in enumerate(self.regControlName):
            self.dss.regcontrols.name = reg_name
            if self.dss.regcontrols.name == reg_name.lower():
                if self.setup_dinamico:
                    # tap_reg.append(self.dss.regcontrols.tap_number)
                    tap_reg.append(self.reg_manual[index].reg_manual.tap_position)
                    # fvreg = self.dss.regcontrols.fv_reg
                    fvreg = self.reg_manual[index].reg_manual.vreg
                    pt_ratio_reg = self.reg_manual[index].ptratio
                    self.dss.transformers.name = self.reg_manual[index].transformer
                    bus_reg_trafo = self.dss.cktelement.bus_names[1].split('.')[0]
                    node_reg_trafo = self.dss.cktelement.bus_names[1].split('.')[1]
                    v_base = self.dss.bus.kv_base * 1000
                else:
                    self.dss.regcontrols.name = reg_name
                    tap_reg.append(self.dss.regcontrols.tap_number)
                    winding = self.dss.regcontrols.winding
                    rreg = self.dss.regcontrols.reverse_vreg
                    fvreg = self.dss.regcontrols.forward_vreg
                    pt_ratio_reg = self.dss.regcontrols.pt_ratio
                    self.dss.transformers.name = self.dss.regcontrols.transformer
                    bus_reg_trafo = self.dss.cktelement.bus_names[1].split('.')[0]
                    node_reg_trafo = self.dss.cktelement.bus_names[1].split('.')[1]
                    self.dss.circuit.set_active_bus(bus_reg_trafo)
                    v_base = self.dss.bus.kv_base * 1000

                # tensão no regulador selecionado
                volt_bus_reg.append(df_patamar_voltage.loc[(df_patamar_voltage['bus'] == bus_reg_trafo.lower()) &
                                                           (df_patamar_voltage['nodes'] == node_reg_trafo)])

        # garantir a ordem das barras igual a lista de entrada das barras de medicao
        df_bus_medicao_faseA.loc[:, 'bus_sort'] = df_bus_medicao_faseA['_bus_node'].map(
            self.bus_medicao_order_map_faseA)
        df_bus_medicao_faseA = df_bus_medicao_faseA.sort_values('bus_sort').drop(columns=['bus_sort', '_bus_node'])

        df_bus_medicao_faseB.loc[:, 'bus_sort'] = df_bus_medicao_faseB['_bus_node'].map(
            self.bus_medicao_order_map_faseB)
        df_bus_medicao_faseB = df_bus_medicao_faseB.sort_values('bus_sort').drop(columns=['bus_sort', '_bus_node'])

        df_bus_medicao_faseC.loc[:, 'bus_sort'] = df_bus_medicao_faseC['_bus_node'].map(
            self.bus_medicao_order_map_faseC)
        df_bus_medicao_faseC = df_bus_medicao_faseC.sort_values('bus_sort').drop(columns=['bus_sort', '_bus_node'])

        # tenta extrair patamar do dataframe
        try:
            pat_val = int(df_patamar_voltage['patamar'].iat[0])
        except Exception:
            pat_val = 0

        pesos = Pesos(voltage_list_faseA=df_bus_medicao_faseA['vln_pu'].tolist(),
                      voltage_list_faseB=df_bus_medicao_faseB['vln_pu'].tolist(),
                      voltage_list_faseC=df_bus_medicao_faseC['vln_pu'].tolist(),
                      tap_faseA=tap_reg[0], tap_faseB=tap_reg[1], tap_faseC=tap_reg[2],
                      patamar=pat_val,
                      reg_voltage_faseA=volt_bus_reg[0]['vln_pu'].values[0],
                      reg_voltage_faseB=volt_bus_reg[1]['vln_pu'].values[0],
                      reg_voltage_faseC=volt_bus_reg[2]['vln_pu'].values[0],
                      vreg=fvreg,
                      ptratio=pt_ratio_reg, v_base=v_base)

        # print('Determinacao dos pesos ok. ')
        return pesos

    def _set_pesos_only_violations(self, voltage_measurement_rows, voltage_reg_rows):
        # patamar_rows: lista de dicts contendo as tensões do patamar atual
        if isinstance(voltage_measurement_rows, pd.DataFrame):
            df_patamar_measurement = voltage_measurement_rows.copy()
        else:
            df_patamar_measurement = pd.DataFrame(voltage_measurement_rows)

        if df_patamar_measurement.empty:
            print("_set_pesos_only_violations: patamar vazio.")
            return None

        if isinstance(voltage_reg_rows, pd.DataFrame):
            df_patamar_voltage_reg = voltage_reg_rows.copy()
        else:
            df_patamar_voltage_reg = pd.DataFrame(voltage_reg_rows)

        if df_patamar_voltage_reg.empty:
            print("_set_pesos_only_violations: patamar vazio.")
            return None

        # normaliza colunas
        df_patamar_measurement.loc[:, '_bus_node'] = df_patamar_measurement['Bus'].astype(str).str.lower() + '.' + df_patamar_measurement['Node'].astype(str)
        df_patamar_voltage_reg.loc[:, '_bus_node'] = df_patamar_voltage_reg['Bus'].astype(str).str.lower() + '.' + df_patamar_voltage_reg['Node'].astype(str)

        df_patamar_voltage = (pd.concat([df_patamar_measurement, df_patamar_voltage_reg], ignore_index=True).drop_duplicates(subset=['Bus', 'Node']))

        # tensões nos barramentos selecionados
        df_bus_medicao_faseA = df_patamar_voltage[df_patamar_voltage['_bus_node'].isin(self.bus_medicao_lookup_faseA)].copy()
        df_bus_medicao_faseB = df_patamar_voltage[df_patamar_voltage['_bus_node'].isin(self.bus_medicao_lookup_faseB)].copy()
        df_bus_medicao_faseC = df_patamar_voltage[df_patamar_voltage['_bus_node'].isin(self.bus_medicao_lookup_faseC)].copy()

        if df_bus_medicao_faseA.shape[0] < len(self.bus_medicao_faseA):
            print(f"Barra não encontrada na fase A! Verificar a lista de barras fornecida.")
            exit()
        if df_bus_medicao_faseB.shape[0] < len(self.bus_medicao_faseB):
            print(f"Barra não encontrada na fase B! Verificar a lista de barras fornecida.")
            exit()
        if df_bus_medicao_faseC.shape[0] < len(self.bus_medicao_faseC):
            print(f"Barra não encontrada na fase C! Verificar a lista de barras fornecida.")
            exit()

        volt_bus_reg = []
        tap_reg = []
        fvreg = 0  # igual para todas as fases
        pt_ratio_reg = 0.0
        v_base = 0
        list_regcontrols = ast.literal_eval(self.regControlName)
        for index, reg_name in enumerate(list_regcontrols):
            self.dss.regcontrols.name = reg_name
            if self.dss.regcontrols.name == reg_name.lower():
                tap_reg.append(self.reg_manual[index].reg_manual.tap_position)
                fvreg = self.reg_manual[index].reg_manual.vreg
                pt_ratio_reg = self.reg_manual[index].ptratio
                self.dss.transformers.name = self.reg_manual[index].transformer
                bus_node_reg_trafo = self.dss.cktelement.bus_names[1].rsplit('.', 1)[0]
                v_base = self.dss.bus.kv_base * 1000

                # tensão no regulador selecionado
                volt_bus_reg.append(df_patamar_voltage.loc[df_patamar_voltage['_bus_node'] == bus_node_reg_trafo.lower()])

        # garantir a ordem das barras igual a lista de entrada das barras de medicao
        df_bus_medicao_faseA.loc[:, 'bus_sort'] = df_bus_medicao_faseA['_bus_node'].map(self.bus_medicao_order_map_faseA)
        df_bus_medicao_faseA = df_bus_medicao_faseA.sort_values('bus_sort').drop(columns=['bus_sort', '_bus_node'])

        df_bus_medicao_faseB.loc[:, 'bus_sort'] = df_bus_medicao_faseB['_bus_node'].map(self.bus_medicao_order_map_faseB)
        df_bus_medicao_faseB = df_bus_medicao_faseB.sort_values('bus_sort').drop(columns=['bus_sort', '_bus_node'])

        df_bus_medicao_faseC.loc[:, 'bus_sort'] = df_bus_medicao_faseC['_bus_node'].map(self.bus_medicao_order_map_faseC)
        df_bus_medicao_faseC = df_bus_medicao_faseC.sort_values('bus_sort').drop(columns=['bus_sort', '_bus_node'])

        # tenta extrair patamar do dataframe
        try:
            pat_val = int(df_patamar_voltage['Number'].iat[0])
        except Exception:
            pat_val = 0

        pesos = Pesos(voltage_list_faseA=df_bus_medicao_faseA['Voltage'].tolist(),
                      voltage_list_faseB=df_bus_medicao_faseB['Voltage'].tolist(),
                      voltage_list_faseC=df_bus_medicao_faseC['Voltage'].tolist(),
                      tap_faseA=tap_reg[0], tap_faseB=tap_reg[1], tap_faseC=tap_reg[2],
                      patamar=pat_val,
                      reg_voltage_faseA=volt_bus_reg[0]['Voltage'].values[0],
                      reg_voltage_faseB=volt_bus_reg[1]['Voltage'].values[0],
                      reg_voltage_faseC=volt_bus_reg[2]['Voltage'].values[0],
                      vreg=fvreg,
                      ptratio=pt_ratio_reg,
                      v_base=v_base)

        # print('Determinacao dos pesos ok. ')
        return pesos

    def _flush_element_buffer(self):
        # escreve buffer acumulado em disco e limpa o buffer
        if not self._element_buffer:
            return
        os.makedirs(self.result_dir, exist_ok=True)
        write_header = not os.path.exists(self.path_result_element)
        df_chunk = pd.DataFrame(self._element_buffer)
        df_chunk.to_csv(self.path_result_element, mode='a', header=write_header, index=False)
        self._element_buffer.clear()

    def _flush_bus_buffer(self):
        # escreve buffer acumulado em disco e limpa o buffer
        if not self._bus_buffer:
            return
        os.makedirs(self.result_dir, exist_ok=True)
        write_header = not os.path.exists(self.path_result_bus)
        df_chunk = pd.DataFrame(self._bus_buffer)
        df_chunk.to_csv(self.path_result_bus, mode='a', header=write_header, index=False)
        self._bus_buffer.clear()

    def _flush_measurement_buffer(self):
        # escreve buffer acumulado em disco e limpa o buffer
        if not self._measurement_buffer:
            return
        os.makedirs(self.result_dir, exist_ok=True)
        write_header = not os.path.exists(self.path_result_measurement)
        df_chunk = pd.DataFrame(self._measurement_buffer)
        df_chunk.to_csv(self.path_result_measurement, mode='a', header=write_header, index=False)
        self._measurement_buffer.clear()

    def _flush_tap_buffer(self):
        # escreve buffer acumulado em disco e limpa o buffer
        if not self._tap_buffer:
            return
        os.makedirs(self.result_dir, exist_ok=True)
        write_header = not os.path.exists(self.path_result_tap)
        df_chunk = pd.DataFrame(self._tap_buffer)
        df_chunk.to_csv(self.path_result_tap, mode='a', header=write_header, index=False)
        self._tap_buffer.clear()

    def _flush_pesos_buffer(self):
        # escreve buffer de pesos acumulado em disco e limpa o buffer
        if not self._pesos_buffer:
            return
        os.makedirs(self.result_dir, exist_ok=True)
        write_header = not os.path.exists(self.path_result_pesos)
        df_chunk = pd.DataFrame([asdict(p) if hasattr(p, '__dataclass_fields__') else p for p in self._pesos_buffer])
        df_chunk.to_csv(self.path_result_pesos, mode='a', header=write_header, index=False)
        self._pesos_buffer.clear()

    def _plot_profile(self, hour):
        dss = self.dss

        va = dss.circuit.nodes_vmag_pu_by_phase(1)
        vb = dss.circuit.nodes_vmag_pu_by_phase(2)
        vc = dss.circuit.nodes_vmag_pu_by_phase(3)
        da = dss.circuit.nodes_distances_by_phase(1)
        db = dss.circuit.nodes_distances_by_phase(2)
        dc = dss.circuit.nodes_distances_by_phase(3)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.scatter(da, va, color="black", label='Fase A', s=0.8)
        ax.scatter(db, vb, color="red", label='Fase B', s=0.8)
        ax.scatter(dc, vc, color="blue", label='Fase C', s=0.8)

        overvoltage = 1.05
        undervoltage_cri = 0.93
        undervoltage_pre = 0.9
        overvoltage_bt_pre = 1.047
        undervoltage_bt_pre = 0.921
        overvoltage_bt_cri = 1.063
        undervoltage_bt_cri = 0.866

        dist_max = max(max(da), max(db), max(dc))
        ax.plot([0, dist_max], [overvoltage, overvoltage], 'r--', linewidth=1)
        ax.plot([0, dist_max], [undervoltage_cri, undervoltage_cri], 'r--', linewidth=1)
        ax.plot([0, dist_max], [undervoltage_pre, undervoltage_pre], 'r--', linewidth=1)
        ax.plot([0, dist_max], [overvoltage_bt_pre, overvoltage_bt_pre], color='limegreen', linestyle='--', linewidth=1)
        ax.plot([0, dist_max], [undervoltage_bt_pre, undervoltage_bt_pre], color='limegreen', linestyle='--', linewidth=1)
        ax.plot([0, dist_max], [overvoltage_bt_cri, overvoltage_bt_cri], color='limegreen', linestyle='--', linewidth=1)
        ax.plot([0, dist_max], [undervoltage_bt_cri, undervoltage_bt_cri], color='limegreen', linestyle='--', linewidth=1)

        ax.set_title(f"Perfil de Tensão - {hour}h")
        ax.set_ylabel("Tensão (pu)")
        ax.set_xlabel("Distância (km)")
        ax.set_ylim(0.8, 1.2)
        ax.legend()
        ax.grid(True)

        plt.margins(x=0)
        os.makedirs(os.path.dirname(self.result_dir), exist_ok=True)
        voltage_profile_path = os.path.join(os.path.dirname(self.result_dir), f"{self.feeder}",f"Perfil_Tensao_{hour}h.png")
        plt.savefig(voltage_profile_path, dpi=300, bbox_inches="tight")
        plt.close()

    def __extract_nodes(self, bus):
        """
        Extrai as fases presentes na barra selecionada.
        """
        nodes = self.dss.bus.nodes
        nos = [no for no in nodes if int(no) not in (0, 4)]
        phases = len(nos)

        if phases == 1:
            node = [nos[0]]
        elif phases == 2:
            node = nos[:2]
        elif phases == 3:
            node = nos[:3]
        else:
            return None

        return ".".join(str(n) for n in node)

    def __filling_phases(self, v_base, voltages, node, phases):
        data_voltages = list()

        if phases == 1:
            if node == '1':
                V_a = voltages[0]
                V_b = ""
                V_c = ""
            elif node == '2':
                V_a = ""
                V_b = voltages[0]
                V_c = ""
            elif node == '3':
                V_a = ""
                V_b = ""
                V_c = voltages[0]

        elif phases == 2:
            if node == '1.2':
                V_a = voltages[0]
                V_b = voltages[1]
                V_c = ""
            elif node == '1.3':
                V_a = voltages[0]
                V_b = ""
                V_c = voltages[1]
            elif node == '2.3':
                V_a = ""
                V_b = voltages[0]
                V_c = voltages[1]

        elif phases == 3:
            V_a = voltages[0]
            V_b = voltages[1]
            V_c = voltages[2]

        volt = {
            1: V_a,
            2: V_b,
            3: V_c
        }

        for node, voltage_mag in volt.items():
            if voltage_mag == "":
                continue

            data_voltages.append({
                "Node":node,
                "V_mag":voltage_mag,
                "V_base":v_base
            })

        return data_voltages

    def __filling_load_violation(self, number, element, bus_load, data_voltages_completed, element_rows):
        for item in data_voltages_completed:
            element_rows.append({
                "Number": number,
                "Element": element,
                "Bus": bus_load,
                "Node": item['Node'],
                "V_mag": item['V_mag'],
                "Occurrence": item['Level']
            })

        return element_rows

    def __filling_bus_violation(self, number, bus_name, data_voltages_completed, voltage_bus_rows):
        for item in data_voltages_completed:
            voltage_bus_rows.append({
                "Number": number,
                "Bus": bus_name,
                "Node": item['Node'],
                "Voltage": item['V_pu'],
                "Occurrence": item['Level']
            })

        return voltage_bus_rows

    def solve_circuit(self):
        ini_tentativa = 1  # valor inicial para o loadmult
        max_tentativa = 10  # número de tentativas após não covergência
        patamar_ini = self.patamar_ini
        patamar_fim = self.patamar_fim

        # start with a fresh output file for incremental writes
        if os.path.exists(self.path_result_element):
            try:
                os.remove(self.path_result_element)
            except OSError:
                pass
        if os.path.exists(self.path_result_bus):
            try:
                os.remove(self.path_result_bus)
            except OSError:
                pass
        if os.path.exists(self.path_result_measurement):
            try:
                os.remove(self.path_result_measurement)
            except OSError:
                pass
        if os.path.exists(self.path_result_tap):
            try:
                os.remove(self.path_result_tap)
            except OSError:
                pass
        if os.path.exists(self.path_result_pesos):
            try:
                os.remove(self.path_result_pesos)
            except OSError:
                pass

        for number in range(patamar_ini, patamar_fim + 1):
            hour = self.dss.solution.hour
            sec = self.dss.solution.seconds
            print(f"{self.feeder}; Patamar:{number}, Hour: {hour}, Seconds: {sec}")

            self.loadmult_ini = self.dss.solution.load_mult
            self.dss.solution.solve()
            status = self.dss.solution.converged
            if status == 0:
                for tentativa in range(ini_tentativa, max_tentativa + ini_tentativa):
                    new_load_mult = self.loadmult_ini + tentativa / 100
                    self.dss.text(f"set loadmult={new_load_mult}")
                    self.dss.text(f"set time = ({hour}, {sec})")

                    self.dss.solution.solve()
                    status = self.dss.solution.converged
                    if status == 0:
                        continue

                    elif status == 0 and tentativa == max_tentativa:
                        print(f'❌ OpenDSS: File {self.dss_file} changed loadMult {new_load_mult} and NOT SOLVED - Patamar:{number}, Hour: {hour}, Seconds: {sec}')
                        logging.info(
                            f'OpenDSS: File {self.dss_file} NOT SOLVED! - loadmult={new_load_mult} '
                            f'Set number: {number}, hour: {hour}, seconds: {sec}, event: {self.dss.solution.event_log}')

                    else:
                        print(f'⚠️ OpenDSS: Feeder {self.feeder} changed loadMult {new_load_mult} and SOLVED - Patamar:{number}, Hour: {hour}, Seconds: {sec}')
                        self.dss.text(f"set loadmult={self.loadmult_ini}")
                        self.__check_kv_base()
                        if hour in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24) and sec == 0:
                            self._plot_profile(hour)

                        break

            if hour in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24) and sec == 0:
                self._plot_profile(hour)

            # controle para inserir ou remover o setup dinamico da simulação
            if self.setup_dinamico:
                tap_atual = [0, 0, 0]
                lado_forte_fonte = [None, None, None]

                list_regcontrols = ast.literal_eval(self.regControlName)
                for index, value in enumerate(list_regcontrols):
                    tap_atual[index], lado_forte_fonte[index] = self.reg_manual[index].ladoForte_ladoFraco_executar(self.set_point)

            if self.record_only_violations:
                element_rows = list()
                voltage_bus_rows = list()
                voltage_measurement_rows = list()
                tap_reg = list()
                voltage_reg_rows = list()

                for element in self.dss.circuit.elements_names:
                    if element.lower().startswith("load.pip"):
                        continue

                    if element.lower().startswith("load."):
                        self.dss.circuit.set_active_element(element)
                        bus_load = self.dss.cktelement.bus_names[0].split(".")[0]
                        self.dss.circuit.set_active_bus(bus_load)
                        v_base = self.dss.bus.kv_base * 1000
                        voltages = self.dss.bus.vmag_angle[::2]
                        nodes = self.__extract_nodes(bus_load)
                        phases = len(nodes.split("."))
                        data_voltages = self.__filling_phases(v_base, voltages, nodes, phases)
                        data_voltages_completed = Prodist.faixa_tensao(data_voltages)

                        for item in data_voltages_completed:
                            level = item['Level']
                            if level != 'adequada':
                                element_rows = self.__filling_load_violation(number, element, bus_load, data_voltages_completed, element_rows)
                                break

                for bus_name in self.dss.circuit.buses_names:
                    self.dss.circuit.set_active_bus(bus_name)
                    v_base = self.dss.bus.kv_base * 1000
                    voltages = self.dss.bus.vmag_angle[::2]
                    nodes = self.__extract_nodes(bus_name)
                    phases = len(nodes.split("."))
                    data_voltages = self.__filling_phases(v_base, voltages, nodes, phases)
                    data_voltages_completed = Prodist.faixa_tensao(data_voltages)

                    if bus_name in self.bus_medicao:
                        voltage_measurement_rows = self.__filling_bus_violation(number, bus_name, data_voltages_completed, voltage_measurement_rows)

                    for item in data_voltages_completed:
                        level = item['Level']
                        if level != 'adequada':
                            voltage_bus_rows = self.__filling_bus_violation(number, bus_name, data_voltages_completed, voltage_bus_rows)
                            break

                list_regcontrols = ast.literal_eval(self.regControlName)
                for reg_name in list_regcontrols:
                    if self.setup_dinamico == False:
                        self.dss.regcontrols.name = reg_name
                        v_reg = self.dss.regcontrols.forward_vreg
                        self.dss.transformers.name = self.dss.regcontrols.transformer
                        bus_reg_trafo = self.dss.cktelement.bus_names[1].split('.')[0]
                        self.dss.circuit.set_active_bus(bus_reg_trafo)
                        v_reg_pu = self.dss.bus.vmag_angle_pu

                        phase = reg_name[-1].upper()
                        idx_fase = {
                            "A": 0,
                            "B": 2,
                            "C": 4
                        }

                        v_phase = v_reg_pu[idx_fase[phase]]
                        tap_reg.append({
                            "Number": number,
                            "Fase": phase,
                            "V_ref": v_reg,
                            "V_reg": v_phase,
                            "Tap": self.dss.regcontrols.tap_number
                        })

                    else:
                        self.dss.regcontrols.name = reg_name
                        transformer_reg = self.dss.regcontrols.transformer
                        element = f"transformer.{transformer_reg}"
                        self.dss.circuit.set_active_element(element)
                        bus_reg_trafo = self.dss.cktelement.bus_names[1].split('.')[0]
                        self.dss.circuit.set_active_bus(bus_reg_trafo)
                        v_base = self.dss.bus.kv_base * 1000
                        voltages = self.dss.bus.vmag_angle[::2]
                        nodes = self.__extract_nodes(bus_reg_trafo)
                        phases = len(nodes.split("."))
                        data_voltages = self.__filling_phases(v_base, voltages, nodes, phases)
                        data_voltages_completed = Prodist.faixa_tensao(data_voltages)
                        voltage_reg_rows = self.__filling_bus_violation(number, bus_reg_trafo, data_voltages_completed, voltage_reg_rows)
                        break

                # append to buffer and flush in blocks
                self._element_buffer.extend(element_rows)
                self._bus_buffer.extend(voltage_bus_rows)
                self._measurement_buffer.extend(voltage_measurement_rows)
                self._tap_buffer.extend(tap_reg)
                if number % self._flush_interval == 0:
                    self._flush_element_buffer()
                    self._flush_bus_buffer()
                    self._flush_measurement_buffer()
                    self._flush_tap_buffer()

                if self.setup_dinamico:
                    set_pesos = self._set_pesos_only_violations(voltage_measurement_rows, voltage_reg_rows)

                    if set_pesos is not None:
                        self._pesos_buffer.append(asdict(set_pesos))
                        if len(self._pesos_buffer) >= self._flush_interval:
                            self._flush_pesos_buffer()
                        # print(set_pesos)

                    setpoint_atual = self.set_point
                    tensao_bucha_faseA = set_pesos.reg_voltage_faseA
                    tensao_bucha_faseB = set_pesos.reg_voltage_faseB
                    tensao_bucha_faseC = set_pesos.reg_voltage_faseC
                    tensoes_faseA = set_pesos.voltage_list_faseA
                    tensoes_faseB = set_pesos.voltage_list_faseB
                    tensoes_faseC = set_pesos.voltage_list_faseC

                    self.set_point = setup_dinamico_TSEA_calcular(
                        tensao_bucha_faseA=tensao_bucha_faseA,
                        tensoes_pontos_faseA=tensoes_faseA,
                        tap_atual_faseA=tap_atual[0],
                        setpoint_atual_faseA=setpoint_atual,
                        lado_forte_fonte_faseA=lado_forte_fonte[0],
                        tensao_bucha_faseB=tensao_bucha_faseB,
                        tensoes_pontos_faseB=tensoes_faseB,
                        tap_atual_faseB=tap_atual[1],
                        setpoint_atual_faseB=setpoint_atual,
                        lado_forte_fonte_faseB=lado_forte_fonte[1],
                        tensao_bucha_faseC=tensao_bucha_faseC,
                        tensoes_pontos_faseC=tensoes_faseC,
                        tap_atual_faseC=tap_atual[2],
                        lado_forte_fonte_faseC=lado_forte_fonte[2],
                        setpoint_atual_faseC=setpoint_atual,
                        setpoint_ideal=self.set_point_ideal
                    )

            else:
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
                        vll_pu_1 = round(convert2polar(self.dss.bus.pu_vll[pos * 2], self.dss.bus.pu_vll[(pos * 2) + 1])[0], 5)
                        # vll_pu_1 = np.float32(vll_pu_1)

                    vln_1 = round(convert2polar(self.dss.bus.voltages[pos * 2], self.dss.bus.voltages[(pos * 2) + 1])[0], 5)
                    # vln_1 = np.float32(vln_1)
                    vln_pu_1 = round(convert2polar(self.dss.bus.pu_voltages[pos * 2], self.dss.bus.pu_voltages[(pos * 2) + 1])[0], 5)
                    # vln_pu_1 = np.float32(vln_pu_1)

                    # para transformadores fase-fase não existe tensão de fase, usar o valor da tensão de linha em pu
                    if math.isnan(vln_pu_1) or vln_pu_1 == 0:
                        vln_pu_1 = vll_pu_1

                    current_voltage_rows.append({
                        "Patamar": number,
                        "Bus": f"{bus_name.split('.')[0]}".lower(),
                        "Node": bus_node,
                        # "vll": vll_1,
                        "vln": vln_1,
                        # "vll_pu": vll_pu_1,
                        "vln_pu": vln_pu_1,
                        "kv_base": int(self.dss.bus.kv_base * 1000)
                        # necessario para verificar o nivel de tensão para analise de barras
                    })

                # append to buffer and flush in blocks
                self._bus_buffer.extend(current_voltage_rows)
                if number % self._flush_interval == 0:
                    self._flush_bus_buffer()

                # Determina e armazena pesos para ESTE patamar (histórico completo)
                set_pesos = self._set_pesos(current_voltage_rows)

                if set_pesos is not None:
                    # buffer e gravação incremental de pesos (não manter em memória)
                    self._pesos_buffer.append(asdict(set_pesos))
                    if len(self._pesos_buffer) >= self._flush_interval:
                        self._flush_pesos_buffer()
                    print(set_pesos)

                setpoint_atual = self.set_point
                tensao_bucha_faseA = set_pesos.reg_voltage_faseA
                tensao_bucha_faseB = set_pesos.reg_voltage_faseB
                tensao_bucha_faseC = set_pesos.reg_voltage_faseC
                tensoes_faseA = set_pesos.voltage_list_faseA
                tensoes_faseB = set_pesos.voltage_list_faseB
                tensoes_faseC = set_pesos.voltage_list_faseC

                if self.setup_dinamico:
                    self.set_point = setup_dinamico_TSEA_calcular(
                        tensao_bucha_faseA=tensao_bucha_faseA,
                        tensoes_pontos_faseA=tensoes_faseA,
                        tap_atual_faseA=tap_atual[0],
                        setpoint_atual_faseA=setpoint_atual,
                        lado_forte_fonte_faseA=lado_forte_fonte[0],
                        tensao_bucha_faseB=tensao_bucha_faseB,
                        tensoes_pontos_faseB=tensoes_faseB,
                        tap_atual_faseB=tap_atual[1],
                        setpoint_atual_faseB=setpoint_atual,
                        lado_forte_fonte_faseB=lado_forte_fonte[1],
                        tensao_bucha_faseC=tensao_bucha_faseC,
                        tensoes_pontos_faseC=tensoes_faseC,
                        tap_atual_faseC=tap_atual[2],
                        lado_forte_fonte_faseC=lado_forte_fonte[2],
                        setpoint_atual_faseC=setpoint_atual,
                        setpoint_ideal=self.set_point_ideal)

        # flush any remaining buffered rows
        self._flush_element_buffer()
        self._flush_bus_buffer()
        self._flush_measurement_buffer()
        self._flush_tap_buffer()
        self._flush_pesos_buffer()

        # Do not load the full voltage_bus.csv into memory to save RAM.
        # The incremental CSV remains on disk at self.path_result_bus for post-processing.
        self.all_bus_kv = None

# ==========================
# Infra de execução
# ==========================
@dataclass
class Task:
    feeder: str
    month: int
    type_day: str
    bus_medicao_faseA: List[Tuple[str, str]]
    bus_medicao_faseB: List[Tuple[str, str]]
    bus_medicao_faseC: List[Tuple[str, str]]
    regcontrolname: str
    num_patamares: int
    patamar_ini: int
    patamar_fim: int
    config: Dict


def find_file(filename: str, search_path: str):
    for root, dirs, files in os.walk(search_path):
        if filename in files:
            return Path(root) / filename
    return None

def run_feeder_mode(substation, feeder, months, type_days, bus_medicao_faseA, bus_medicao_faseB, bus_medicao_faseC,
                    regcontrolname, num_patamares, patamar_ini, patamar_fim, config):

    master_filename = f"{type_days[0]}_{months[0]}_Master_391_{substation}_{feeder}_{num_patamares}.dss"
    feeder_path = Path(config["feeder_path"]).resolve()
    master_path = find_file(master_filename, search_path=feeder_path)
    if master_path is None:
        print(f"❌ Master file não encontrado: {master_filename}")
        return

    print(f"🚀 Processando o Master: {master_filename} | {multiprocessing.current_process().name}")

    simul = SmartRT(feeder=feeder,
                    dss_file=master_path,
                    bus_medicao_faseA=bus_medicao_faseA,
                    bus_medicao_faseB=bus_medicao_faseB,
                    bus_medicao_faseC=bus_medicao_faseC,
                    regcontrolname=regcontrolname,
                    num_patamares=num_patamares,
                    patamar_ini=patamar_ini,
                    patamar_fim=patamar_fim,
                    record_only_violations=config["record_only_violations"],
                    usar_setup_dinamico=config["usar_setup_dinamico"])

    if config["usar_setup_dinamico"]:
        simul.regcontrol_tsea_init()

    simul.solve_circuit()

    print(f"✅ Alimentador {master_filename} processado com sucesso.")

def process_task(task: Task):
    feeders = task.feeder
    months = task.month
    type_days = task.type_day
    bus_medicao_faseA = task.bus_medicao_faseA
    bus_medicao_faseB = task.bus_medicao_faseB
    bus_medicao_faseC = task.bus_medicao_faseC
    regcontrolname = task.regcontrolname
    num_patamares = task.num_patamares
    patamar_ini = task.patamar_ini
    patamar_fim = task.patamar_fim
    config = task.config

    if isinstance(feeders, str):
        feeders = [feeders]
    if isinstance(months, (str, int)):
        months = [months]
    if isinstance(type_days, str):
        type_days = [type_days]

    for feeder in feeders:
        run_feeder_mode(
            substation=feeder[1:4],
            feeder=feeder,
            months=months,
            type_days=type_days,
            bus_medicao_faseA=bus_medicao_faseA,
            bus_medicao_faseB=bus_medicao_faseB,
            bus_medicao_faseC=bus_medicao_faseC,
            regcontrolname=regcontrolname,
            num_patamares=num_patamares,
            patamar_ini=patamar_ini,
            patamar_fim=patamar_fim,
            config=config
        )

def to_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]

def build_tasks_from_config(config: Dict) -> List[Task]:
    feeders = to_list(config.get("feeder", []))
    months = to_list(config.get("month", []))
    type_days = to_list(config.get("type_day", []))
    num_patamares = config["num_patamares"]
    patamar_ini = config["patamar_ini"]
    patamar_fim = config["patamar_fim"]

    tasks: List[Task] = []

    for feeder in feeders:
        pontos_med_faseA = config["points"][f"{feeder[0:8]}"]["Node1"]
        pontos_med_faseB = config["points"][f"{feeder[0:8]}"]["Node2"]
        pontos_med_faseC = config["points"][f"{feeder[0:8]}"]["Node3"]
        reguladores = config["points"][f"{feeder[0:8]}"]["Reguladores"]

        for m in months:
            for td in type_days:
                tasks.append(Task(
                    feeder=str(feeder),
                    month=int(m),
                    type_day=str(td),
                    bus_medicao_faseA=pontos_med_faseA,
                    bus_medicao_faseB=pontos_med_faseB,
                    bus_medicao_faseC=pontos_med_faseC,
                    regcontrolname=str(reguladores),
                    num_patamares=int(num_patamares),
                    patamar_ini=int(patamar_ini),
                    patamar_fim=int(patamar_fim),
                    config=config)
                )

    return tasks


if __name__ == '__main__':
    application_path = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(application_path, "config_smartRT.yml")

    inicio = time.time()

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)["data_SmartRT"]

    tasks = build_tasks_from_config(config)
    if not tasks:
        print("Nenhuma tarefa criada. Verifique o config_smartRT.yml.")
        sys.exit(1)

    if config["multiprocess"]:
        cpu_cores = max(multiprocessing.cpu_count() - 3, 1)
        print(f"⚡ Utilizando {cpu_cores} processadores.")

        with multiprocessing.Pool(processes=cpu_cores) as pool:
            pool.map(process_task, tasks)

    else:
        process_task(tasks[0])

    fim = time.time()
    tempo_total = fim - inicio

    horas = int(tempo_total // 3600)
    minutos = int((tempo_total % 3600) // 60)
    segundos = int(tempo_total % 60)

    print(f"Tempo total de execução: {horas:02d}h{minutos:02d}min{segundos:02d}seg")
    print("✅ Execução Completa")
