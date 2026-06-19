import py_dss_interface
import pandas as pd
import numpy as np
import polars as pl
import time
import os
import matplotlib
import yaml
from pathlib import Path
from dataclasses import dataclass
from DRP_DRC import demanda_load, indic_DRP_DRC

matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

pd.set_option('display.max_rows', None)

def get_loads(dss, dss_file):
    """
    Retorna listas de cargas separadas em BT e MT
    """
    loads_bt = list()
    loads_mt = list()
    loads_pip = list()
    loads_TOTAL = list()

    dss.text(f"compile [{dss_file}]")

    dss.loads.first()
    for _ in range(dss.loads.count):
        dss.circuit.set_active_element(f"load[{dss.loads.name}]")
        load = dss.loads.name
        loads_TOTAL.append(load)

        if load.lower().startswith("pip"):
            loads_pip.append(load)
            dss.loads.next()
            continue

        if load.lower().endswith("m1"):
            bus_load = dss.cktelement.bus_names[0].split(".")[0]
            dss.circuit.set_active_bus(bus_load)
            v_base = dss.bus.kv_base * 1000

            if v_base <= 1000:
                loads_bt.append(load)

            if v_base >= 1000:
                loads_mt.append(load)

            dss.loads.next()

        else:
            dss.loads.next()

    total_loads_bt = len(loads_bt)
    total_loads_mt = len(loads_mt)

    return total_loads_bt, total_loads_mt

def get_nodes(dss, dss_file):
    """
    Retorna uma lista de nós violados
    """
    total_nodes_bt = list()
    total_nodes_mt = list()

    dss.text(f"compile [{dss_file}]")
    nodes_names = dss.circuit.nodes_names

    for node in nodes_names:
        if node.lower().startswith("mt") or node.split(".")[0].lower().startswith("busa"):
            total_nodes_mt.append(node)

        if node.lower().startswith("bt"):
            # if not node.lower().startswith("bt4") and not node.lower().endswith("4"):
            total_nodes_bt.append(node)

    total_nodes_names = len(nodes_names)
    total_nodes_bt = len(total_nodes_bt)
    total_nodes_mt = len(total_nodes_mt)

    return total_nodes_bt, total_nodes_mt, total_nodes_names

def plotar_voltage(df, titulo, tipo, feeder, caminho_arquivo, patamar_ini, patamar_fim):
    all_numbers = np.arange(patamar_ini, patamar_fim + 1)

    df_count = (df.groupby(["Number", "Occurrence"]).size().unstack(fill_value=0).reindex(all_numbers, fill_value=0))

    ordem = [
        "Subtensão Precária",
        "Subtensão Crítica",
        "Sobretensão Precária",
        "Sobretensão Crítica"
    ]

    df_count = df_count.reindex(columns=ordem, fill_value=0)

    grupos = [("subtensao", ["Subtensão Precária", "Subtensão Crítica"]),
             ("sobretensao", ["Sobretensão Precária", "Sobretensão Crítica"])]

    caminho_arquivo.parent.mkdir(parents=True, exist_ok=True)

    for nome_grupo, colunas in grupos:
        fig, ax = plt.subplots(figsize=(12, 6))
        bottom = np.zeros(len(df_count))

        for col in colunas:
            values = df_count[col].values

            ax.bar(
                df_count.index,
                values,
                bottom=bottom,
                label=col
            )

            bottom += values

        ax.set_xlabel("Passo de Tempo")
        ax.set_ylabel("Quantidade")

        if nome_grupo == "subtensao":
            ax.set_title(f"{titulo} {tipo}: Subtensão - {feeder}")
        else:
            ax.set_title(f"{titulo} {tipo}: Sobretensão - {feeder}")

        ax.legend(title=None)
        ax.grid(axis="y", linestyle="-", alpha=0.6, color="gray")

        if patamar_fim not in (24, 144):
            step = int(max(1, len(all_numbers) // (patamar_fim / 690))) # TODO
            xticks = all_numbers[::step]
            ax.tick_params(axis='x', rotation=90)
            ax.set_xticks(xticks)
            ax.margins(x=0)

        else:
            step = max(1, len(all_numbers) // 24)
            xticks = all_numbers[::step]
            ax.set_xticks(xticks)
            ax.tick_params(axis='x', rotation=90)
            ax.margins(x=0)

        arquivo_saida = (caminho_arquivo.parent / f"{caminho_arquivo.stem}_{nome_grupo}{caminho_arquivo.suffix}")
        plt.savefig(arquivo_saida, dpi=300, bbox_inches="tight")
        plt.close()

def plotar_voltage_per(df, titulo, tipo, feeder, caminho_arquivo, total_loads, patamar_ini, patamar_fim):
    all_numbers = np.arange(patamar_ini, patamar_fim + 1)

    df_count = (df.groupby(["Number", "Occurrence"]).size().unstack(fill_value=0).reindex(all_numbers, fill_value=0))

    ordem = [
        "Subtensão Precária",
        "Subtensão Crítica",
        "Sobretensão Precária",
        "Sobretensão Crítica"
    ]

    df_count = df_count.reindex(columns=ordem, fill_value=0)
    df_percent = (df_count / total_loads) * 100

    grupos = [("subtensao", ["Subtensão Precária", "Subtensão Crítica"]),
              ("sobretensao", ["Sobretensão Precária", "Sobretensão Crítica"])]

    caminho_arquivo.parent.mkdir(parents=True, exist_ok=True)

    for nome_grupo, colunas in grupos:
        fig, ax = plt.subplots(figsize=(12, 6))
        bottom = np.zeros(len(df_count))

        for col in colunas:
            values = df_percent[col].values

            ax.bar(
                df_percent.index,
                values,
                bottom=bottom,
                label=col
            )

            bottom += values

        ax.set_xlabel("Passo de Tempo")
        ax.set_ylabel("Quantidade (%)")

        if nome_grupo == "subtensao":
            ax.set_title(f"{titulo} {tipo}: Subtensão - {feeder}")
        else:
            ax.set_title(f"{titulo} {tipo}: Sobretensão - {feeder}")

        ax.legend(title=None)
        ax.grid(axis="y", linestyle="-", alpha=0.6, color="gray")

        if patamar_fim not in (24, 144):
            step = int(max(1, len(all_numbers) // (patamar_fim / 690))) # TODO
            xticks = all_numbers[::step]
            ax.tick_params(axis='x', rotation=90)
            ax.set_xticks(xticks)
            ax.margins(x=0)

        else:
            step = max(1, len(all_numbers) // 24)
            xticks = all_numbers[::step]
            ax.set_xticks(xticks)
            ax.tick_params(axis='x', rotation=90)
            ax.margins(x=0)

        arquivo_saida = (caminho_arquivo.parent / f"{caminho_arquivo.stem}_{nome_grupo}{caminho_arquivo.suffix}")
        plt.savefig(arquivo_saida, dpi=300, bbox_inches="tight")
        plt.close()

def plotar_measurement(df, caminho_arquivo, measurement_node1, measurement_node2, measurement_node3, patamar_ini, patamar_fim):
    all_numbers = np.arange(patamar_ini, patamar_fim + 1)

    df["Bus_Node"] = (df["Bus"].astype(str).str.lower() + "." + df["Node"].astype(str))

    measurement_node1 = [x.lower() for x in measurement_node1]
    measurement_node2 = [x.lower() for x in measurement_node2]
    measurement_node3 = [x.lower() for x in measurement_node3]

    fig, axes = plt.subplots(nrows=3, ncols=1, sharey=True , figsize=(10, 9))

    grupos = [
        ("fase A", measurement_node1),
        ("fase B", measurement_node2),
        ("fase C", measurement_node3)
    ]

    mt_over_crit = 1.05
    mt_under_prec = 0.93
    mt_under_crit = 0.90
    bt_over_crit = 1.06
    bt_over_prec = 1.047
    bt_under_prec = 0.92
    bt_under_crit = 0.866

    for ax, (titulo_node, measurement_nodes) in zip(axes, grupos):

        df_group = df[df["Bus_Node"].isin(measurement_nodes)]

        for bus_node in measurement_nodes:
            df_bus = (df_group[df_group["Bus_Node"] == bus_node].sort_values(by="Number"))
            df_bus = (df_bus.set_index("Number").reindex(all_numbers))

            ax.plot(
                all_numbers,
                df_bus["Voltage"],
                linewidth=1.5,
                label=bus_node
            )

        ax.plot([patamar_ini, patamar_fim], [mt_over_crit, mt_over_crit], color='r', linestyle='--', alpha=0.8)
        ax.plot([patamar_ini, patamar_fim], [mt_under_prec, mt_under_prec], color='r', linestyle='--', alpha=0.8)
        ax.plot([patamar_ini, patamar_fim], [mt_under_crit, mt_under_crit], color='r', linestyle='--', alpha=0.8)
        ax.plot([patamar_ini, patamar_fim], [bt_over_crit, bt_over_crit], color='g', linestyle='--', alpha=0.8)
        ax.plot([patamar_ini, patamar_fim], [bt_over_prec, bt_over_prec], color='g', linestyle='--', alpha=0.8)
        ax.plot([patamar_ini, patamar_fim], [bt_under_prec, bt_under_prec], color='g', linestyle='--', alpha=0.8)
        ax.plot([patamar_ini, patamar_fim], [bt_under_crit, bt_under_crit], color='g', linestyle='--', alpha=0.8)

        ax.set_title(f"Tensão na Barra - {titulo_node}", fontsize=10)
        ax.set_xlim(patamar_ini, patamar_fim)
        ax.tick_params(axis='both', labelsize=10)
        ax.legend(fontsize=8, loc='upper right')

    plt.xlabel(f"Passo de Tempo", fontsize=10)
    fig.tight_layout()

    caminho_arquivo.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(caminho_arquivo, dpi=300, bbox_inches="tight")
    plt.close()

def plotar_taps(df, caminho_arquivo, patamar_ini, patamar_fim):

    all_numbers = np.arange(patamar_ini, patamar_fim + 1)

    fases = [
        ('A', 'Fase A'),
        ('B', 'Fase B'),
        ('C', 'Fase C')
    ]

    for fase, titulo_fase in fases:

        df_fase = (df[df['Fase'] == fase].sort_values(by='Number').set_index('Number').reindex(all_numbers))

        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 8))

        axes[0].plot(all_numbers, df_fase['Tap'], linewidth=1.5, color='blue')
        axes[0].set_title(f'TAP {titulo_fase}', fontsize=10)
        axes[0].tick_params(axis='both', labelsize=10)

        axes[1].plot(all_numbers, df_fase['V_ref'], linewidth=1.5, color='red')
        axes[1].set_title('Vref', fontsize=10)
        axes[1].tick_params(axis='both', labelsize=10)

        axes[2].plot(all_numbers, df_fase['V_reg'], linewidth=1.5, color='green')
        axes[2].set_title('Vreg', fontsize=10)
        axes[2].grid(axis='y', linestyle='-', alpha=0.6, color='gray')
        axes[2].tick_params(axis='both', labelsize=10)
        axes[2].set_xlabel('Passo de Tempo', fontsize=10)

        fig.tight_layout()

        arquivo_saida = (caminho_arquivo.parent / f"{caminho_arquivo.stem}_{fase}{caminho_arquivo.suffix}")
        plt.savefig(arquivo_saida, dpi=300, bbox_inches='tight')
        plt.close()


@dataclass
class Analisys:
    data_circuit: str
    data_path: str
    pesos_data: str
    data_dir = str
    data_file = str
    pesos_data_path= str
    load_demand = []

    def __post_init__(self):
        full_path = Path(self.data_path)
        self.data_dir = full_path.parent
        self.data_file = full_path.name
        #self.data_circuit = full_path.parts[-2]

        self.load_demand = demanda_load(self.data_circuit)

        self.pesos_data_path  = os.path.join(self.data_dir, self.pesos_data)

    def plot_voltage_by_pesos(self, buses_phases):

        df_all = (pl.scan_csv(self.pesos_data_path)
                  .select('patamar', 'voltage_list_faseA', 'voltage_list_faseB', 'voltage_list_faseC')
                  # .filter((pl.col("patamar") >= 0) & (pl.col("patamar") < 1000) )
                  )

        dados = df_all.collect().to_pandas()
        dados_plot = []
        for index, buses_phase in enumerate(buses_phases):
            dados_fase = pd.DataFrame(dados.iloc[:, index + 1]
                                      .str.replace('[', '', regex=False)
                                      .str.replace(']', '', regex=False))
            dados_fase[buses_phase] = dados_fase[dados_fase.columns[0]].str.split(',', expand=True)
            dados_fase = dados_fase.drop(columns=[dados_fase.columns[0]])
            dados_fase = dados_fase.astype(float)
            dados_plot.append(dados_fase)

        fases = ['fase A', 'fase B', 'fase C']

        # plot gráficos
        sup_limit = 1.05
        inf_limit = 0.95
        inf_limit_bt = 0.93
        lw = 0.75
        fig, axes = plt.subplots(nrows=3, ncols=1, sharey=True, figsize=(10, 9))

        dados_plot[0].plot(ax=axes[0], title=f"Voltage Bus - {fases[0]}").legend(fontsize=8)
        axes[0].axhline(y=1.05, color='r', linestyle='--', alpha=0.8, label='mt_over_crit')
        axes[0].axhline(y=0.93, color='r', linestyle='--', alpha=0.8, label='mt_under_prec')
        axes[0].axhline(y=0.90, color='r', linestyle='--', alpha=0.8, label='mt_under_crit')
        axes[0].axhline(y=1.06, color='g', linestyle='--', alpha=0.8, label='bt_over_crit')
        axes[0].axhline(y=1.047, color='g', linestyle='--', alpha=0.8, label='bt_over_prec')
        axes[0].axhline(y=0.866, color='g', linestyle='--', alpha=0.8, label='bt_under_prec')  # 110/127
        axes[0].axhline(y=0.92, color='g', linestyle='--', alpha=0.8, label='bt_under_crit')

        dados_plot[1].plot(ax=axes[1], title=f"Voltage Bus - {fases[1]}").legend(fontsize=8)
        axes[1].axhline(y=1.05, color='r', linestyle='--', alpha=0.8, label='mt_over_crit')
        axes[1].axhline(y=0.93, color='r', linestyle='--', alpha=0.8, label='mt_under_prec')
        axes[1].axhline(y=0.90, color='r', linestyle='--', alpha=0.8, label='mt_under_crit')
        axes[1].axhline(y=1.06, color='g', linestyle='--', alpha=0.8, label='bt_over_crit')
        axes[1].axhline(y=1.047, color='g', linestyle='--', alpha=0.8, label='bt_over_prec')
        axes[1].axhline(y=0.866, color='g', linestyle='--', alpha=0.8, label='bt_under_prec')
        axes[1].axhline(y=0.92, color='g', linestyle='--', alpha=0.8, label='bt_under_crit')

        dados_plot[2].plot(ax=axes[2], title=f"Voltage Bus - {fases[2]}").legend(fontsize=8)
        axes[2].axhline(y=1.05, color='r', linestyle='--', alpha=0.8, label='mt_over_crit')
        axes[2].axhline(y=0.93, color='r', linestyle='--', alpha=0.8, label='mt_under_prec')
        axes[2].axhline(y=0.90, color='r', linestyle='--', alpha=0.8, label='mt_under_crit')
        axes[2].axhline(y=1.06, color='g', linestyle='--', alpha=0.8, label='bt_over_crit')
        axes[2].axhline(y=1.047, color='g', linestyle='--', alpha=0.8, label='bt_over_prec')
        axes[2].axhline(y=0.866, color='g', linestyle='--', alpha=0.8, label='bt_under_prec')
        axes[2].axhline(y=0.92, color='g', linestyle='--', alpha=0.8, label='bt_under_crit')

        # plt.title(f"Voltage Bus - {self.data_circuit}")
        plt.xlabel(f"Time steps")
        plt.tight_layout()  # Prevents label overlapping
        plt.grid(axis='y')
        plt_path = os.path.join(self.data_dir, f"Voltage_bus_fases.png")
        plt.savefig(plt_path)

    """
    def plot_voltage(self, buses_phases):
        list_buses = []
        list_nodes = []
        # scan_csv doesn't load the file; it creates a plan
        df_all = pl.scan_csv(self.data_path).select('patamar', 'bus', 'nodes', 'vln_pu')

        fases = ['faseA', 'faseB', 'faseC']

        for buses_phase in buses_phases:
            for bus_phase in buses_phase:
                bus_atual, bus_node = bus_phase.lower().split('.', 1)
                list_buses.append(bus_atual)
                list_nodes.append(int(bus_node) )

            df_filtros = pl.LazyFrame({
                "bus": list_buses,
                "nodes": list_nodes
            })

            df = df_all.join(df_filtros,
                on=["bus", "nodes"],
                how="inner"
            )

            df = df.select(["patamar", "bus", "vln_pu"]).filter(pl.col("patamar") >= 0 &
                                                                (pl.col("patamar") < 10 ))
            dados = df.collect().to_pandas()
            dados = dados.pivot(index='patamar', columns='bus', values='vln_pu')


        for index, fase in enumerate(fases):
            df = df_all.filter(pl.col("bus").is_in(buses) & (pl.col("nodes")==(index+1) ) )
            df = df.select(["patamar", "bus", "vln_pu"])
            dados = df.collect().to_pandas()
            dados = dados.pivot(index='patamar', columns='bus', values='vln_pu')

            #print(dados)
            ax = dados.plot( y=buses, figsize=(7, 5))
            plt.title(f"BUS voltage: - {self.data_circuit} - {fase}")
            plt.ylabel(f"Voltage (p.u.)")
            plt.xlabel(f"Time steps")
            plt.grid(axis='y')
            ax.legend(loc='upper right', bbox_to_anchor=(1, 1), fontsize='small')
            #ax.xaxis.set_major_locator(ticker.MultipleLocator(690))
            plt_path = os.path.join(self.data_dir, f"bus_medicoes_voltages_{fase}.png")
            #plt.savefig(plt_path, dpi=300, bbox_inches='tight', transparent=False)
            plt.savefig(plt_path)
            plt.show()
    """

    def plot_taps(self):
        df = pl.scan_csv(self.pesos_data_path)
        df = df.select(["patamar", "vreg", "tap_faseA", "tap_faseB", "tap_faseC",
                        "reg_voltage_faseA", "reg_voltage_faseB", "reg_voltage_faseC"])
        dados = df.collect().to_pandas()

        fases = ['faseA', 'faseB', 'faseC']
        tap_max = int(dados[['tap_faseA', 'tap_faseB', 'tap_faseC']].max().max() + 1)
        tap_min = int(dados[['tap_faseA', 'tap_faseB', 'tap_faseC']].min().min() - 1)
        vreg_max = dados["vreg"].max().max() * 1.1
        vreg_min = dados["vreg"].max().min() * 0.9
        reg_volt_max = round(dados[['reg_voltage_faseA', 'reg_voltage_faseB', 'reg_voltage_faseC']].max().max() * 1.02, 3)
        reg_volt_mim = round(dados[['reg_voltage_faseA', 'reg_voltage_faseB', 'reg_voltage_faseC']].min().min() * 0.95, 3)

        for fase in fases:
            fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 8))

            dados[f"tap_{fase}"].plot(ax=axes[0], ylim=(tap_min, tap_max), title=f"TAP Change {fase}", color='blue')
            axes[0].grid(True, linestyle='--', alpha=0.6, color='gray')
            dados["vreg"].plot(ax=axes[1], ylim=(vreg_min, vreg_max), title="Vref", color='red')
            axes[1].grid(True, linestyle='--', alpha=0.6, color='gray')
            dados[f"reg_voltage_{fase}"].plot(ax=axes[2], ylim=(reg_volt_mim, reg_volt_max), title="Vreg", color='green')
            axes[2].grid(True, linestyle='--', alpha=0.6, color='gray')

            plt.xlabel(f"Time steps")
            plt.tight_layout()  # Prevents label overlapping
            # plt.grid(axis='y')
            plt_path = os.path.join(self.data_dir, f"tap_change_{fase}.png")
            plt.savefig(plt_path)
            plt.close()

    def plot_results(self, dados):
        circuit = self.data_circuit
        # plt_path_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados", circuit)
        plt_path_base = self.data_dir
        # pdf = dados.to_pandas()
        pdf = dados
        mt_df = pdf[['mt_undervolt_crit', 'mt_undervolt_prec', 'mt_overvolt_crit']].copy()
        bt_df = pdf[['bt_undervolt_prec', 'bt_undervolt_crit', 'bt_overvolt_prec', 'bt_overvolt_crit']].copy()

        counts_bt_under_prec_perc = bt_df['bt_undervolt_prec'] / pdf['cnt_bt_bus'][0] * 100
        counts_bt_under_crit_perc = bt_df['bt_undervolt_crit'] / pdf['cnt_bt_bus'][0] * 100
        counts_bt_over_prec_perc = bt_df['bt_overvolt_prec'] / pdf['cnt_bt_bus'][0] * 100
        counts_bt_over_crit_perc = bt_df['bt_overvolt_crit'] / pdf['cnt_bt_bus'][0] * 100
        bt_df_under = pd.DataFrame({'bt_undervoltage_prec': counts_bt_under_prec_perc,
                                    'bt_undervoltage_crit': counts_bt_under_crit_perc})

        bt_df_over = pd.DataFrame({'bt_overvoltage_prec': counts_bt_over_prec_perc,
                                   'bt_overvoltage_crit': counts_bt_over_crit_perc})

        counts_mt_under_prec_perc = mt_df['mt_undervolt_prec'] / pdf['cnt_mt_bus'][0] * 100
        counts_mt_under_crit_perc = mt_df['mt_undervolt_crit'] / pdf['cnt_mt_bus'][0] * 100
        counts_mt_over_crit_perc = mt_df['mt_overvolt_crit'] / pdf['cnt_mt_bus'][0] * 100
        mt_df_perc = pd.DataFrame({'mt_undervoltage_prec': counts_mt_under_prec_perc,
                                   'mt_undervoltage_crit': counts_mt_under_crit_perc,
                                   'mt_overvoltage_crit': counts_mt_over_crit_perc})

        espacamento = 690
        if not bt_df.empty:
            max_under = bt_df_under.max().sum()
            max_over = bt_df_over.max().sum()
            escala_max = int(max(max_under, max_over) * 1.1)

            ax = bt_df.plot(kind='bar', stacked=True)
            plt.title(f"BUS Violation : {circuit}")
            plt.ylabel(f"Number")
            plt.xlabel(f"Time steps")
            plt.grid(axis='y')
            ax.xaxis.set_major_locator(ticker.MultipleLocator(espacamento))
            plt_path = os.path.join(plt_path_base, "bt_voltages.png")
            plt.savefig(plt_path, dpi=600, bbox_inches='tight', transparent=False)
            plt.close()

            # grafico de porcentagem - undervoltage
            ax = bt_df_under.plot(kind='bar', ylim=(0, escala_max), stacked=True)
            plt.title(f"BUS Violation: Undervoltage - {circuit}")
            plt.ylabel(f"Number (%)")
            plt.xlabel(f"Time steps")
            plt.grid(axis='y')
            ax.xaxis.set_major_locator(ticker.MultipleLocator(espacamento))
            plt_path = os.path.join(plt_path_base, "bt_voltages_under.png")
            plt.savefig(plt_path, dpi=600, bbox_inches='tight', transparent=False)
            # plt.show(block=False)
            plt.close()

            # grafico de porcentagem - overvoltage
            ax = bt_df_over.plot(kind='bar', ylim=(0, escala_max), stacked=True)
            plt.title(f"BUS Violation: Overvoltage - {circuit}")
            plt.ylabel(f"Number (%)")
            plt.xlabel(f"Time steps")
            plt.grid(axis='y')
            ax.xaxis.set_major_locator(ticker.MultipleLocator(espacamento))
            plt_path = os.path.join(plt_path_base, "bt_voltages_over.png")
            plt.savefig(plt_path, dpi=600, bbox_inches='tight', transparent=False)
            plt.close()

        else:
            print("Sem violação de tensão BT.")

        if not mt_df.empty:
            if self.setup_dinamico == 'True':
                escala_max = 25
            else:
                escala_max = int(mt_df_perc.max().sum() * 1.1)

            ax = mt_df.plot(kind='bar', stacked=True)
            plt.title(f"BUS Violation : {circuit}")
            plt.ylabel(f"Number")
            plt.xlabel(f"Time")
            ax.xaxis.set_major_locator(ticker.MultipleLocator(espacamento))
            plt_path = os.path.join(plt_path_base, "mt_voltages.png")
            plt.savefig(plt_path, dpi=600, bbox_inches='tight', transparent=False)
            plt.close()

            # grafico de porcentagem
            ax = mt_df_perc.plot(kind='bar', ylim=(0, escala_max), stacked=True)  # Todo avaliar para cada caso...
            plt.title(f"BUS Violation : {circuit}")
            plt.ylabel(f"Number (%)")
            plt.xlabel(f"Time")
            ax.xaxis.set_major_locator(ticker.MultipleLocator(espacamento))
            plt_path = os.path.join(plt_path_base, "mt_voltages_perc.png")
            plt.savefig(plt_path, dpi=600, bbox_inches='tight', transparent=False)
            plt.close()
        else:
            print("Sem violação de tensão MT.")

    def polar_read_csv(self):
        proc_time_ini = time.time()
        results_combined = pd.DataFrame()
        load_demand = pl.from_pandas(self.load_demand).lazy().with_columns([
            pl.col("bus").cast(pl.Categorical),
        ])
        all_dados_drc_drp = pd.DataFrame()

        bloco_ini = -1
        points = np.linspace(0, 17280, 5)
        points = points[1:]
        # Processa os dados em bloco - evita estouro de memoria
        for bloco in points:
            # bloco = 10   # só para teste... remover!!!
            # scan_csv doesn't load the file; it creates a plan
            lazy_df = ((pl.scan_csv(self.data_path).select(
                ["patamar", "bus", "nodes", "kv_base", "vln_pu", "vln"])).with_columns([
                pl.col("bus").cast(pl.Categorical),
                pl.col("patamar").cast(pl.UInt16),
                pl.col("kv_base").cast(pl.Float32),
                pl.col("vln_pu").cast(pl.Float32),
                pl.col("vln").cast(pl.Float32),
            ])
                       .filter((pl.col("patamar") > bloco_ini) & (pl.col("patamar") <= bloco))
                       )

            # results = lazy_df.collect(engine="streaming").to_pandas()
            # print('')
            # ------------------------------------------------------------------
            # Obtém o MAIOR valor por NODE dentro de cada patamar
            # ------------------------------------------------------------------
            node_max_df = (
                lazy_df
                .group_by(["patamar", "bus", ])
                .agg(
                    pl.col("kv_base").first().alias("kv_base"),
                    pl.col("vln_pu").max().alias("max_vln_pu"),
                    pl.col("vln_pu").min().alias("min_vln_pu"),
                    pl.col("vln").max().alias("max_vln"),
                    pl.col("vln").min().alias("min_vln"),
                )
            )

            # results = node_max_df.collect(engine="streaming").to_pandas()
            # print('')

            # ------------------------------------------------------------------
            # Calcula os indicadores
            # ------------------------------------------------------------------
            filtered_df = (
                node_max_df
                .group_by("patamar")
                .agg(
                    ((pl.col("kv_base") > 1000) & (pl.col("max_vln_pu") > 1.05)).sum().alias("mt_overvolt_crit"),
                    ((pl.col("kv_base") > 1000) & (
                                (pl.col("min_vln_pu") >= 0.90) & (pl.col("min_vln_pu") < 0.93))).sum().alias(
                        "mt_undervolt_prec"),
                    ((pl.col("kv_base") > 1000) & (pl.col("min_vln_pu") < 0.90)).sum().alias("mt_undervolt_crit"),

                    ((pl.col("kv_base") == 127) & ((pl.col("min_vln") > 0.2) & (pl.col("min_vln") < 110)) |
                     (pl.col("kv_base") == 120) & ((pl.col("min_vln") > 0.2) & (pl.col("min_vln") < 104))
                     ).sum().alias("bt_undervolt_crit"),
                    ((pl.col("kv_base") == 127) & ((pl.col("min_vln") >= 110) & (pl.col("min_vln") < 117)) |
                     (pl.col("kv_base") == 120) & ((pl.col("min_vln") >= 104) & (pl.col("min_vln") < 110))
                     ).sum().alias("bt_undervolt_prec"),
                    ((pl.col("kv_base") == 127) & ((pl.col("max_vln") > 133) & (pl.col("max_vln") <= 135)) |
                     (pl.col("kv_base") == 120) & ((pl.col("max_vln") > 126) & (pl.col("max_vln") <= 127))
                     ).sum().alias("bt_overvolt_prec"),
                    ((pl.col("kv_base") == 127) & (pl.col("max_vln") >= 135) |
                     (pl.col("kv_base") == 120) & (pl.col("max_vln") >= 127)
                     ).sum().alias("bt_overvolt_crit"),

                    (pl.col("kv_base") < 1000).sum().alias("cnt_bt_bus"),
                    (pl.col("kv_base") > 1000).sum().alias("cnt_mt_bus"),

                )
                .sort(pl.col("patamar").cast(pl.UInt16), descending=False)
            )

            # Filter the LazyFrame using the DataFrame das cargas
            # dados_drc_drp = node_max_df.filter(pl.col('bus').is_in(self.load_demand))
            dados_drc_drp = (
                node_max_df.join(other=load_demand, on="bus", how="inner", suffix="_right")
            ).sort("patamar", "bus")

            # teste = dados_drc_drp.collect(engine="streaming").to_pandas()
            # print("")

            # ------------------------------------------------------------------
            # Calcula os indicadores por carga
            # ------------------------------------------------------------------
            filtered_load = (
                dados_drc_drp
                .group_by("cod_id")
                .agg(
                    ((pl.col("kv_base") > 1000) & (pl.col("max_vln_pu") > 1.05)).sum().alias("mt_overvolt_crit"),
                    ((pl.col("kv_base") > 1000) & (
                            (pl.col("min_vln_pu") >= 0.90) & (pl.col("min_vln_pu") < 0.93))).sum().alias(
                        "mt_undervolt_prec"),
                    ((pl.col("kv_base") > 1000) & (pl.col("min_vln_pu") < 0.90)).sum().alias("mt_undervolt_crit"),

                    ((pl.col("kv_base") == 127) & ((pl.col("min_vln") > 0.2) & (pl.col("min_vln") < 110)) |
                     (pl.col("kv_base") == 120) & ((pl.col("min_vln") > 0.2) & (pl.col("min_vln") < 104))
                     ).sum().alias("bt_undervolt_crit"),
                    ((pl.col("kv_base") == 127) & ((pl.col("min_vln") >= 110) & (pl.col("min_vln") < 117)) |
                     (pl.col("kv_base") == 120) & ((pl.col("min_vln") >= 104) & (pl.col("min_vln") < 110))
                     ).sum().alias("bt_undervolt_prec"),
                    ((pl.col("kv_base") == 127) & ((pl.col("max_vln") > 133) & (pl.col("max_vln") <= 135)) |
                     (pl.col("kv_base") == 120) & ((pl.col("max_vln") > 126) & (pl.col("max_vln") <= 127))
                     ).sum().alias("bt_overvolt_prec"),
                    ((pl.col("kv_base") == 127) & (pl.col("max_vln") >= 135) |
                     (pl.col("kv_base") == 120) & (pl.col("max_vln") >= 127)
                     ).sum().alias("bt_overvolt_crit"),

                    (pl.col("kv_base") < 1000).sum().alias("cnt_bt_bus"),
                    (pl.col("kv_base") > 1000).sum().alias("cnt_mt_bus"),

                )
                # .sort(pl.col("patamar").cast(pl.UInt16), descending=False)
            )

            results = filtered_df.collect(engine="streaming").to_pandas()
            results_combined = pd.concat([results_combined, results], axis=0)

            dados = filtered_load.collect(engine="streaming").to_pandas()
            all_dados_drc_drp = pd.concat([all_dados_drc_drp, dados], axis=0)

            # print(f"Result:   {results}")
            print(f"Processo concluído para o bloco: {bloco} em {time.time() - proc_time_ini}")
            bloco_ini = bloco

        results_combined.reset_index(drop=True, inplace=True)
        all_dados_drc_drp = all_dados_drc_drp.groupby('cod_id').sum().reset_index()
        return results_combined, all_dados_drc_drp

    def plot_perfil_tensao(self,
                           path_file,
                           col_x1=" Distance1",
                           col_y1=" puV1",
                           col_x2=" Distance2",
                           col_y2=" puV2",
                           titulo=f"Perfil de Tensão do Circuito",
                           figsize=(10, 5),
                           mostrar_nos=False,
                           setup_dinamico="False",

                           ):

        for file in Path(path_file).glob(f'*{setup_dinamico}_EXP_Profile*.csv'):
            print(file.name)
            titulo = f"Perfil de Tensão - {self.data_circuit} - Hora: {file.name.split("_")[-1].split('.')[0]}h"
            df = pd.read_csv(file.absolute())

            plt.figure(figsize=figsize)
            color = ['blue', 'red', 'black']
            linetype = ['-', ':', ':', '--']
            super_limite = 1.1
            inf_limite = 0.80
            # Desenha cada segmento
            for _, row in df.iterrows():
                x = [row[col_x1], row[col_x2]]
                y = [row[col_y1], row[col_y2]]

                idcolor = row[' Color']
                tipo = row[' Linetype']

                # verifica o maior valor para a média tensão e altera a escala do gráfico
                if tipo == 0:
                    if row[col_y1] > super_limite:
                        super_limite = row[col_y1] * 1.03
                    if row[col_y2] > super_limite:
                        super_limite = row[col_y2] * 1.03

                lw = 1.5
                if tipo == 2:
                    lw = 0.75
                plt.plot(x, y, color=color[idcolor - 1], linestyle=linetype[tipo], linewidth=lw)

                # Opcional: mostrar pontos
                if mostrar_nos:
                    plt.scatter(x, y, color="red", s=15)

            # Configurações do gráfico
            plt.title(titulo, fontsize=14)
            plt.xlim(left=0)
            plt.xlabel("Distância (km)", fontsize=12)
            plt.ylabel("Tensâo (pu)", fontsize=12)
            plt.axhline(y=1.05, color='r', linestyle='--', label='mt_over_crit')
            plt.axhline(y=0.93, color='r', linestyle='--', label='mt_under_prec')
            plt.axhline(y=0.90, color='r', linestyle='--', label='mt_under_crit')
            plt.axhline(y=1.06, color='g', linestyle='--', label='bt_over_crit')
            plt.axhline(y=1.047, color='g', linestyle='--', label='bt_over_prec')
            plt.axhline(y=0.866, color='g', linestyle='--', label='bt_under_prec')
            plt.axhline(y=0.92, color='g', linestyle='--', label='bt_under_crit')
            plt.grid(True, linestyle="--", alpha=0.5)

            # Ajusta limites autom�ticos
            plt.tight_layout()
            plt.ylim(inf_limite, super_limite)
            plt_path = os.path.join(self.data_dir, f"{file.stem}.png")
            plt.savefig(plt_path)
            # plt.savefig(plt_path, dpi=600, bbox_inches='tight', transparent=False)
            # plt.show()
            plt.close()


    def read_csv(filename):
        # Define a chunk size (number of rows)
        chunk_size = 10000

        for chunk in pd.read_csv(filename, chunksize=chunk_size):
            # Process each chunk individually
            print(chunk.head())


if __name__ == "__main__":
    application_path = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(application_path, "config_smartRT.yml")
    dss = py_dss_interface.DSS()

    inicio = time.time()

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)["data_SmartRT"]

    feeders = config["feeder"]
    months = config["month"]
    type_days = config["type_day"]
    num_patamares = config["num_patamares"]
    patamar_ini = config["patamar_ini"]
    patamar_fim = config["patamar_fim"]
    record_only_violations = config["record_only_violations"]
    setup_dinamico = config["usar_setup_dinamico"]

    if record_only_violations == True:
        for feeder in feeders:
            print(f"🚀 Processando o Feeder: {feeder}")

            # 1. Caminho dos arquivos
            substation = feeder[1:4]
            measurement_node1 = config["points"][f"{feeder[0:8]}"]["Node1"]
            measurement_node2 = config["points"][f"{feeder[0:8]}"]["Node2"]
            measurement_node3 = config["points"][f"{feeder[0:8]}"]["Node3"]
            reguladores = config["points"][f"{feeder[0:8]}"]["Reguladores"]
            regulador = reguladores[0].lower()
            regulator_name = regulador[:-1]

            if setup_dinamico == True:
                file_taps = Path(f"E:/SmartRT/resultados/{num_patamares}/{feeder}/pesos.csv") #TODO - CONFIRMAR ENDEREÇO
                df_pesos = pd.read_csv(file_taps)
                df_pesos = df_pesos[["patamar", "vreg", "reg_voltage_faseA", "reg_voltage_faseB", "reg_voltage_faseC", "tap_faseA", "tap_faseB", "tap_faseC"]]
                df_taps = (pd.concat([
                            pd.DataFrame({
                                "Number": df_pesos["patamar"],
                                "Fase": fase,
                                "V_ref": df_pesos["vreg"],
                                "V_reg": df_pesos[f"reg_voltage_fase{fase}"],
                                "Tap": df_pesos[f"tap_fase{fase}"]
                            })
                            for fase in ["A", "B", "C"]],ignore_index=True).sort_values(["Number", "Fase"]).reset_index(drop=True))
                del df_pesos

            else:
                file_taps = Path(f"E:/SmartRT/resultados/{num_patamares}/{feeder}/taps.csv") #TODO - CONFIRMAR ENDEREÇO
                df_taps = pd.read_csv(file_taps)

            file_voltage_element = Path(f"E:/SmartRT/resultados/{num_patamares}/{feeder}/voltage_element.csv") #TODO - CONFIRMAR ENDEREÇO
            file_voltage_bus = Path(f"E:/SmartRT/resultados/{num_patamares}/{feeder}/voltage_bus.csv") #TODO - CONFIRMAR ENDEREÇO
            file_voltage_measurement = Path(f"E:/SmartRT/resultados/{num_patamares}/{feeder}/voltage_measurement.csv") #TODO - CONFIRMAR ENDEREÇO
            dss_file = Path(f"C:/SmartRT/feeders/{feeder}/{type_days}_{months}_Master_391_{substation}_{feeder}_{num_patamares}.dss") #TODO - CONFIRMAR ENDEREÇO

            # 2. Ler o Excel
            df_voltage_element = pd.read_csv(file_voltage_element)
            df_voltage_bus = pd.read_csv(file_voltage_bus)
            df_voltage_measurement = pd.read_csv(file_voltage_measurement)

            # 3. Filtrar apenas elementos que terminam com m1
            df_voltage_element = df_voltage_element[df_voltage_element["Element"].str.endswith("m1")]

            # 4. Remover duplicidade
            df_voltage_element = df_voltage_element.drop_duplicates(subset=["Number", "Element", "Occurrence"])
            df_voltage_bus = df_voltage_bus.drop_duplicates(subset=["Number", "Bus", "Node", "Occurrence"])

            # 5. Definir prioridade
            prioridade = {
                "sobre_critica": 4,
                "sub_critica": 3,
                "sobre_precaria": 2,
                "sub_precaria": 1
            }

            # 6. Criar coluna de prioridade
            df_voltage_element["priority"] = df_voltage_element["Occurrence"].map(prioridade).fillna(0)

            # 7. Selecionar pior caso por (Number, Element)
            df_voltage_element = df_voltage_element.loc[df_voltage_element.groupby(["Number", "Element"])["priority"].idxmax()]

            # 8. Dataframe filtrado
            df_voltage_element_filtered = df_voltage_element[["Number", "Element", "Occurrence"]]
            df_voltage_bus_filtered = df_voltage_bus[["Number", "Bus", "Node", "Occurrence"]]

            del df_voltage_element
            del df_voltage_bus

            # 9. Mapear ocorrências
            map_ocorrencias = {
                "sobre_critica": "Sobretensão Crítica",
                "sobre_precaria": "Sobretensão Precária",
                "sub_precaria": "Subtensão Precária",
                "sub_critica": "Subtensão Crítica"
            }

            df_voltage_element_filtered["Occurrence"] = df_voltage_element_filtered["Occurrence"].map(map_ocorrencias)
            df_voltage_bus_filtered["Occurrence"] = df_voltage_bus_filtered["Occurrence"].map(map_ocorrencias)

            # 10. Separar BT e MT
            df_voltage_element_bt = df_voltage_element_filtered[df_voltage_element_filtered["Element"].str.startswith("Load.bt")]
            df_voltage_element_mt = df_voltage_element_filtered[df_voltage_element_filtered["Element"].str.startswith("Load.mt")]
            df_voltage_bus_bt = df_voltage_bus_filtered[df_voltage_bus_filtered["Bus"].str.startswith("bt")]
            df_voltage_bus_mt = df_voltage_bus_filtered[df_voltage_bus_filtered["Bus"].str.startswith("mt")]

            del df_voltage_element_filtered
            del df_voltage_bus_filtered

            # 11. Contabilizar as cargas e as barras em BT e MT
            total_loads_bt, total_loads_mt = get_loads(dss, dss_file)
            total_nodes_bt, total_nodes_mt, total_nodes_names = get_nodes(dss, dss_file)

            # 12. Caminho para salvar os gráficos
            file_voltage_element_bt = file_voltage_element.parent / f"{feeder}_BT.png"
            file_voltage_element_mt = file_voltage_element.parent / f"{feeder}_MT.png"
            file_voltage_element_bt_per = file_voltage_element.parent / f"{feeder}_BT_per.png"
            file_voltage_element_mt_per = file_voltage_element.parent / f"{feeder}_MT_per.png"

            file_voltage_bus_bt = file_voltage_bus.parent / f"{feeder}_BT_bus.png"
            file_voltage_bus_mt = file_voltage_bus.parent / f"{feeder}_MT_bus.png"
            file_voltage_bus_bt_per = file_voltage_bus.parent / f"{feeder}_BT_bus_per.png"
            file_voltage_bus_mt_per = file_voltage_bus.parent / f"{feeder}_MT_bus_per.png"

            file_voltage_measurement = file_voltage_measurement.parent / f"{feeder}_measurement.png"

            file_taps_reg = file_taps.parent / f"{feeder}_taps.png"

            # 13. Função para gerar gráfico
            plotar_voltage(df_voltage_element_bt, f"CARGAS", "BT Violadas", feeder, file_voltage_element_bt, patamar_ini, patamar_fim)
            plotar_voltage(df_voltage_element_mt, f"CARGAS", "MT Violadas", feeder, file_voltage_element_mt, patamar_ini, patamar_fim)
            plotar_voltage_per(df_voltage_element_bt, f"CARGAS", "BT Violadas", feeder, file_voltage_element_bt_per, total_loads_bt, patamar_ini, patamar_fim)
            plotar_voltage_per(df_voltage_element_mt, f"CARGAS", "MT Violadas", feeder, file_voltage_element_mt_per, total_loads_mt, patamar_ini, patamar_fim)

            plotar_voltage(df_voltage_bus_bt, f"BARRAS", "BT Violadas", feeder, file_voltage_bus_bt, patamar_ini, patamar_fim)
            plotar_voltage(df_voltage_bus_mt, f"BARRAS", "MT Violadas", feeder, file_voltage_bus_mt, patamar_ini, patamar_fim)
            plotar_voltage_per(df_voltage_bus_bt, f"BARRAS", "BT Violadas", feeder, file_voltage_bus_bt_per, total_nodes_names, patamar_ini, patamar_fim)
            plotar_voltage_per(df_voltage_bus_mt, f"BARRAS", "MT Violadas", feeder, file_voltage_bus_mt_per, total_nodes_names, patamar_ini, patamar_fim)

            plotar_measurement(df_voltage_measurement, file_voltage_measurement, measurement_node1, measurement_node2, measurement_node3, patamar_ini, patamar_fim)

            plotar_taps(df_taps, file_taps_reg, patamar_ini, patamar_fim)

    else:
        for feeder in feeders:
            print(f"🚀 Processando o Feeder: {feeder}")

            csv_file = os.path.join(application_path, fr'.\resultados\{feeder}\voltage_bus.csv')

            # Leitura dos dados de configuração das pontos de medição e dos reguladores
            pontos_med_faseA = config["points"][f"{feeder[0:8]}"]["Node1"]
            pontos_med_faseB = config["points"][f"{feeder[0:8]}"]["Node2"]
            pontos_med_faseC = config["points"][f"{feeder[0:8]}"]["Node3"]
            reguladores = config["points"][f"{feeder[0:8]}"]["Reguladores"]


            # inicializa a classe de analise dos resultados
            print(f"Inicialização da classe de análise gráfica para {feeder}... ")
            results = Analisys(feeder, csv_file, "pesos.csv")
            # Análise das condições das barras ao longo do dia


            # prefil de tensões
            print("Gráficos de pefil de tensão...")
            results.plot_perfil_tensao(os.path.join(application_path, fr'cenarios\{feeder}'), setup_dinamico)

            # Tensões dos pontos de medição
            print("Gráficos de tensões nos pontos de medição...")
            results.plot_voltage_by_pesos(buses_phases=[pontos_med_faseA, pontos_med_faseB, pontos_med_faseC])

            # Taps - Vreg - Tensão do regulador
            print("Gráficos de Taps - Vreg - Tensão do regulador...")
            results.plot_taps()

            # Análise das condições das barras ao longo do dia
            print("Gráficos de análise de tensões de toddas as barras...")
            dados, dados_drc_drp = results.polar_read_csv()
            results.plot_results(dados)

            print("Calculando compesações - DRP - DRC")
            indic_DRP_DRC(feeder, dados_drc_drp, results.load_demand)

            exit()

    fim = time.time()
    tempo_total = fim - inicio

    horas = int(tempo_total // 3600)
    minutos = int((tempo_total % 3600) // 60)
    segundos = int(tempo_total % 60)

    print(f"Tempo total de execução: {horas:02d}h{minutos:02d}min{segundos:02d}seg")
    print("✅ Execução Completa")