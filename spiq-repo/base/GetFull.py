import json
import os
import pathlib 
import Scripts.ProblemGenerator as ProblemGenerator
import Scripts.QUBOGenerator as QUBOGenerator
import Scripts.CircuitGeneration as CircuitGenerator
import Scripts.TopologyGenerator as TopologyGenerator
import Scripts.Postprocessing as Postprocessing
from multiprocessing import Pool
import csv
import IBMQExperiments
import numpy as np
import config as config
import pickle
from decimal import *
from pathlib import Path
import math
import pandas as pd


ROWS_PER_PAGE = 5      
BUFFER_PAGES = 4

NUM_TABLES = 3



pred = []  

pred_sel = [] 

def bnlj_cost_pages(outer_pages: int, inner_pages: int, buffer_pages: int) -> int:
    block_pages = buffer_pages - 2
    return outer_pages + math.ceil(outer_pages / block_pages) * inner_pages

def ghj_passes(build_pages, buffer_pages):
    fanout = buffer_pages - 1
    mem = buffer_pages - 2
    if fanout <= 1 or mem <= 0:
        raise ValueError("buffer_pages too small for GHJ model")
    k = 0
    while build_pages / (fanout ** k) > mem:
        k += 1
    return k

def ghj_cost_pages(build_pages, probe_pages, buffer_pages):
    k = ghj_passes(build_pages, buffer_pages)
    return (2 * k + 1) * (build_pages + probe_pages)

def ghj_3table_total_cost(join_order, card_rows, rows_per_page=5, buffer_pages=4):
    pages = [rows_to_pages(r, rows_per_page) for r in card_rows]
    j0, j1, j2 = join_order
    cost1 = ghj_cost_pages(pages[j0], pages[j1], buffer_pages)
    out1_rows = card_rows[j0] * card_rows[j1]
    out1_pages = rows_to_pages(out1_rows, rows_per_page)
    write_out1 = out1_pages
    cost2 = ghj_cost_pages(out1_pages, pages[j2], buffer_pages)

    cost_bnj1=bnlj_cost_pages(pages[j0], pages[j1], buffer_pages)
    out_bnj1 = float(card_rows[j0]) * float(card_rows[j1])
    out1_bnj_pages = rows_to_pages(out_bnj1, rows_per_page)
    write_out1_bnj = out1_bnj_pages
    cost2_bnj = bnlj_cost_pages(out1_bnj_pages, pages[j2], buffer_pages)
    return cost1 + write_out1 + cost2, cost_bnj1+write_out1_bnj+cost2_bnj



def rows_to_pages(rows: float, rows_per_page: int = ROWS_PER_PAGE) -> int:
    return int(math.ceil(rows / rows_per_page))

def hash_join_step_cost_pages(build_pages: int, probe_pages: int,
                              buffer_pages: int = BUFFER_PAGES) -> int:
    if build_pages <= buffer_pages:
        return build_pages + probe_pages
    else:
        return 3 * (build_pages + probe_pages)
    
def nested_loop_join(join_order,card):
    j0,j1,j2=join_order[0],join_order[1],join_order[2]

def final_cost_for_join_order(join_order, card, pred, pred_sel):
    total = 0.0
    for j in range(1, NUM_TABLES): 
        total += float(
            Postprocessing.calculate_intermediate_cardinality_for_join(
                j, card, pred, pred_sel, join_order
            )
        )
    return total

def hash_join_total_cost_for_order(join_order, card_rows,
                                  rows_per_page: int = ROWS_PER_PAGE,
                                  buffer_pages: int = BUFFER_PAGES) -> int:
 
    if join_order is None:
        return np.nan

    j0, j1, j2 = join_order[0], join_order[1], join_order[2]
    build_rows = float(card_rows[j0])
    probe_rows = float(card_rows[j1])

    build_pages = rows_to_pages(build_rows, rows_per_page)
    probe_pages = rows_to_pages(probe_rows, rows_per_page)

    cost1 = hash_join_step_cost_pages(build_pages, probe_pages, buffer_pages)

    out1_rows = build_rows * probe_rows
    out1_pages = rows_to_pages(out1_rows, rows_per_page)

    cost1 += out1_pages

    probe2_rows = float(card_rows[j2])
    probe2_pages = rows_to_pages(probe2_rows, rows_per_page)

    cost2 = hash_join_step_cost_pages(out1_pages, probe2_pages, buffer_pages)

    return int(cost1 + cost2)


def saveFinalCost(inputPath):
    card = [10, 15, 20]
    input_csv = Path(inputPath)
    output_csv = input_csv.with_name(input_csv.stem + "_FINALCOST.csv")

    df = pd.read_csv(input_csv, dtype={"bitstring": str})


    out_rows = []
    for _, row in df.iterrows():
        idx_val = row["index"]
        bitstr = row["bitstring"]
        energy = row["energy"]
        prob=row["probability"]
        cost=row["cost"]

        sample = {i: int(ch) for i, ch in enumerate(bitstr)}

        join_order = Postprocessing.get_join_tree_leaves(sample, NUM_TABLES)
        join_order_str = "" if join_order is None else "-".join(map(str, join_order))

        if join_order is None:
            final_cost = np.nan
        else:
            final_cost = final_cost_for_join_order(join_order, card, pred, pred_sel)
            hash_cost,bnj_cost = ghj_3table_total_cost(
                join_order=join_order,
                card_rows=card,
                rows_per_page=ROWS_PER_PAGE,
                buffer_pages=BUFFER_PAGES
            )

        out_rows.append({
            "index": idx_val,
            "bitstring": bitstr,
            "energy": energy,
            "join_order": join_order_str,
            "GHJHashJoinCost": hash_cost,
            "BNLJoinCost":bnj_cost,
            "cost":cost,
            "probability":prob
        })

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(output_csv, index=False)





def parse_QPU_data(include_header=True,currentInput=0):
    inputPath=""
    # if include_header:
    #     IBMQExperiments.save_to_csv(['num_qaoa_iterations', 'num_predicates', 'valid_ratio', 'opt_ratio',f'Trial: {IBMQExperiments.TRIAL_ID}',f'Reps:{IBMQExperiments.TAG}',f'Optimizer:{IBMQExperiments.current_optim}'], 'ExperimentalAnalysis/IBMQ/QPUPerformance/Results', 'results.txt')
 
    # processing = config.configuration["ibmq-processing"]
    # if processing == "qpu":
    #     result_path_prefix = 'ExperimentalAnalysis/IBMQ/QPUPerformance/Results/QPU_Data'
    # elif processing == "cpu":
    #     result_path_prefix = 'ExperimentalAnalysis/IBMQ/QPUPerformance/Results/CPU_Data/'
    # else:
    #     result_path_prefix = 'ExperimentalAnalysis/IBMQ/QPUPerformance/Results/Collected_Data/'
        
    iterations_categories = [1000]
    thres_vals = {0:[150, 200, 300],1:[160, 200, 240, 280],2:[120, 150, 180, 220, 260, 300], 3: [10]}

    optimizers={"AQGD","COBYLA","SPSA"}
    for optimizer in optimizers:
        for i in range(3):
            input=f"./base/Week10/ExperimentalAnalysis/IBMQ/QPUPerformance/Results/CPU_Data/iterations_1000/reps_2/{optimizer}/bitstring_1/bitstring_valid_1000_{optimizer}_2_1_input{i}.csv"
            saveFinalCost(input)

            

if __name__ == '__main__':
    processing = config.configuration["ibmq-processing"]
    parse_QPU_data()