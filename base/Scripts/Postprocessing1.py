#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from math import prod
import math
import itertools
from math import inf
from sympy.utilities.iterables import multiset_permutations

from os import listdir
from os.path import isfile, join
from pathlib import Path
import time


# In[ ]:


def get_selectivity_for_new_relation(join_order, j, pred, pred_sel):
    sel = 1
    new_relation = join_order[j]
    for i in range(j):
        relation = join_order[i]
        if (relation, new_relation) in pred:
            sel = sel * pred_sel[pred.index((relation, new_relation))]
        elif (new_relation, relation) in pred:
            sel = sel * pred_sel[pred.index((new_relation, relation))]
    return sel

def get_intermediate_costs_for_join_order(join_order, card, pred, pred_sel, card_dict, verbose=False):
    int_costs = []
    join_order = join_order.copy()
    if join_order[0] > join_order[1]:
        join_order[0], join_order[1] = join_order[1], join_order[0]
    prev_join_result = card[join_order[0]]
    for j in range(1, len(card)-1):
        jo_hash = str(join_order[0:j+1])
        if jo_hash in card_dict:
            int_card = card_dict[jo_hash]
        else:
            sel = get_selectivity_for_new_relation(join_order, j, pred, pred_sel)
            int_card = prev_join_result * card[join_order[j]] * sel
            card_dict[jo_hash] = int_card
        prev_join_result = int_card
        int_costs.append(int_card)
    if verbose:
        print(int_costs)
    return int_costs

def calculate_cost_for_join(relations, card, pred, pred_sel):
    cost = 0
    card_prod = np.prod(np.array(card)[relations])
    pred_prod = 1
    for p in range(len(pred)):
        (r1, r2) = pred[p]
        if r1 in relations and r2 in relations:
            pred_prod = pred_prod * pred_sel[p]
    cost = card_prod * pred_prod
    return cost

def get_costs_for_leftdeep_tree(join_order, card, pred, pred_sel, card_dict, verbose=False):
    total_costs = 0 
    int_costs = get_intermediate_costs_for_join_order(join_order, card, pred, pred_sel, card_dict, verbose=verbose)
    for cost in int_costs:
        total_costs = total_costs + cost
    return total_costs

def get_costs_for_bushy_tree(join_list, card, pred, pred_sel):
    cost = 0
    for relations in join_list:
        if len(relations) < 2:
            continue
        if len(relations) == len(card):
            continue
        card_prod = np.prod(np.array(card)[relations])
        pred_prod = 1
        for p in range(len(pred)):
            (r1, r2) = pred[p]
            if r1 in relations and r2 in relations:
                pred_prod = pred_prod * pred_sel[p]
        intermediate_cost = card_prod * pred_prod
        cost = cost + intermediate_cost
    return cost


# In[ ]:


def get_raw_join_order(cost_vector):
    join_order = np.argsort(cost_vector).tolist()
    join_order.reverse()
    return join_order

def postprocess_join_order(raw_join_order, cost_vector, num_relations, pred):
    join_order = [raw_join_order[0]]
    while len(join_order) < num_relations:
        
        applicable_predicates = [pred_tuple for t in join_order for pred_tuple in pred if t in pred_tuple]
        neighborhood_indices = [t for t in set(sum(applicable_predicates, ())) if t not in join_order]
        
        if len(neighborhood_indices) != 0:
            best_neighbor_relation = neighborhood_indices[np.argmax(cost_vector[neighborhood_indices])]
            join_order.append(best_neighbor_relation)
        else:
            global_indices = [x for x in raw_join_order if x not in join_order]
            best_global_relation = global_indices[np.argmax(cost_vector[global_indices])]
            join_order.append(best_global_relation)
    return join_order
   
def readout(response, card, pred, pred_sel, card_dict):
    start = time.time()
    bitstrings = []
    for solution in response[0]:
        # for i in range(int(solution[1])):
        bit = solution[0]
        energy = solution[2]
        occ=solution[1]
        stringbit=solution[3]
        probability=solution[4]
        for _ in range(occ):
            bitstrings.append((bit,energy,stringbit,occ,probability))
            # bitstrings.append(solution[0])
            
            
    weight_vector = np.arange(1, len(card)-1)
    weight_vector = weight_vector[len(card)-3::-1]
    
    best_solutions_for_time = []
    best_costs = inf
        
    solutions = []
    
    num_relations = len(card)
    
    for i in range(len(bitstrings)):
        bitstring,energy,stringbit,occ,probability = bitstrings[i]
        
        bitstring = bitstring[:len(card)*(len(card)-2)]
        partial_bitstrings = np.array_split(bitstring, len(card))
        cost_vector = np.array(partial_bitstrings).dot(weight_vector)
        
        raw_join_order = get_raw_join_order(cost_vector)
        #print("Raw:")
        #print(raw_join_order)
        
        costs = get_costs_for_leftdeep_tree(raw_join_order, card, pred, pred_sel, card_dict)
        
        solution = [stringbit,raw_join_order, int(costs), (time.time()-start)*1000, False,energy,occ,probability]
        solutions.append(solution)
        if costs < best_costs:
            best_costs = costs
            best_solutions_for_time.append(solution)
            
        # Fallback
        join_order = postprocess_join_order(raw_join_order, cost_vector, num_relations, pred)
        costs = get_costs_for_leftdeep_tree(join_order, card, pred, pred_sel, card_dict)
        solution = [stringbit,join_order, int(costs), (time.time()-start)*1000, True,energy,occ,probability]
        solutions.append(solution)
        if costs < best_costs:
            best_costs = costs
            best_solutions_for_time.append(solution)
        #print("Processed:")
        #print(join_order)
    
    return best_solutions_for_time, solutions

