from ortools.algorithms import pywrapknapsack_solver
import os
import random
from itertools import islice
import timeit
import TestCase_Generator as TCG
import time
import pandas as pd

def BranchandBounds(n,capacities,weights,values):
    solver = pywrapknapsack_solver.KnapsackSolver(
      pywrapknapsack_solver.KnapsackSolver.
      KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER, 'KnapsackExample')

    solver.Init(values, weights, capacities)
    solver.set_time_limit(120.0)
    computed_value = solver.Solve()

    total_weight = 0
    for i in range(len(values)):
        if solver.BestSolutionContains(i):
            total_weight += weights[0][i]
    return [computed_value,total_weight]
    

def Solve_BranchandBounds(name):
    fin = open(name)
    lines=fin.read().splitlines()
    
    n=int(lines[1])
    capacities=[]
    values=[]
    weights=[]
    tmp=[]
    total_cap=int(lines[2])
    total_weight=0
    #Mathematical observation
    for line in islice(lines, 4, None):
        data = line.split()
        values.append(int(data[0]))
        tmp.append(int(data[1]))
        total_weight=total_weight+int(data[1])
    weights.append(tmp)
    total_cap=min(total_cap,total_weight)
    capacities.append(total_cap)
    return BranchandBounds(n,capacities,weights,values)


def Solve_All_Testcases_BranchandBounds():
    #Get all available testcases
    TestCases=TCG.Get_All_TestCases()
    file_test_path='.\\Dataset\\kplib\\'
    data=[]
    #For each testcase, solve the problem
    for name in TestCases:
        file_name=file_test_path+name
        print('BaB: ',file_name)
        start=time.time()
        optimal_answer=Solve_BranchandBounds(file_name)
        end=time.time()
        print('Done.')
        tmp=[]
        for e in name:
            if(e=='\\'):
                tmp.append('_')
            else:
                tmp.append(e)
        data.append([optimal_answer[0],optimal_answer[1],end-start])
    return data
