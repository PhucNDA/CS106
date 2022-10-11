from lib2to3.pytree import convert
import sys
import os
import random
from itertools import islice
import timeit
import time
import pandas as pd
import TestCase_Generator as TCG

# inp = [a.strip() for a in sys.stdin.readlines()] 
# n,W=[int(x) for x in inp[0].split()]
# weight=[int(x) for x in inp[1].split()]
# value=[int(x) for x in inp[2].split()]
# weight.insert(0,0)
# value.insert(0,0)

def Basic_Knapsack(n,capacitites,weights,values):
    d=[[-1 for x in range(capacitites+1)] for y in range(n+1)] 
    
    #d[i][j]: maximum value choosing from subset 1->i with the weight of j
    #d[i][j] = max(d[i-1][j],d[i-1][j-weights[i]]+values[i])
    #d[i][j]=-1: initial state
    #d[i][0]=0: base state

    for i in range(0,n+1):
        d[i][0]=0
    for i in range(1,n+1):
        for j in range(1,capacitites+1):
            d[i][j]=d[i-1][j]
            if(j-weights[i]>=0 and d[i-1][j-weights[i]]!=-1):
                d[i][j]=max(d[i][j],d[i-1][j-weights[i]]+values[i])
    result=[0,0]
    for j in range (1,capacitites+1):
        if(d[n][j]!=-1 and d[n][j]>=result[0]):
            result=[d[n][j],j]
    return result

def Optimized_Knapsack(n,capacitites,weights,values):
    d=[[-1 for x in range(capacitites+1)] for y in range(2)] 

    #Optimized the dimension of the matrix
    #We only use 2 row of state so it's possible solving it with matrix (2xW)
    for i in range(0,2):
        d[i][0]=0
    for i in range(1,n+1):
        for j in range(1,capacitites+1):
            d[1][j]=d[0][j]
            if(j-weights[i]>=0 and d[0][j-weights[i]]!=-1):
                d[1][j]=max(d[1][j],d[0][j-weights[i]]+values[i])
        d[0]=d[1]
        d[1]=[-1 for x in range(0,capacitites+1)]
        d[1][0]=0
    result=[0,0]
    for j in range (1,capacitites+1):
        if(d[0][j]!=-1 and d[0][j]>=result[0]):
            result=[d[0][j],j]
    return result

def Solve_DynamicProgramming(name):
    fin = open(name)
    lines=fin.read().splitlines()
    
    n=int(lines[1])
    capacities=int(lines[2])
    values=[]
    weights=[]
    for line in islice(lines, 4, None):
        data = line.split()
        values.append(int(data[0]))
        weights.append(int(data[1]))
    #Optimize based on mathematical observation    
    total_weight=0
    for weight in weights:
        total_weight=total_weight+weight
    capacities=min(capacities,total_weight)
    #Convert to ordinary DP style
    weights.insert(0,0)
    values.insert(0,0)
    if(n*capacities>1000000000):
        return [-1,-1] 
    return Optimized_Knapsack(n,capacities,weights,values)

def Solve_All_Testcases_Optimal():
    #Get all the available test cases
    TestCases=TCG.Get_All_TestCases()
    file_test_path='.\\Dataset\\kplib\\'
    data=[]
    #For each testcase, solve the Knapsack problem
    for name in TestCases:
        file_name=file_test_path+name
        print('Knapsack: ',file_name)
        start=time.time()
        optimal_answer=Solve_DynamicProgramming(file_name)
        end=time.time()
        print('Done.')
        tmp=[]
        for e in name:
            if(e=='\\'):
                tmp.append('_')
            else:
                tmp.append(e)
        data.append(["".join(tmp),optimal_answer[0],optimal_answer[1],end-start])
    return data