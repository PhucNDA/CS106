from lib2to3.pytree import convert
import sys
import os
import random
from itertools import islice
import timeit
import time
import pandas as pd

def Get_All_TestCases():
    # n=50,100,200,500,1000
    # R=1000
    #Choose these testcase
    temp_name1=['n00050','n00100','n00200','n00500','n01000']
    temp_name2=['R01000']
    path='.\\Dataset\\kplib\\'
    temp_dir=os.listdir(r'.\Dataset\kplib')
    temp_dir.sort()
    name_dir=[]
    for i in range(1,14):
        name_dir.append(temp_dir[i])
    All_TestCase=[]
    for name in name_dir:
        for name1 in temp_name1:
            for name2 in temp_name2:
                temp_name3=os.listdir(path+name+'\\'+name1+'\\'+name2)
                All_TestCase.append(name+'\\'+name1+'\\'+name2+'\\'+temp_name3[5])#Choose the fifth test case
    return All_TestCase