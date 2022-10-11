import Knapsack_optimal_answer_generator as koag
import GoogleORTools_BranchandBounds as gotb
import time
import pandas as pd
import TestCase_Generator as TCG

def Excecute():
    columns=['Case','Total_Value_Optimal','Total_Weight_Optimal','Runtime_Optimal','Total_Value_gORt','Total_Weight_gORt','Runtime_gORt']
    knapsack=koag.Solve_All_Testcases_Optimal()
    branchandbounds=gotb.Solve_All_Testcases_BranchandBounds()
    #Results are lists of information
    num_case=len(knapsack)
    data=[]
    #Concanating corresponding lists
    for row in range(num_case-1):
        data.append(knapsack[row]+branchandbounds[row])
    print(data)
    df=pd.DataFrame(data,columns=columns)
    #Result to csv
    df.to_csv('Result.csv')

if __name__ == '__main__':
    Excecute()
