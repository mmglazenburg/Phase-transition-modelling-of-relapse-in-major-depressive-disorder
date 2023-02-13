# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 15:58:37 2021

@author: Marieke
"""
import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
from random import randint

        
#%% Analysis functions

def checkFluctuations(ts):                  # checks for fluctuations in specific experimental phases

    phases = [29, 43, 99, 156]              # start of phase days
    transition = [127,141]
    
    baseline = ts[:phases[1]]
    
    base_var = baseline.var()
    
    phase3 = ts[phases[1]:phases[2]]
    phase3_var = phase3.var()
    
    phase4_bf = ts[phases[2]:transition[1]]
    phase4_bf_var = phase4_bf.var()
    
    phase4_af = ts[transition[1]:]
    phase4_af_var = phase4_af.var()
    
    means = [baseline.mean(), phase3.mean(), phase4_bf.mean(), phase4_af.mean()]
    varia = [base_var, phase3_var, phase4_bf_var, phase4_af_var]

    
    return varia, means



#%% Averaging function

def plotItemAvg(item_name,xaxis = 'ms',label=None,span=14,plot_item=False,return_data = False):

    data_full = pd.read_csv("D:/Documenten/Universiteit/SiS/Afstudeerproject/Data/ESMdata/ESMdata.csv")
    data_cut = data_full.dropna(subset=[item_name])
    
    
    vert_shift = ['mood_down',
              'mood_lonely',
              'mood_anxious',
              'mood_guilty',
              'event_pleas',
              'event_import']
    
   
    df = data_cut.loc[:,('dayno',item_name)]
    df['dayno'] = df['dayno'] - 225                     # set first day to 1
    df.loc[df['dayno']<0,'dayno'] += 366                # cycle day numbers
    if item_name in vert_shift:                         # correct for 0-centred scale
        df[item_name] += 4

    df_mean = df.groupby('dayno',as_index=False).mean()      # average duplicates on day
            
    item_ewm = df_mean[item_name].ewm(halflife=span,min_periods=14).mean()

    print(len(item_ewm))

    fluctuations = df_mean[item_name] - item_ewm           # obtain detrended fluctuations

    varia, means = checkFluctuations(fluctuations)
    varia = [round(num,3) for num in varia]
    
    if (varia[0]<varia[1]) & (varia[0]<varia[2]) & (varia[2]>varia[3]):
        check_fluct = True
        label = '** Variance increase before transition**\n'
    else:
        check_fluct = False
        label = ''
     
    label = label + 'Baseline: ' + str(varia[0]) + '\nPhase 3: ' + str(varia[1]) + '\nPhase 4 before/during transition: ' + str(varia[2]) + '\nAfter transition: ' + str(varia[3])
    

    varia_avg, means_avg = checkFluctuations(item_ewm)
    sd_avg = [math.sqrt(num) for num in varia_avg]
    sd_avg = [round(num,3) for num in sd_avg]
    means_avg = [round(num,3) for num in means_avg]
    
    
    if (means_avg[0] + sd_avg[0]) < (means_avg[3] - sd_avg[3]) or (means_avg[0] - sd_avg[0]) > (means_avg[3] + sd_avg[3]):
        # consider both positive and negative shifts
        check_shift = True
        label_avg = '**Shift over transition** \n'
    else:
        label_avg = ''
        check_shift = False
    label_avg = label_avg + 'Baseline: ' + str(means_avg[0]) + '+/- ' + str(sd_avg[0]) + '\nAfter transition: ' + str(means_avg[3]) + '+/- ' + str(sd_avg[3])


           
    ax1 = plt.gca()
    ax1.plot(df_mean['dayno'],item_ewm,label=label,c='r',alpha=0.9,lw=2)        # plot ewm
    
    if plot_item == True:
        plt.plot(df_mean['dayno'],df_mean[item_name],alpha=0.5)
    
    
    plt.xlim(0,240)
    plt.xlabel('Experiment day')
    ax1.set_ylabel('%s Score' % item_name)
    
    min_item, max_item = min(df_mean[item_name]), max(df_mean[item_name])
    # ax1.set_ylim(min_item,max_item)
        
    plt.axvspan(127, 141, color='y', alpha=0.3, lw=0)
    phases = [[29, 43, 99, 156], ['[2]', '[3]', '[4]', '[5]']]          # indicate experiment phases
    plt.vlines(phases[0],plt.ylim()[0],plt.ylim()[1],colors='k',linestyles='dotted',alpha=0.5)    
         
    
    plt.plot(item_ewm,linestyle=' ',label=label_avg)
    
    plt.legend(loc='upper left')
    
    ax1.plot(df_mean['dayno'],fluctuations)

    if return_data == True:
        d = {'dayno': df_mean['dayno'], 'data_raw': df_mean[item_name], 'average': item_ewm}
        # return [check_fluct,check_shift]
        return d


#%% All momentary items; fluctuation and level shift analysis; check both

item_names = [
    ('mood_relaxed'),
    ('mood_down'),
    ('mood_irritat'),
    ('mood_satisfi'),
    ('mood_lonely'),
    ('mood_anxious'),
    ('mood_enthus'),
    ('mood_suspic'),
    ('mood_cheerf'),
    ('mood_guilty'),
    ('mood_doubt'),
    ('mood_strong'),
    ('pat_restl'),
    ('pat_agitate'),
    ('pat_worry'),
    ('pat_concent'),
    ('se_selflike'),
    ('se_ashamed'),
    ('se_selfdoub'),
    ('se_handle'),
    ('phy_hungry'),
    ('phy_tired'),
    ('phy_pain'),
    ('phy_dizzy'),
    ('phy_drymouth'),
    ('phy_nauseous'),
    ('phy_headache'),
    ('phy_sleepy'),
    ('phy_physact'),
    ('evn_ordinary'),
    ('evn_niceday'),
    ('mor_qualsleep'),
    ('mor_feellike')
    ]



fig = plt.figure(figsize=(25,len(item_names)*8.5))
op_items = []

print('Progress:\n')

for i in range(len(item_names)):
    plt.subplot(len(item_names),1,i+1)
    
    results = plotItemAvg(item_names[i],span=14,plot_item=True,return_data=True)
    
    if results[0]==True & results[1]==True:
        op_items.append(item_names[i])
    
    print(i+1,'/',len(item_names))
    
    
#%% Both criteria checked

item_names = op_items

fig = plt.figure(figsize=(25,len(item_names)*8.5))


for i in range(len(item_names)):
    plt.subplot(len(item_names),1,i+1)
    
    results = plotItemAvg(item_names[i],plot_item=False,return_data=False)
    



#%% Check relative shifts and fluctuation increase

def checkShifts(item_name,label=None,span=14,plot_item=False,return_data = False):

    data_full = pd.read_csv("D:/Documenten/Universiteit/SiS/Afstudeerproject/Data/ESMdata/ESMdata.csv")
    data_cut = data_full.dropna(subset=[item_name])
    
    
    vert_shift = ['mood_down',
              'mood_lonely',
              'mood_anxious',
              'mood_guilty',
              'event_pleas',
              'event_import']
    
   
    df = data_cut.loc[:,('dayno',item_name)]
    df['dayno'] = df['dayno'] - 225                     # set first day to 1
    df.loc[df['dayno']<0,'dayno'] += 366                # cycle day numbers
    if item_name in vert_shift:                         # correct for 0-centred scale
        df[item_name] += 4

    df_mean = df.groupby('dayno',as_index=False).mean()      # average duplicates on day
            
    item_ewm = df_mean[item_name].ewm(halflife=span,min_periods=14).mean()

    fluctuations = df_mean[item_name] - item_ewm           # obtain detrended fluctuations


    varia, means = checkFluctuations(fluctuations)
    varia = [round(num,3) for num in varia]
    
    
    if (varia[0]<varia[1]) & (varia[0]<varia[2]) & (varia[2]>varia[3]):
        check_fluct = True
        fluct = (varia[2] - np.mean([varia[0],varia[3]])) / np.mean([varia[0],varia[3]]) * 100   # fluctuation increase percentage relative to average before and after
        if item_name == 'phy_nauseous':
            print(varia,fluct)
    else:
        check_fluct = False
        fluct = 0  
    
    varia_avg, means_avg = checkFluctuations(item_ewm)
    sd_avg = [math.sqrt(num) for num in varia_avg]
    sd_avg = [round(num,3) for num in sd_avg]
    means_avg = [round(num,3) for num in means_avg]
        
    
    if (means_avg[0] + sd_avg[0]) < (means_avg[3] - sd_avg[3]) or (means_avg[0] - sd_avg[0]) > (means_avg[3] + sd_avg[3]):
        # consider both positive and negative shifts
        check_shift = True
        shift = abs((means_avg[3] - means_avg[0])/means_avg[0]*100)     # shift percentage relative to baseline
    else:
        check_shift = False
        shift = 0

         
    return fluct, shift
    

#%% Apply function

item_names = [
    ('mood_relaxed'),
    ('mood_down'),
    ('mood_irritat'),
    ('mood_satisfi'),
    ('mood_lonely'),
    ('mood_anxious'),
    ('mood_enthus'),
    ('mood_suspic'),
    ('mood_cheerf'),
    ('mood_guilty'),
    ('mood_doubt'),
    ('mood_strong'),
    ('pat_restl'),
    ('pat_agitate'),
    ('pat_worry'),
    ('pat_concent'),
    ('se_selflike'),
    ('se_ashamed'),
    ('se_selfdoub'),
    ('se_handle'),
    ('phy_hungry'),
    ('phy_tired'),
    ('phy_pain'),
    ('phy_dizzy'),
    ('phy_drymouth'),
    ('phy_nauseous'),
    ('phy_headache'),
    ('phy_sleepy'),
    ('phy_physact'),
    ('evn_ordinary'),
    ('evn_niceday'),
    ('mor_qualsleep'),
    ('mor_feellike')
    ]

# item_names = [
#     ('mood_down'),
#     ('mood_lonely'),
#     ('mood_anxious'),
#     ('mood_guilty'),
#     ('pat_restl'),
#     ('pat_agitate'),
#     ('mood_doubt')
#     ]



spans = np.arange(10,80,10)
column_names = [str(num) for num in spans]

shifts = []
flucts = []

for span in spans:
    column_shift = []
    column_fluct = []    

    for j in range(len(item_names)):
        fluct, shift = checkShifts(item_names[j],span=span,plot_item=False,return_data=True)
        column_shift.append(shift)
        column_fluct.append(fluct)
        
    shifts.append(column_shift)  
    flucts.append(column_fluct)
    
    print(span)
    
shifts = np.array(shifts).transpose()
flucts = np.array(flucts).transpose()

fig, [ax1,ax2] = plt.subplots(2,figsize=(13,13))

colors = []
# colors = ['r','b','green','orange','purple','lightblue','darkgreen']

for i in range(len(item_names)): 
    colors.append('#%06X' % randint(0, 0xFFFFFF))       #create random spaced colors
    shifts_true = np.where(shifts[i]!=0)
    flucts_true = np.where(flucts[i]!=0)
    
    
    if np.isin(shifts_true, flucts_true).any():      #plot the ones that have both fluct and shift for at least 1 halflife
        
        ax1.plot(spans,shifts[i],label=item_names[i],c=colors[i])
        ax1.set_ylabel('Relative shift (%)')
        ax1.set_xlabel('EWMA halflife (days)')
        
        ax2.plot(spans,flucts[i],label=item_names[i],c=colors[i])
        ax2.set_ylabel('Relative variance increase (%)')
        ax2.set_xlabel('EWMA halflife (days)')

            
ax1.set_ylim(0,ax1.set_ylim()[1])
ax2.set_ylim(0,ax2.set_ylim()[1])

ax1.legend(loc='upper right')
ax2.legend(loc='upper right')
