# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 15:07:25 2021

@author: Marieke
"""

import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
import matplotlib as mpl
from matplotlib.patches import ConnectionPatch
import matplotlib.gridspec as gridspec

tw = 6.0                                # LaTeX textwidth
item_height = 2.3                       # height of singular item plot

mpl.rcParams['lines.linewidth'] = 1     # linewidth of line plots
mpl.rcParams['axes.linewidth'] = 0.5    # linewidth of axes

axes_color = 'dimgrey'
mpl.rcParams['axes.edgecolor'] = axes_color
mpl.rcParams['xtick.color'] = axes_color
mpl.rcParams['xtick.labelcolor'] = 'black'
mpl.rcParams['ytick.color'] = axes_color
mpl.rcParams['ytick.labelcolor'] = 'black'

#%% Item examples figure

def plotItem(item_name,xaxis = 'ms',label=None,overrideAxis=None):

    if overrideAxis != None:            # allow to plot all without preferred x-axis
        xaxis = overrideAxis

   
    data_full = pd.read_csv("H:/Nijmegen/PLOS One/ESMdata.csv")
    data_cut = data_full.dropna(subset=[item_name])
    
    item = data_cut[item_name]
    
    vert_shift = ['mood_down',
              'mood_lonely',
              'mood_anxious',
              'mood_guilty',
              'event_pleas',
              'event_import']
    
    if item_name in vert_shift:                                 # correct for 0-centerend scale
        item += 4
    
    
    measno = len(data_full.loc[data_full['resp_abort']==0])     # lenght of full measurement 
    
    
    if item_name[:3] == 'SCL' or item_name[:3] == 'dep':              # set different ylim for SCL scores
        plt.ylim(0.75,4.25)
    else:
        plt.ylim(0.5,7.5)
    
    if xaxis == 'days':                                     # plot as a function of days
    
        df = data_cut.loc[:,('dayno',item_name)]
        df['dayno'] = df['dayno'] - 225                     # set first day to 1
        df.loc[df['dayno']<0,'dayno'] += 366                # cycle day numbers

        df = df.groupby('dayno',as_index=False).mean()      # average duplicates on day
        
        if len(item) < measno:
            plt.step(df['dayno'],df[item_name],where='post',label=label)
        else:
            plt.plot(df['dayno'],df[item_name],label=label)
            
            
        plt.axvspan(127, 141, color='y', alpha=0.3, lw=0)
        phases = [[29, 43, 99, 156], ['[2]', '[3]', '[4]', '[5]']]          # indicate experiment phases
        plt.vlines(phases[0],plt.ylim()[0],plt.ylim()[1],colors='k',linestyles='dotted',alpha=0.3)
        
        
        plt.xlim(0,240)
        plt.xlabel('Experiment day')
        plt.ylabel('%s' % label)
    
    elif xaxis == 'ms':                         # plot as a function of measurements
        if len(item) < measno:
            plt.step(item.index+1,item,where='post',label=label)           # make symptom score step plot
        else:
            plt.plot(item,label=label)
            
        plt.xlim(0,1477)
            
        
        phases = [[178, 288, 673, 991], ['[2]', '[3]', '[4]', '[5]']]          # indicate experiment phases
        plt.vlines(phases[0],plt.ylim()[0],plt.ylim()[1],colors='k',linestyles='dotted',alpha=0.3)
        plt.axvspan(824, 903, color='y', alpha=0.3, lw=0)
        
        plt.xlabel('Measurement number')
        plt.ylabel('%s' % label)
        
    elif xaxis == 'con':                         # plot as a function of antidepressant concentration
        concentrat = data_cut['concentrat']
        plt.plot(concentrat,item,label=label)
        
        plt.xlabel('Concentration')
        plt.ylabel('%s' % label)
        
    # if label != None:
    #     plt.legend(loc='upper left')
        
    
item_names = [
    ('dep','days','Average symptom score'),
    ('SCL.90.R.31','days','I worry too much'),
    ('mor_qualsleep','days','I slept well'),
    ('mood_down','ms','I feel down'),
    ]


fig = plt.figure(figsize=(tw,len(item_names)*item_height/1.1))

labels = ['a','b','c','d']


for i in range(len(item_names)):
    plt.subplot(len(item_names),1,i+1)
    
    phases = [29, 43, 99, 156]
    if i == 0:
        plt.vlines(phases[0],plt.ylim()[0],plt.ylim()[1],colors='k',linestyles='dotted',alpha=0.3)
        
    
    plotItem(*item_names[i])
    
    ax = plt.gca()
    ax.set_title(labels[i],x=-0.035,fontsize=15,fontweight='bold')

# very ugly, but it works in LaTeX. '~' needed for spacing.
plt.suptitle('Phase start\n1~~~~~~~~~~~~~~~2~~~~~~3~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~4~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~5',ha='left',x=0.075,y=0.952,fontsize=8,alpha=0.8)


fig.tight_layout(pad=0.5)
fig.savefig("H:/Nijmegen/PLOS One/Figures_test/item_examples.pgf",bbox_inches='tight')


#%% Item EWMA figure

def plotItemAvg(item_name,label=None,span=14,plot_item=False,return_data = False):

    data_full = pd.read_csv("H:/Nijmegen/PLOS One/ESMdata.csv")
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
    
            
    item_ewm = df_mean[item_name].ewm(halflife=span,min_periods=span).mean()

           
    ax1 = plt.gca()
    ax1.plot(df_mean['dayno'],item_ewm,label=label,c='r',alpha=0.9,lw=1.5)        # plot ewm
    
    plt.plot(df_mean['dayno'],df_mean[item_name],alpha=0.5)
    
    
    plt.xlim(0,240)
    plt.xlabel('Experiment day')
    ax1.set_ylabel('%s' % label)
        
    plt.axvspan(127, 141, color='y', alpha=0.3, lw=0)
    phases = [[29, 43, 99, 156], ['[2]', '[3]', '[4]', '[5]']]          # indicate experiment phases
    plt.vlines(phases[0],ax1.set_ylim()[0],ax1.set_ylim()[1],colors='k',linestyles='dotted',alpha=0.3)    
         

    
item_names = [
    ('mood_down','I feel down'),
    ]


fig = plt.figure(figsize=(tw,len(item_names)*item_height))

labels = ['a','b','c','d']


for i in range(len(item_names)):
    plt.subplot(len(item_names),1,i+1)
    
    plotItemAvg(*item_names[i])
    

fig.tight_layout(pad=0.5)
fig.savefig("H:/Nijmegen/PLOS One/Figures_test/item_ewma.pgf",bbox_inches='tight')



#%% Item EWMA shift figure

def checkFluctuations(ts):                  # checks for fluctuations in specific experimental phases
    if len(ts)>238:
        phases = [178, 288, 673, 991]              # start of phase days
        transition = [824,903]
    else:
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
    vars = [base_var, phase3_var, phase4_bf_var, phase4_af_var]
    
    return vars, means

def plotItemAvg(item_name,label=None,span=14,plot_item=False,return_data = False):

    data_full = pd.read_csv("H:/Nijmegen/PLOS One/ESMdata.csv")
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
    
            
    item_ewm = df_mean[item_name].ewm(halflife=span,min_periods=span).mean()


    vars_avg, means_avg = checkFluctuations(item_ewm)
    
    sd_avg = [math.sqrt(num) for num in vars_avg]
    sd_avg = [round(num,3) for num in sd_avg]
    means_avg = [round(num,2) for num in means_avg]
    
    
    if (means_avg[0] + sd_avg[0]) < (means_avg[3] - sd_avg[3]) or (means_avg[0] - sd_avg[0]) > (means_avg[3] + sd_avg[3]):
        plt.plot([span,43],[means_avg[0],means_avg[0]],c='grey',lw=0.8)            #plot levels
        plt.plot([141,len(item_ewm)],[means_avg[3],means_avg[3]],c='grey',lw=0.8)

        plt.axhspan(means_avg[0]-sd_avg[0],means_avg[0]+sd_avg[0],xmin=0,xmax=139.5/238,color='grey',alpha=0.2)
        plt.axhspan(means_avg[3]-sd_avg[3],means_avg[3]+sd_avg[3],xmin=126.25/238,xmax=1,color='grey',alpha=0.2)
           
    ax1 = plt.gca()
    ax1.plot(df_mean['dayno'],item_ewm,label=label,c='r',alpha=0.9,lw=1.5)        # plot ewm
    
    
    plt.xlim(0,240)
    plt.xlabel('Experiment day')
    ax1.set_ylabel('%s' % label)
        
    plt.axvspan(127, 141, color='y', alpha=0.3, lw=0)
    phases = [[29, 43, 99, 156], ['[2]', '[3]', '[4]', '[5]']]          # indicate experiment phases
    plt.vlines(phases[0],ax1.set_ylim()[0],ax1.set_ylim()[1],colors='k',linestyles='dotted',alpha=0.3)    
         

    
item_names = [
    ('mood_down','I feel down (EWMA)'),
    ]


fig = plt.figure(figsize=(tw,len(item_names)*item_height))

labels = ['a','b','c','d']


for i in range(len(item_names)):
    plt.subplot(len(item_names),1,i+1)
    
    plotItemAvg(*item_names[i])
    

fig.tight_layout(pad=0.5)
fig.savefig("H:/Nijmegen/PLOS One/Figures_test/item_ewmashift.pgf",bbox_inches='tight')


#%% Item fluctuations figure
def checkFluctuations(ts):                  # checks for fluctuations in specific experimental phases
    if len(ts)>238:
        phases = [178, 288, 673, 991]              # start of phase days
        transition = [824,903]
    else:
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
    vars = [base_var, phase3_var, phase4_bf_var, phase4_af_var]
    
    return vars, means

def plotItemAvg(item_name,label=None,span=14,plot_item=False,return_data = False):

    data_full = pd.read_csv("H:/Nijmegen/PLOS One/ESMdata.csv")
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
    
            
    item_ewm = df_mean[item_name].ewm(halflife=span,min_periods=span).mean()


    fluctuations = df_mean[item_name] - item_ewm           # obtain detrended fluctuations


    vars, means = checkFluctuations(fluctuations)
    vars = [round(num,3) for num in vars]

  
    ax1 = plt.gca()
    ax1.plot(df_mean['dayno'],fluctuations)
    
    plt.xlim(0,240)
    plt.xlabel('Experiment day')
    ax1.set_ylabel('%s' % label)
        

  
    # variance regions
    alpha = 0.35
    color = 'red'
    plt.axvspan(0,43,color=color,alpha=vars[0]/vars[2]*alpha,lw=0)
    plt.axvspan(43,99,color=color,alpha=vars[1]/vars[2]*alpha,lw=0)
    plt.axvspan(99,141,color=color,alpha=alpha,lw=0)
    plt.axvspan(141,240,color=color,alpha=vars[3]/vars[2]*alpha,lw=0)

    # indicate transition
    # plt.axvspan(127, 141, color='grey', alpha=0.3, lw=0)
    # plt.vlines([127,141],ax1.set_ylim()[0],ax1.set_ylim()[1],colors='k',linestyles='dashed',alpha=0.5)    
        

    
item_names = [
    ('mood_down',"I feel down (detrended)"),
    ]

fig = plt.figure(figsize=(tw,len(item_names)*item_height))

labels = ['a','b','c','d']


for i in range(len(item_names)):
    plt.subplot(len(item_names),1,i+1)
    
    plotItemAvg(*item_names[i])
    

fig.tight_layout(pad=0.5)

fig.savefig("H:/Nijmegen/PLOS One/Figures_test/item_fluct.pgf",bbox_inches='tight')


#%% Half life sensitivity

def checkFluctuations(ts):                  # checks for fluctuations in specific experimental phases
    if len(ts)>238:
        phases = [178, 288, 673, 991]              # start of phase days
        transition = [824,903]
    else:
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
    vars = [base_var, phase3_var, phase4_bf_var, phase4_af_var]
    
    return vars, means


def checkShifts(item_name,label=None,span=14,plot_item=False,return_data = False):

    data_full = pd.read_csv("H:/Nijmegen/PLOS One/ESMdata.csv")
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
    
    
    if (varia[0]<varia[1]) & (varia[1]<varia[2]) & (varia[2]>varia[3]):
        check_fluct = True
        fluct = (varia[2] - np.mean([varia[0],varia[3]])) / np.mean([varia[0],varia[3]]) * 100   # fluctuation increase percentage relative to average before and after

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
    


item_names = [
    ('mood_down','I feel down'),
    ('mood_lonely','I feel lonely'),
    ('mood_anxious','I feel anxious'),
    ('mood_guilty','I feel guilty'),
    ('pat_restl','I feel restless'),
    ('pat_agitate','I feel agitated'),
    ]



spans = np.arange(5,85,5)
column_names = [str(num) for num in spans]

shifts = []
flucts = []

for span in spans:
    column_shift = []
    column_fluct = []    

    for j in range(len(item_names)):
        fluct, shift = checkShifts(item_names[j][0],span=span,plot_item=False,return_data=True)
        column_shift.append(shift)
        column_fluct.append(fluct)
        
    shifts.append(column_shift)  
    flucts.append(column_fluct)

    # print(span)
    
shifts = np.array(shifts).transpose()
flucts = np.array(flucts).transpose()


fig1 = plt.figure(figsize=(tw,7))
ax1 = fig1.add_subplot(2,2,1)
ax2 = fig1.add_subplot(2,2,2)
ax3 = fig1.add_subplot(2,1,2)

#colors = ['r','darkblue','green','orange','darkmagenta','skyblue']
#colors = ['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e','#e6ab02']
colors = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c']
markers = ['x','^','o','v','s','D']

# colors = sns.color_palette()


for i in range(len(item_names)): 
          
    ax1.plot(spans,shifts[i],label=item_names[i][1],c=colors[i],marker=markers[i],ms=3)
    ax1.set_ylabel('Relative shift (%)')
    ax1.set_xlabel('EWMA half life (days)')
    
    #if i < 4:
    ax2.plot(spans,shifts[i],label=item_names[i][1],c=colors[i],marker=markers[i],ms=3)
    ax2.set_ylabel('Relative shift (%)')
    ax2.set_xlabel('EWMA half life (days)')      
        
    ax3.plot(spans,flucts[i],label=item_names[i][1],c=colors[i],marker=markers[i],ms=3)
    ax3.set_ylabel('Relative variance increase (%)')
    ax3.set_xlabel('EWMA half life (days)')



            
ax1.set_ylim(0,ax1.set_ylim()[1])
ax2.set_ylim(0,10.5)
ax3.set_ylim(0,ax3.set_ylim()[1])

# ax1.legend(loc='upper right')
ax2.legend(loc='upper right')
# ax3.legend(loc='upper right')

ax1.set_title('a',x=-0.08,y=1.01,fontsize=15,fontweight='bold')
ax2.set_title('b',x=-0.08,y=1.01,fontsize=15,fontweight='bold')
ax3.set_title('c',x=-0.04,y=1.01,fontsize=15,fontweight='bold')


zoomline1 = ConnectionPatch(xyA=(0,0.2), xyB=(0,1), coordsA="axes fraction", coordsB="axes fraction",
                      axesA=ax1, axesB=ax2, color="k", ls='dashed',alpha=0.3)
zoomline2 = ConnectionPatch(xyA=(0,0), xyB=(0,0), coordsA="axes fraction", coordsB="axes fraction",
                      axesA=ax1, axesB=ax2, color="k", ls='dashed',alpha=0.3)
ax1.add_artist(zoomline1)
ax1.add_artist(zoomline2)


fig1.subplots_adjust(hspace=0.1, wspace=0.06)

fig1.tight_layout(pad=0.2)
#fig1.savefig('D:/Documenten/Universiteit/SiS/Afstudeerproject/Thesis/Figures/hl_sensitivity.pgf',bbox_inches='tight')
fig1.savefig('H:/Nijmegen/PLOS One/Figures_test/hl_sensitivity.png',bbox_inches='tight')


#%% Compare 'Agitated' to 'Lonely' figure

def plotItemAvg(item_name,label=None,span=14,plot_item=False,return_data = False):

    data_full = pd.read_csv("H:/Nijmegen/PLOS One/ESMdata.csv")
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
    
            
    item_ewm = df_mean[item_name].ewm(halflife=span,min_periods=span).mean()

           
    ax1 = plt.gca()
    ax1.plot(df_mean['dayno'],item_ewm,label=label,c='r',alpha=0.9,lw=1.5)        # plot ewm
    
    plt.plot(df_mean['dayno'],df_mean[item_name],alpha=0.5)
    
    
    plt.xlim(0,240)
    plt.xlabel('Experiment day')
    ax1.set_ylabel('%s' % label)
        
    plt.axvspan(127, 141, color='y', alpha=0.3, lw=0)
    phases = [[29, 43, 99, 156], ['[2]', '[3]', '[4]', '[5]']]          # indicate experiment phases
    plt.vlines(phases[0],ax1.set_ylim()[0],ax1.set_ylim()[1],colors='k',linestyles='dotted',alpha=0.3)    
         

    
item_names = [
    ('mood_lonely','I feel lonely'),
    ('pat_agitate','I feel agitated')
    ]


fig = plt.figure(figsize=(tw,len(item_names)*item_height))

labels = ['a','b','c','d']


for i in range(len(item_names)):
    plt.subplot(len(item_names),1,i+1)
    
    ax = plt.gca()
    ax.set_title(labels[i],x=-0.04,fontsize=15,fontweight='bold')
    
    plotItemAvg(*item_names[i])
    

fig.tight_layout(pad=0.5)
fig.savefig("H:/Nijmegen/PLOS One/Figures_test/item_diffs.pgf",bbox_inches='tight')



#%% All items appendix figure

def plotItemAvg(item_name,label=None,span=14,plot_item=False,return_data = False):

    data_full = pd.read_csv("H:/Nijmegen/PLOS One/ESMdata.csv")
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
    
            
    item_ewm = df_mean[item_name].ewm(halflife=span,min_periods=span).mean()

           
    ax1 = plt.gca()
    ax1.plot(df_mean['dayno'],item_ewm,label=label,c='r',alpha=0.9,lw=1.5)        # plot ewm
    
    plt.plot(df_mean['dayno'],df_mean[item_name],alpha=0.5)
    
    
    plt.xlim(0,240)
    plt.xlabel('Experiment day')
    ax1.set_ylabel('%s' % label)
        
    plt.axvspan(127, 141, color='y', alpha=0.3, lw=0)
    phases = [[29, 43, 99, 156], ['[2]', '[3]', '[4]', '[5]']]          # indicate experiment phases
    plt.vlines(phases[0],ax1.set_ylim()[0],ax1.set_ylim()[1],colors='k',linestyles='dotted',alpha=0.3)    
         


figureheight = 2.3
figureheight_first_page = 2.0
padding = 0.8
tw = 6
    
item_names = [
    ('mood_relaxed','I feel relaxed'),
    ('mood_down','I feel down'),
    ('mood_irritat','I feel irritated'),
    ('mood_satisfi','I feel satisfied'),
    ]


fig = plt.figure(figsize=(tw,len(item_names)*figureheight_first_page))

for i in range(len(item_names)):
    plt.subplot(len(item_names),1,i+1)
    
    plotItemAvg(*item_names[i])
    

fig.tight_layout(pad=padding)
fig.savefig("H:/Nijmegen/PLOS One/Figures_test/app_items1.pgf",bbox_inches='tight')


item_names = [
    ('mood_lonely','I feel lonely'),
    ('mood_anxious','I feel anxious'),
    ('mood_enthus','I feel enthusiastic'),
    ('mood_suspic','I feel suspicious'),
    ]

fig = plt.figure(figsize=(tw,len(item_names)*figureheight))

for i in range(len(item_names)):
    plt.subplot(len(item_names),1,i+1)
    
    plotItemAvg(*item_names[i])
    

fig.tight_layout(pad=padding)
fig.savefig("H:/Nijmegen/PLOS One/Figures_test/app_items2.pgf",bbox_inches='tight')


item_names = [
    ('mood_cheerf','I feel cheerful'),
    ('mood_guilty','I feel guilty'),
    ('mood_doubt','I feel indecisive'),
    ('mood_strong','I feel strong'),
    ]

fig = plt.figure(figsize=(tw,len(item_names)*figureheight))

for i in range(len(item_names)):
    plt.subplot(len(item_names),1,i+1)
    
    plotItemAvg(*item_names[i])
    

fig.tight_layout(pad=padding)
fig.savefig("H:/Nijmegen/PLOS One/Figures_test/app_items3.pgf",bbox_inches='tight')


item_names = [
    ('pat_restl','I feel restless'),
    ('pat_agitate','I feel agitated'),
    ('pat_worry','I worry'),
    ('pat_concent','I can concentrate well'),
    ]

fig = plt.figure(figsize=(tw,len(item_names)*figureheight))

for i in range(len(item_names)):
    plt.subplot(len(item_names),1,i+1)
    
    plotItemAvg(*item_names[i])
    

fig.tight_layout(pad=padding)
fig.savefig("H:/Nijmegen/PLOS One/Figures_test/app_items4.pgf",bbox_inches='tight')


item_names = [
    ('se_selflike','I like myself'),
    ('se_ashamed','I am ashamed of myself'),
    ('se_selfdoub','I doubt myself'),
    ('se_handle','I can handle everything'),
    ]

fig = plt.figure(figsize=(tw,len(item_names)*figureheight))

for i in range(len(item_names)):
    plt.subplot(len(item_names),1,i+1)
    
    plotItemAvg(*item_names[i])
    

fig.tight_layout(pad=padding)
fig.savefig("H:/Nijmegen/PLOS One/Figures_test/app_items5.pgf",bbox_inches='tight')


item_names = [
    ('phy_hungry','I am hungry'),
    ('phy_tired','I am tired'),
    ('phy_pain','I am in pain'),
    ('phy_dizzy','I am dizzy'),
    ]

fig = plt.figure(figsize=(tw,len(item_names)*figureheight))

for i in range(len(item_names)):
    plt.subplot(len(item_names),1,i+1)
    
    plotItemAvg(*item_names[i])
    

fig.tight_layout(pad=padding)
fig.savefig("H:/Nijmegen/PLOS One/Figures_test/app_items6.pgf",bbox_inches='tight')


item_names = [
    ('phy_drymouth','I have a dry mouth'),
    ('phy_nauseous','I am nauseous'),
    ('phy_headache','I have a headache'),
    ('phy_sleepy','I am sleepy'),
    ]

fig = plt.figure(figsize=(tw,len(item_names)*figureheight))

for i in range(len(item_names)):
    plt.subplot(len(item_names),1,i+1)
    
    plotItemAvg(*item_names[i])
    

fig.tight_layout(pad=padding)
fig.savefig("H:/Nijmegen/PLOS One/Figures_test/app_items7.pgf",bbox_inches='tight')

