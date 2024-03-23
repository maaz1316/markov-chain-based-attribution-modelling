import pandas as pd
import numpy as np
import re
import string
import pyvis
import ast
import plotly.graph_objects as go
from plotly.subplots import make_subplots


df = pd.read_csv('Input/attribution_data.csv')
if 'Source' in df.columns:
    df.rename(columns={'Source':'Default channel group'},inplace=True)
    
df['Default channel group'] = df['Default channel group'].apply(ast.literal_eval)
df = df[['Default channel group','Conversions']]

def get_early_middle_late_touchpoints(df):

    df['Early touchpoints'] = None
    df['Middle touchpoints'] = None
    df['Late touchpoints'] = None
    for i in range(len(df)):
        a = df['Default channel group'][i]
        if len(a) < 2:
            df.at[i, 'Late touchpoints'] = a
        elif len(a) < 3:
            df.at[i, 'Early touchpoints'] = a[:1]
            df.at[i, 'Late touchpoints'] = a[1:2]
        else:
            df.at[i, 'Early touchpoints'] = a[:int(np.round(len(a)*0.25))]
            df.at[i, 'Middle touchpoints'] = a[int(np.round(len(a)*0.25)):int(np.round(len(a)*0.75))]
            df.at[i, 'Late touchpoints'] = a[int(np.round(len(a)*0.75)):]


    df_early = df[['Early touchpoints','Conversions']][~df['Early touchpoints'].isna()]
    df_middle = df[['Middle touchpoints','Conversions']][~df['Middle touchpoints'].isna()]
    df_late = df[['Late touchpoints','Conversions']][~df['Late touchpoints'].isna()]
    df_early.rename(columns={'Early touchpoints':'touchpoints'},inplace=True)
    df_middle.rename(columns={'Middle touchpoints':'touchpoints'},inplace=True)
    df_late.rename(columns={'Late touchpoints':'touchpoints'},inplace=True)
    df_early.reset_index(inplace=True,drop=True)
    df_middle.reset_index(inplace=True,drop=True)
    df_late.reset_index(inplace=True,drop=True)
    
    return df_early, df_middle, df_late

def get_credit_score(df):

    tot_channels=[]
    for i in df['touchpoints']:
        for j in i:
            if j not in tot_channels:
                tot_channels.append(j)

    ## for count of channel transitions(not including start)
    dfs = pd.DataFrame(np.zeros((len(tot_channels),len(tot_channels))),columns=tot_channels,index=tot_channels)

    for k in range(len(df)):
        a = pd.Series(df['touchpoints'][k])
        for i,j in enumerate(zip(a,a.shift(-1))):
            if i < len(a)-1:
                dfs[j[1]][j[0]] = dfs[j[1]][j[0]]+1

    # making start channel
    dfa = pd.DataFrame(np.zeros((1,len(tot_channels))),columns=tot_channels,index=['Start'])
    for i in range(len(df)):
        a = pd.Series(df['touchpoints'][i])
        a = pd.concat([pd.Series('Start'),a],ignore_index=True)
        a[:2]
        dfa[a[1]][a[0]] = dfa[a[1]][a[0]]+1

    #dataframe of count of paths
    df_count = pd.concat([dfa,dfs])
    for i in tot_channels:
        df_count[i][i] = 0

    ##transition probability
    df_prob = df_count.copy()
    for i in range(len(df_count)):
        if df_prob[i:i+1].sum(axis=1).values[0] == 0:
            df_prob[i:i+1] = 0
        else:
            df_prob[i:i+1] = df_prob[i:i+1]/df_prob[i:i+1].sum(axis=1).values[0]

    ## new data frame for conversion probability of paths
    lista = []
    listb = []
    listd = []
    for i in range(len(df)):
        a = df['touchpoints'][i]
        b = ['Start']
        for j in a:
            if b[-1] != j:
                b.append(j)

        lista.append(b)
        listb.append('>'.join(x for x in b))
        listd.append(df['Conversions'][i])
    df_new = pd.DataFrame({'channels':lista,'paths':listb,'conversions':listd})
    df_new['channels'] = df_new['channels'].apply(lambda x: ','.join(x))
    df_new = df_new.groupby(['channels','paths']).sum().reset_index()
    df_new = df_new.sort_values(by='conversions',ascending=False).reset_index()
    df_new['channels'] = df_new['channels'].apply(lambda x: x.split(','))
    df_new['conversions_nm'] = (df_new.conversions - df_new.conversions.min()) / (df_new.conversions.max() - df_new.conversions.min())
    listc = []

    for i in range(len(df_new)):
        a = pd.Series(df_new.iloc[i]['channels'])
        val = 1
        for i,j in enumerate(zip(a,a.shift(-1))):
            if i < len(a)-1:
                val*= df_prob[j[1]][j[0]]
        listc.append(val)
    df_new['probability'] = listc
    df_new['probability'] = df_new['conversions_nm']*df_new['probability']

    ## removal effect
    channels_lvl = {}
    for i in tot_channels:
        sum = 0
        for j in range(len(df_new)):
            if i not in df_new['paths'][j]:
                sum+=df_new['probability'][j]
        channels_lvl[i] = sum

    rem_sum = 0
    for i in channels_lvl.values():
        rem_sum+= 1-(i/df_new.probability.sum())

    grph_dt = {}
    for keys,values in channels_lvl.items():
        grph_dt[keys] = np.round(((1-(values/df_new.probability.sum()))/rem_sum)*100,2)

    return grph_dt

def generate_plot_output(df):

    fig = make_subplots(rows=1, cols=3)

    fig.add_trace(
        go.Bar(name='early',x=df['Channels'],y=df['Early'],texttemplate="%{y}",textposition='outside',textfont_size=14),
              row=1,col=1
    )
    fig.add_trace(
        go.Bar(name='middle',x=df['Channels'],y=df['Middle'],texttemplate="%{y}",textposition='outside',textfont_size=14),
              row=1,col=2
    )
    fig.add_trace(
        go.Bar(name='late',x=df['Channels'],y=df['Late'],texttemplate="%{y}",textposition='outside',textfont_size=14),
              row=1,col=3
    )
    fig.update_layout(height=600, width=1100, title_text="Side By Side Subplots")
    fig.write_image('Outputs/CHANNEL ATTR.png')
    df.to_excel('Outputs/Score sheet.xlsx',index=False)






##generate tables for early, middle and late touch points
df_early, df_middle, df_late = get_early_middle_late_touchpoints(df)

##generate scores for each touchpoint level
grph_dt_er = get_credit_score(df_early)
grph_dt_md = get_credit_score(df_middle)
grph_dt_lt = get_credit_score(df_late)

#create a single table to get score of each channel for each touch point level
tot_channels=[]
for i in df['Default channel group']:
    for j in i:
        if j not in tot_channels:
            tot_channels.append(j)

df_pp = pd.DataFrame({'Channels':tot_channels[:-1]})
df_pp['Early'] = None
df_pp['Middle'] = None
df_pp['Late'] = None

for i in df_pp.Channels:
    if i in grph_dt_er.keys():
        df_pp.loc[df_pp.Channels==i,'Early'] = grph_dt_er[i]
    if i in grph_dt_md.keys():
        df_pp.loc[df_pp.Channels==i,'Middle'] = grph_dt_md[i]
    if i in grph_dt_lt.keys():
        df_pp.loc[df_pp.Channels==i,'Late'] = grph_dt_lt[i]

df_pp = df_pp.fillna(0)
df_pp['Early'] = np.round(df_pp['Early'],1)
df_pp['Middle'] = np.round(df_pp['Middle'],1)
df_pp['Late'] = np.round(df_pp['Late'],1)

generate_plot_output(df_pp)