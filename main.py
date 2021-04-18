import pandas as pd
import re
import numpy as np
import os
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objs as go


cl = 5
n = 8 # указать колличество задач!!!
ol_code = 'sma'
v = 8  # - колличество вариантов

df = pd.read_csv(ol_code + str(cl) + '.csv', encoding='cp1251')

del df['reg_user_id']
del df['last_name']
del df['first_name']
del df['patronymic']
del df['ip']
del df['ua']
del df['key_grade']
del df['issued_at']

raw_results = pd.read_csv('raw_results.csv', sep=';')

df_merged = pd.merge(df, raw_results)

# Preprocessing
ans = []
times = []
n_subm = []
for i in range(n):
    ans.append([])
    times.append([])
    n_subm.append([])
for i in np.arange(len(df_merged)):
    exec('a = ' + re.sub('null', 'None', df_merged.iloc[i]['solutions']))
    res = {}
    time = {}
    subm = {}
    for s in a:
        res[s['task']] = s['submissions'][-1]['data']
        time[s['task']] = s['submissions'][-1]['time']
        subm[s['task']] = len(s['submissions'])
    for j in range(1, n + 1):
        try:
            ans[j-1].append(res[j])
        except:
            ans[j-1].append(None)
        try:
            times[j-1].append(time[j])
        except:
            times[j-1].append(None)
        try:
            n_subm[j-1].append(subm[j])
        except:
            n_subm[j-1].append(0)
for i in range(n):
    df_merged[str(i + 1) + '_ans'] = ans[i]
    df_merged[str(i + 1) + '_time'] = times[i]
    df_merged[str(i + 1) + '_subm'] = n_subm[i]
for i in range(n):
    df_merged[str(i + 1) + '_time'] = pd.to_datetime(df_merged[str(i + 1) + '_time'].apply(lambda x: str(x).replace('T', ' ')[:-1] if x != None else x))
df_merged['perc'] = sum([df_merged[str(i + 1)].fillna(0) for i in range(n)])/(n*10)
df_merged['school'] = df_merged['key'].apply(lambda x: x.split('/')[1][3:])
df_merged['minutes'] = df_merged['minutes'].apply(lambda x: int(x))
# \preprocessing

dff = df_merged.copy()

variants = {}
for i in range(1, v + 1):
    for l in lb:
        if l[0] == str(i) and len(l) > 1:
            if l not in variants and l[1] == '(':
                variants[l] = dff[dff[l] == 1.].iloc[0][str(i) + '_ans']

meansss = []
regions = list(set(dff['school'].apply(lambda x: x[:2]).values))
#regions.remove('12') #нет данных о границах для 12 региона (7,8 класс)
if not os.path.isdir('ma'):
    os.mkdir('ma')
os.chdir('ma')
if not os.path.isdir('ma'+str(cl)):
    os.mkdir('ma'+str(cl))
os.chdir('ma'+str(cl))

df_merged = dff[dff.school.apply(lambda x: x[:2]) == rn]
mean_perc = df_merged.groupby('school')['perc'].mean()
ind = mean_perc.index

numb_st = df_merged.groupby('school')['perc'].count()

romb = df_merged[df_merged['perc'] >= prcts[rn]].groupby('school')['perc'].count()
numb_st_val = [romb.loc[i] if i in list(romb.index) else 0 for i in ind]

scs = []
for i in ind:
    scs.append(df_merged[(df_merged['school'] == i) & (df_merged['perc'] >= prcts[rn])])

pr = dff[(dff['minutes'] > 10) & (dff['perc'] >= 0.1)].groupby('perc')['minutes'].mean()

coef = np.polyfit(list(pr), list(pr.index), 1)

def check(x, ans, ans1=0):
    res = []
    for i in x:
        if i == ans or i == ans1:
            res.append(True)
        else:
            res.append(False)
    return res

vno = []
vvo = []
dt = []
mv = []
for s in scs:
    jj = s.assign(foo=1).merge(s.assign(foo=1), on='foo').drop('foo', 1)
    jj = jj[jj['session_id_x'] < jj['session_id_y']]
    res = 0
    res2 = 0
    m = 0
    for i in range(1, n + 1):
        for p in range(len(s)):
            if s[str(i)].iloc[p] == 0:
                ccl = s.columns
                for l in ccl:
                    if len(l) > 1:
                        if l[0] == str(i) and l[1] == '(':
                            if s[str(i) + '_ans'].iloc[p] == variants[l]:
                                m += 1
        for j in range(len(jj)):
            if jj[str(i) + '_x'].iloc[j] == 0 and jj[str(i) + '_y'].iloc[j] == 0: 
                oks = 0
                if jj[str(i) + '_ans_x'].iloc[j] is None or jj[str(i) + '_ans_y'].iloc[j] is None:
                        continue
                for k in range(len(jj[str(i) + '_ans_x'].iloc[j])):
                    if len(jj[str(i) + '_ans_x'].iloc[j]) != 1:
                        if jj[str(i) + '_ans_x'].iloc[j][k] == jj[str(i) + '_ans_y'].iloc[j][k]:
                            oks += 1
                    else:
                        if jj[str(i) + '_ans_x'].iloc[j] == jj[str(i) + '_ans_y'].iloc[j]:
                            oks = 1
                a = (oks/len(jj[str(i) + '_ans_x'].iloc[j]))**2# СВОБОДНЫЙ ПАРАМЕТР!!!
                # замена для if из формулы выше...

                if a != 0:# здесь посчитаем разницу времен... как-нибудь
                    del_t = abs((jj[str(i) + '_time_x'].iloc[j] - jj[str(i) + '_time_y'].iloc[j]).total_seconds()/60)
                    ans = jj[str(i) + '_ans_x'].iloc[j]
                    ans1 = jj[str(i) + '_ans_y'].iloc[j]
                    col = s[check(s[str(i) + '_ans'], ans, ans1)]
                    col = col['session_id'].count()
                    res += a/(1 + del_t/col**0.33)# Пока возьмем время в минутах, далбше посмотрим
            elif jj[str(i) + '_x'].iloc[j] == 10 and jj[str(i) + '_y'].iloc[j] == 10:
                b = 0.5**2# СВОБОДНЫЙ ПАРАМЕТР!!!
                col = s[s[str(i)] == 10][str(i)].count()
                del_t = abs((jj[str(i) + '_time_x'].iloc[j] - jj[str(i) + '_time_y'].iloc[j]).total_seconds()/60)
                res2 += b/(1 + del_t/col**0.33)
        
    if len(s) != 0:
        mv.append(m/n/len(s)*3)
    else:
        mv.append(0)
    if len(jj) > 0:
        vno.append(res/len(jj))# Здесь нормируем на колличество пар
        vvo.append(res2/len(jj))
    else:
        vno.append(0)
        vvo.append(0)
    tt = 0
    for i in range(len(s)):
        pr = coef[1] + s['minutes'].iloc[i]*coef[0]
        tt += (s['perc'].iloc[i] - pr)/((s['minutes'].iloc[i] + 1)**0.5/3)
    if len(s) != 0:
        dt.append(tt/len(s))
    else:
        dt.append(0.)
    
means = pd.DataFrame(data=np.array([ind, mean_perc.values, numb_st.loc[ind].values, numb_st_val, vno, vvo, dt, mv]).T,
                     columns=['school', 'mean_perc', 'numb_st', 'numb_val', 'vno', 'vvo', 'dt', 'mv'])
means['lvl_sus'] = means['numb_val']/means['numb_st']*(means['dt'] + means['mv'] + means['vno'] + means['vvo']*2 + (means['mean_perc'] - df_merged['perc'].mean())/(((means['mean_perc'] - df_merged['perc'].mean())**2).mean())**0.5/10)
real_sc = {i[1]: i[0] for i in dff[dff.school.apply(lambda x: x[:2]) == rn][['real_school', 'school']].values}
means['real_school'] = means['school'].apply(lambda x: real_sc[x])
means[['school', 'real_school', 'lvl_sus']].to_csv(ol_code + str(cl) + '_analys' + '.csv', index=False, encoding='cp1251')
os.chdir('..')
os.chdir('..')

means['O'] = (means['vno'] + means['vvo']*2)*means['numb_val']/means['numb_st']
means['dp'] = (means['mean_perc'] - df_merged['perc'].mean())/(((means['mean_perc'] - dff[dff.school.apply(lambda x: x[:2]) == '05']['perc'].mean())**2).mean())**0.5/10*means['numb_val']/means['numb_st']
means['dtt'] = means['dt']*3*means['numb_val']/means['numb_st']
means['lvl_sus'] = means['O'] + means['dp'] + means['dtt']

maxx = means['lvl_sus'].max()
pc = 0.45

fig1 = go.Scatter3d(x=means[means['lvl_sus'] < pc*maxx]['O'],
                    y=means[means['lvl_sus'] < pc*maxx]['dp'],
                    z=means[means['lvl_sus'] < pc*maxx]['dtt'],
                    marker=dict(opacity=0.9,
                                reversescale=True,
                                colorscale='Blues',
                                size=5),
                    line=dict (width=0.02),
                    mode='markers')

fig2 = go.Scatter3d(x=means[means['lvl_sus'] >= pc*maxx]['O'],
                    y=means[means['lvl_sus'] >= pc*maxx]['dp'],
                    z=means[means['lvl_sus'] >= pc*maxx]['dtt'],
                    marker=dict(opacity=0.9,
                                reversescale=True,
                                colorscale='Reds',
                                size=5),
                    line=dict (width=0.02),
                    mode='markers')

#Make Plot.ly Layout
mylayout = go.Layout(scene=dict(xaxis=dict( title="O"),
                                yaxis=dict( title="dp"),
                                zaxis=dict(title="d<p(t)>")),)

#Plot and save html
plotly.offline.plot({"data": [fig1, fig2],
                     "layout": mylayout},
                     auto_open=True,
                     filename=ol_code + str(cl) + ".html")
