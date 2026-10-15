from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
ROOT=Path(__file__).resolve().parents[1]
RES=ROOT/'outputs'
FIG=ROOT/'figures';FIG.mkdir(parents=True,exist_ok=True)
records=pd.read_csv(RES/'07_corrected_comparison.csv')
primary=records[records.split=='REPO_DISJOINT'].sort_values(['seed','target_preservation']).reset_index(drop=True)
temporal=records[records.split=='TEMPORAL_STRICT'].sort_values('target_preservation')
loao=records[records.split.str.startswith('LOAO:')]
a=pd.read_csv(RES/'07_component_ablation.csv')
abpri=a[a.split=='REPO_DISJOINT']
NAVY='#2C4861';BLUE='#587E9D';TEAL='#42878A';ORANGE='#B48D63';GREY='#687B8A';RED='#B35A52';PAPER='#FFFFFF'
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
def save(fig,name):
    p=FIG/name
    fig.savefig(p,dpi=600,bbox_inches='tight',pad_inches=0.12,facecolor='white')
    plt.close(fig)
    return p

# Figure 1 is provided as a publication schematic in figures/.
# The statistical figures below are regenerated from archived numeric outputs.

# Figure 2: precise repo contrasts and 95% repository bootstrap intervals.
fig,ax=plt.subplots(figsize=(7.3,4.45))
r=primary
labels=[f'{int(z.seed)} / {z.target_preservation:.2f}' for z in r.itertuples(index=False)]
y=np.arange(len(r));v=r.delta_vs_validation_selected_baseline.values*100
low=r.delta_ci_low.values*100;high=r.delta_ci_high.values*100
for i,(x,z) in enumerate(zip(v,r.itertuples(index=False))):
    ax.barh(i,x,color=(NAVY if bool(z.stopapr_meets_target) else '#9EA9B0'),height=.55,zorder=2)
ax.errorbar(v,y,xerr=np.vstack((v-low,high-v)),fmt='none',ecolor='#262626',elinewidth=1.0,capsize=2,zorder=3)
ax.set_yticks(y,labels);ax.invert_yaxis();ax.axvline(0,lw=.75,color='#394750')
ax.grid(axis='x',alpha=.13,zorder=0);ax.set_xlabel('Avoided-event-rate difference (percentage points)')
ax.set_ylabel('Repository split / target')
ax.text(.98,.03,'Light bar: preservation target not met',transform=ax.transAxes,ha='right',fontsize=8,color=GREY)
fig2=save(fig,'Figure2_Repository_Bootstrap.png')

# Figure 3: achieved preservation by target, including temporal transfer.
fig,ax=plt.subplots(figsize=(7.0,3.6))
for sd,col,mark in [(2026,NAVY,'o'),(2027,TEAL,'s'),(2028,BLUE,'^')]:
    t=r[r.seed==sd];ax.plot(t.target_preservation,t.stopapr_preservation,marker=mark,color=col,lw=1.45,ms=5,label=f'Repository split {sd}')
ax.plot(temporal.target_preservation,temporal.stopapr_preservation,marker='D',color=RED,lw=1.4,linestyle='--',ms=4.5,label='Strict temporal')
ax.plot([.9,.98],[.9,.98],':',color='#5C6267',lw=.9,label='Target')
ax.set_xlabel('Validation preservation target');ax.set_ylabel('Held-out merged-PR preservation')
ax.set_xticks([.90,.95,.98]);ax.set_ylim(.85,1.006);ax.grid(alpha=.15)
ax.legend(loc='lower right',fontsize=7.3,frameon=False)
fig3=save(fig,'Figure3_Transfer_Preservation.png')

# Figure 4: agent target margins.
fig,ax=plt.subplots(figsize=(7.6,3.65))
agents=['Claude_Code','Copilot','Cursor','Devin','OpenAI_Codex']; xx=np.arange(len(agents));w=.245
for i,(tar,col) in enumerate(zip((.90,.95,.98),(NAVY,TEAL,ORANGE))):
    ss=loao[np.isclose(loao.target_preservation,tar)].set_index('split')
    margins=[(float(ss.loc['LOAO:'+ag,'stopapr_preservation'])-tar)*100 for ag in agents]
    ax.bar(xx+(i-1)*w,margins,width=w,color=col,label=f'ρ = {tar:.2f}')
ax.axhline(0,color='#3A4651',lw=.85)
ax.set_xticks(xx,agents,rotation=12,ha='right');ax.set_ylabel('Preservation margin (percentage points)')
ax.legend(ncol=3,loc='upper left',fontsize=7.7,frameon=False);ax.grid(axis='y',alpha=.11)
fig4=save(fig,'Figure5_Agent_Transfer.png')

# Figure 5: paired ablation contrast intervals.
fig,ax=plt.subplots(figsize=(7.75,4.5))
rout=abpri[abpri.comparison=='outcome_only'].sort_values(['seed','target_preservation']).reset_index(drop=True)
rbur=abpri[abpri.comparison=='burden_only'].sort_values(['seed','target_preservation']).reset_index(drop=True)
labels=[f'{int(z.seed)} / {z.target_preservation:.2f}' for z in rout.itertuples(index=False)]
y=np.arange(len(labels));gap=.18
for d,ys,co,lab in [(rout,y-gap,NAVY,'Combined minus outcome-only'),(rbur,y+gap,TEAL,'Combined minus burden-only')]:
    av=d.gain_full_minus_ablation.values*100
    ax.errorbar(av,ys,xerr=np.vstack([av-d.delta_ci_low.values*100,d.delta_ci_high.values*100-av]),fmt='o',markersize=3.8,color=co,ecolor=co,capsize=2,lw=1,label=lab)
ax.axvline(0,color='#4D5359',lw=.85);ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_ylabel('Repository split / target')
ax.set_xlabel('Difference in avoided-event rate (percentage points)')
ax.legend(frameon=False,fontsize=8,loc='lower right');ax.grid(axis='x',alpha=.12)
fig5=save(fig,'Figure4_Component_Ablation.png')

print("FIGURES_REGENERATED",len(list(FIG.glob("Figure*.png"))))
