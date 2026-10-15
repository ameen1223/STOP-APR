"""Rebuild the 600-DPI calibration-margin sensitivity figure from the archived CSV."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent.parent
s=pd.read_csv(root/'outputs'/'08_calibration_sensitivity.csv')
s=s[(s.policy=='STOP_APR_PC')&(s.calibration_window=='all_validation')]
fig,ax=plt.subplots(figsize=(7.25,3.4))
for split,label,sty,col,total in [
    ('REPO_DISJOINT','Repository-disjoint','-','#344F63',9),
    ('TEMPORAL_STRICT','Strict temporal','--','#A66454',3),
]:
    d=(s[s.split==split].groupby('validation_margin')
       .agg(avoided=('test_avoided_event_rate','mean'),compliant=('test_meets_requested_target','sum')).sort_index())
    x=np.array(d.index)*100
    y=d.avoided.to_numpy()*100
    ax.plot(x,y,sty,marker='o',color=col,lw=1.6,ms=5,label=label)
    for xx,yy,n in zip(x,y,d.compliant):
        ax.annotate(f'{int(n)}/{total}',(xx,yy),xytext=(0,8 if total==9 else -14),
                    textcoords='offset points',ha='center',fontsize=8,color=col)
ax.set_xticks([0,1,3]);ax.set_xlim(-.3,3.34);ax.set_ylim(0,13)
ax.set_xlabel('Additional validation preservation margin (pp)')
ax.set_ylabel('Mean held-out avoided-event rate (%)')
ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
ax.grid(axis='y',alpha=.2);ax.legend(loc='upper right',frameon=False,fontsize=8.2)
fig.tight_layout()
p=root/'figures'/'Figure6_Calibration_Tradeoff.png'
fig.savefig(p,dpi=600,bbox_inches='tight',pad_inches=.10)
print(p)
