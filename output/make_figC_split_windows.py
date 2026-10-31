import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
U="preprocessing/"
OUT="figures/figC_split_and_windows"
z=np.load(U+"c3_windows_gap.npz",allow_pickle=False)
td=pd.to_datetime(z["target_date"]); sp=z["split"]
df=pd.DataFrame({"t":td,"s":sp}); df["q"]=df.t.dt.to_period("Q")
cnt=df.groupby(["q","s"]).size().unstack(fill_value=0)
qs=pd.period_range("2015Q1","2024Q4",freq="Q"); cnt=cnt.reindex(qs,fill_value=0)
COL={"train":"#0072B2","val":"#E69F00","test":"#CC79A7"}
INK="#222222"; MUT="#666666"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.edgecolor":"#999999","axes.linewidth":0.8})
fig,(a,b)=plt.subplots(1,2,figsize=(12.2,4.4),gridspec_kw={"width_ratios":[1.15,1]})
x=np.arange(len(qs))
for s in ["train","val","test"]:
    if s in cnt: a.bar(x,cnt[s].values,width=0.78,color=COL[s],edgecolor="white",linewidth=0.6)
q1=[i for i,p in enumerate(qs) if p.quarter==1]; a.set_xticks(q1); a.set_xticklabels([str(qs[i].year) for i in q1])
a.set_ylabel("Windows per target quarter"); a.set_ylim(0,max(cnt.sum(axis=1))*1.55)
for sp_ in ["top","right"]: a.spines[sp_].set_visible(False)
a.yaxis.grid(True,color="#e5e5e5",lw=0.7); a.set_axisbelow(True)
def qi(p): return list(qs).index(pd.Period(p,freq="Q"))
ct=qi("2022Q3")+0.5; cv=qi("2023Q4")+0.5
for c in (ct,cv): a.axvline(c,color=MUT,ls=(0,(4,3)),lw=1)
ymax=a.get_ylim()[1]
n={s:int((sp==s).sum()) for s in COL}
a.text((x[0]+ct)/2,ymax*0.93,f"Train: {n['train']} windows\ntarget up to 2022-09-30",ha="center",va="top",color=INK,fontsize=8.5,fontweight="bold")
a.text((ct+cv)/2-0.2,ymax*0.93,f"Val\n{n['val']}",ha="center",va="top",color=INK,fontsize=8.5,fontweight="bold")
a.text((cv+x[-1])/2+0.6,ymax*0.93,f"Test\n{n['test']}",ha="center",va="top",color=INK,fontsize=8.5,fontweight="bold")
a.set_title("(a) Chronological split of the 840 windows, by target date",loc="left",fontsize=10,color=INK,pad=8)
# panel b
d=pd.read_csv(U+"c2_normalised.csv",parse_dates=["date"])
k=d[d.well_id=="Keelvalur"].sort_values("date").reset_index(drop=True)
inp=k.iloc[:8]; tg=k.iloc[8]
b.plot(k.date.iloc[:12],k.depth_m.iloc[:12],color="#bbbbbb",lw=1,zorder=1)
b.scatter(k.date.iloc[9:12],k.depth_m.iloc[9:12],s=26,color="#bbbbbb",zorder=2)
b.plot(inp.date,inp.depth_m,color=COL["train"],lw=1.6,zorder=3)
b.scatter(inp.date,inp.depth_m,s=42,color=COL["train"],edgecolor="white",linewidth=1,zorder=4)
b.scatter([tg.date],[tg.depth_m],s=120,facecolor="white",edgecolor=COL["val"],linewidth=2.2,zorder=5)
b.plot([inp.date.iloc[-1],tg.date],[inp.depth_m.iloc[-1],tg.depth_m],color=COL["val"],lw=1.6,ls=(0,(3,2)),zorder=3)
b.set_ylim(6.0,1.0); b.set_xlim(pd.Timestamp("2013-08-01"),pd.Timestamp("2018-04-01"))
b.set_ylabel("Depth to water level (m)")
b.xaxis.set_major_locator(mdates.YearLocator()); b.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
for sp_ in ["top","right"]: b.spines[sp_].set_visible(False)
b.yaxis.grid(True,color="#e5e5e5",lw=0.7); b.set_axisbelow(True)
ymin=5.78
gaps=inp.gap_from_prev_quarters.values
for i in range(1,8):
    mid=inp.date.iloc[i-1]+(inp.date.iloc[i]-inp.date.iloc[i-1])/2
    b.text(mid,ymin,f"{int(gaps[i])}",ha="center",va="center",fontsize=9,color=INK)
b.text(pd.Timestamp("2016-12-20"),ymin,"quarters since the\nprevious reading",ha="left",va="center",fontsize=7.5,color=MUT)
b.annotate("8 input readings",xy=(inp.date.iloc[4],inp.depth_m.iloc[4]),xytext=(pd.Timestamp("2014-02-01"),1.5),
           fontsize=9,color=INK,ha="center",arrowprops=dict(arrowstyle="-",color=MUT,lw=0.8))
b.annotate("target: next reading\n(5.10 m)",xy=(tg.date,tg.depth_m),xytext=(pd.Timestamp("2017-01-15"),3.0),
           fontsize=9,color=INK,ha="left",arrowprops=dict(arrowstyle="-",color=MUT,lw=0.8))

b.set_title("(b) One window at Keelvalur: 8 readings and the next one",loc="left",fontsize=10,color=INK,pad=8)

fig.tight_layout()
fig.savefig(OUT+".png",dpi=300); fig.savefig(OUT+".pdf")
print(cnt.sum().to_dict(), tg.date, tg.depth_m, list(gaps))
