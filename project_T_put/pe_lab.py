import sys, pandas as pd, numpy as np
sys.path.insert(0,'/Users/tej/vs/fo_paper_trading_tool_FINAL/project_T'); sys.path.insert(0,'/Users/tej/vs/fo_paper_trading_tool_FINAL')
import optbacktest as OB, levels as LV
from app.options import black_scholes
chain=OB.load_chain(); ivm=float(chain.atm_iv.median())/100
b=LV.load_bars('NIFTY','5m'); b['date']=b.datetime.dt.date
daily=b.groupby('date').agg(o=('open','first'),h=('high','max'),l=('low','min'),c=('close','last'))
di=list(daily.index)
SP,CRUSH=0.02,0.15
def ctx(day):
    i=di.index(day)
    if i<1: return None
    p=daily.iloc[i-1]; o=float(daily.iloc[i].o)
    return dict(gap=o-float(p.c), pdm=float(p.c)-float(p.o))
def gen(kind, pos_thr=35, mv=20, need_newlow=False):
    out=[]
    for day,cur in b.groupby('date'):
        cur=cur.sort_values('datetime'); op=float(cur.open.iloc[0])
        hi,lo=cur.high.cummax(),cur.low.cummin(); rng=(hi-lo).replace(0,np.nan)
        pos=100*(cur.close-lo)/rng; t=cur.datetime.dt.strftime('%H:%M')
        win=(t>='09:30')&(t<='11:30')
        newlow=cur.low<=lo.shift(1)
        if kind=='dip':        m=win&(pos<=pos_thr)&((cur.close-op)<=-mv)
        elif kind=='pop':      m=win&(pos>=100-pos_thr)&((cur.close-op)>=mv)
        elif kind=='breakdown':m=win&((cur.close-op)<=-mv)&(newlow if need_newlow else True)
        elif kind=='breakout': m=win&((cur.close-op)>=mv)&((cur.high>=hi.shift(1)) if need_newlow else True)
        if not m.any(): continue
        i=m.idxmax(); c=ctx(day)
        out.append(dict(date=day,ts=cur.loc[i,'datetime'],spot=float(cur.loc[i,'close']),
                        pos=float(pos.loc[i]),vs_open=float(cur.loc[i,'close'])-op,
                        gap=(c or {}).get('gap',np.nan),pdm=(c or {}).get('pdm',np.nan)))
    return pd.DataFrame(out)
def run(sig,is_call,t1=.5,t2=1.,book=.5,mon_tue=True,tstop='13:30'):
    v=[];dates=[]
    for _,s in sig.iterrows():
        day,ts,spot=s.date,s.ts,s.spot
        if mon_tue and pd.Timestamp(day).weekday() not in (0,1): continue
        exp_dt,iv0=OB.session_params(day,ts,chain,ivm)
        if exp_dt<pd.Timestamp(f"{day} 09:15"): continue
        K=int(round(spot/50)*50); T0=max((exp_dt-ts).total_seconds(),60)/(365*24*3600)
        e=black_scholes(spot,K,T0,iv0,is_call)
        if not np.isfinite(e) or e<=1: continue
        ef=e*(1+SP); P1,P2=ef*(1+t1),ef*(1+t2)
        cur=b[b.date==day].sort_values('datetime')
        w=cur[(cur.datetime>ts)&(cur.datetime.dt.strftime('%H:%M')<=OB.EOD)]
        if w.empty: continue
        n=len(w); booked=False; r=None
        for j,(_,bar) in enumerate(w.iterrows()):
            iv=iv0*(1-CRUSH*(j+1)/n); T=max((exp_dt-bar.datetime).total_seconds(),60)/(365*24*3600)
            pa=black_scholes(float(bar.high),K,T,iv,is_call); pb=black_scholes(float(bar.low),K,T,iv,is_call)
            ph,pl=max(pa,pb)*(1-SP),min(pa,pb)*(1-SP)
            if not booked:
                if ph>=P1: booked=True; continue
                if tstop and bar.datetime.strftime('%H:%M')>=tstop:
                    r=black_scholes(float(bar.close),K,T,iv,is_call)*(1-SP)-ef; break
            else:
                if ph>=P2: r=book*(P1-ef)+(1-book)*(P2-ef); break
                if pl<=ef: r=book*(P1-ef); break
        if r is None:
            last=w.iloc[-1]; T=max((exp_dt-last.datetime).total_seconds(),60)/(365*24*3600)
            px=black_scholes(float(last.close),K,T,iv0*(1-CRUSH),is_call)*(1-SP)
            r=(book*(P1-ef)+(1-book)*(px-ef)) if booked else (px-ef)
        v.append(r*OB.LOT-OB.CHARGES); dates.append(day)
    return np.array(v),dates
def rep(v,lab):
    if len(v)<8: print(f"  {lab:<40}n={len(v)} too few"); return
    eq=np.cumsum(v); dd=(eq-np.maximum.accumulate(eq)).min(); h=len(v)//2
    both='YES' if v[:h].sum()>0 and v[h:].sum()>0 else 'no'
    print(f"  {lab:<40}{len(v):>4}{100*(v>0).mean():>7.1f}%{v.sum():>+10,.0f}{v.mean():>+8,.0f}{dd:>9,.0f}{both:>6}")
