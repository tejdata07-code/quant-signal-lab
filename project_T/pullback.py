import sys, pandas as pd, numpy as np
sys.path.insert(0,'/Users/tej/vs/fo_paper_trading_tool_FINAL/project_T'); sys.path.insert(0,'/Users/tej/vs/fo_paper_trading_tool_FINAL')
import optbacktest as OB, levels as LV
from app.options import black_scholes
chain=OB.load_chain(); ivm=float(chain.atm_iv.median())/100
b=LV.load_bars('NIFTY','5m'); b['date']=b.datetime.dt.date
SP,CRUSH=0.02,0.15
EXITT={0:'13:30',1:'13:30',2:'13:30',3:'13:30',4:'15:25'}

def build(minpb=5.0):
    """Every bar where premium is >=minpb% off its running peak.
       Target: does the premium make a NEW HIGH before the day's exit?"""
    sig=OB.signals('5m'); sig['dow']=pd.to_datetime(sig.date).dt.weekday
    rows=[]
    for _,s in sig.iterrows():
        day,ts,spot0,dw=s.date,s.ts,s.spot,s.dow
        exp_dt,iv0=OB.session_params(day,ts,chain,ivm)
        if exp_dt<pd.Timestamp(f"{day} 09:15"): continue
        K=int(round(spot0/50)*50); T0=max((exp_dt-ts).total_seconds(),60)/(365*24*3600)
        e=black_scholes(spot0,K,T0,iv0,True)
        if not np.isfinite(e) or e<=2: continue
        ef=e*(1+SP)
        cur=b[b.date==day].sort_values('datetime')
        w=cur[(cur['datetime']>ts)&(cur['datetime'].dt.strftime('%H:%M')<=EXITT[dw])].copy()
        if len(w)<8: continue
        # premium path
        P=[]
        for j,(_,bar) in enumerate(w.iterrows()):
            iv=iv0*(1-CRUSH*(j+1)/len(w)); T=max((exp_dt-bar['datetime']).total_seconds(),60)/(365*24*3600)
            P.append(dict(dt=bar['datetime'],
                          hi=black_scholes(float(bar.high),K,T,iv,True)*(1-SP),
                          lo=black_scholes(float(bar.low),K,T,iv,True)*(1-SP),
                          cl=black_scholes(float(bar.close),K,T,iv,True)*(1-SP),
                          spot=float(bar.close),shi=float(bar.high),slo=float(bar.low),iv=iv))
        P=pd.DataFrame(P)
        P['peak']=P['hi'].cummax()
        tp=(cur.high+cur.low+cur.close)/3; vw=tp.expanding().mean()
        dhi,dlo=cur.high.cummax(),cur.low.cummin()
        for i in range(3,len(P)-1):
            r=P.iloc[i]; pk=r['peak']
            if pk<=ef: continue                        # never been green
            off=100*(1-r['cl']/pk)
            if off<minpb: continue
            fut=P.iloc[i+1:]
            if fut.empty: continue
            newhigh=bool((fut['hi']>pk).any())            # TARGET
            # peak timing
            pj=int(P['hi'][:i+1].idxmax())
            bars_since=i-pj
            ci=cur.index[cur['datetime']==r['dt']]
            ci=ci[0] if len(ci) else None
            rows.append(dict(date=day,dow=dw,t=r['dt'].strftime('%H:%M'),
                hhmm=int(r['dt'].strftime('%H'))*60+int(r['dt'].strftime('%M')),
                off_peak=off, gain_from_entry=100*(r['cl']/ef-1),
                peak_gain=100*(pk/ef-1), bars_since_peak=bars_since,
                spot_mom_30=float(r['spot']-P['spot'].iloc[max(0,i-6)]),
                spot_mom_15=float(r['spot']-P['spot'].iloc[max(0,i-3)]),
                vs_vwap=float(r['spot']-vw.loc[ci]) if ci is not None else np.nan,
                pos_range=100*(r['spot']-float(dlo.loc[ci]))/max(float(dhi.loc[ci])-float(dlo.loc[ci]),1e-9) if ci is not None else np.nan,
                iv_chg=100*(r['iv']/iv0-1),
                spot_vs_peakspot=float(r['spot']-P['spot'].iloc[pj]),
                mins_left=(pd.Timestamp(f"{day} {EXITT[dw]}")-r['dt']).total_seconds()/60,
                dte_h=(exp_dt-r['dt']).total_seconds()/3600,
                y=int(newhigh)))
    return pd.DataFrame(rows)
