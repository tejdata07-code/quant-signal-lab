"""SYNTHETIC ANUJ — predict his NIFTY/BANKNIFTY support/resistance BEFORE his video.

Information set = price action through the prior session + his OWN previously-published
levels (available before today's video). A gradient-boosted model, trained on his
historical calls, ranks price-derived candidate levels (recent intraday swings, prior-
day H/L, round psychological numbers, trendline projections, his prior levels) by how
"Anuj-like" they are, then emits the top few resistances and supports per index.

HONEST ACCURACY (date-split backtest, tol +/-50pts = 1 round tick):
  * candidate pool CONTAINS his level 98% of the time (coverage is ~solved)
  * exact top-2-per-side match ~50% (the final pick is discretionary)
  * his level is inside the top-4 shortlist ~72%, top-6 ~79%
100% exact replication is NOT achievable: ~half the time the specific 2 he emphasizes
depend on judgment (trendlines drawn by eye, intraday bias) not present in price data.

Usage:  python scripts/anuj_predict.py [YYYY-MM-DD]   (default: session after last bar)
        python scripts/anuj_predict.py --backtest
"""
from __future__ import annotations
import sys, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))          # so `app.*` imports work under cron (any cwd)
IDXS=["NIFTY","BANKNIFTY"]
r50=lambda x: round(x/50)*50
LV=pd.read_csv(ROOT/"data/anuj_levels/levels.csv",parse_dates=["date"]); LV["mid"]=(LV.low+LV.high)/2

def _daily(df):
    d=df.copy(); d["day"]=d.datetime.dt.date
    g=d.groupby("day").agg(H=("high","max"),Lo=("low","min"),C=("close","last")).reset_index()
    g["day"]=pd.to_datetime(g.day); return g.sort_values("day").reset_index(drop=True)

def load_frames():
    global FIVE, OH
    FIVE={i:pd.read_csv(ROOT/f"data/history/{i}_5m.csv",parse_dates=["datetime"]) for i in IDXS}
    OH={i:_daily(FIVE[i]) for i in IDXS}

def refresh_history():
    """Best-effort: pull fresh index candles ourselves so predictions never depend on
    the (flaky) scanner daemon being up. Falls back to existing history on any failure."""
    try:
        from app.market_data import maintain_index_history
        maintain_index_history()
    except Exception as e:
        print(f"[warn] index-history refresh failed ({type(e).__name__}: {str(e)[:80]}); using existing history")
    load_frames()

def next_session():
    """Real upcoming NSE session: next business day after the last bar, but never before
    today -> stale history can no longer mis-date the prediction."""
    today=pd.Timestamp.now().normalize()
    last=max(OH["NIFTY"].day.max(), OH["BANKNIFTY"].day.max())
    D=last+pd.offsets.BDay(1)
    if D<today: D=today
    if D.weekday()>=5: D=(D+pd.offsets.BDay(1)).normalize()
    return pd.Timestamp(D)

load_frames()
def pivots(idx,D,lb=15):
    d=FIVE[idx]; d=d[(d.datetime<pd.Timestamp(D))&(d.datetime>=pd.Timestamp(D)-pd.Timedelta(days=lb+4))]
    if len(d)<20: return [],[]
    h,l=d.high.values,d.low.values; hi=[];lo=[]
    for i in range(3,len(d)-3):
        if h[i]==max(h[i-3:i+4]): hi.append(h[i])
        if l[i]==min(l[i-3:i+4]): lo.append(l[i])
    return hi,lo
def tproj(p):
    if len(p)<3: return None
    y=np.array(p[-5:]); m,b=np.polyfit(np.arange(len(y)),y,1); return m*len(y)+b
FEATS=["dist","adist","prior_wt","prior_cnt","prior_recent","prior_res","prior_sup","touch","is_pivot",
       "is_prevH","is_prevL","is5H","is5L","is10H","is10L","near_trendH","near_trendL","r100","r250","r500","r1000","near_rank"]
def feats(idx,D,label=True):
    g=OH[idx]; past=g[g.day<pd.Timestamp(D)]
    if len(past)<5: return None
    C=past.iloc[-1].C; pv=past.iloc[-1]; d5=past.tail(5); d10=past.tail(10)
    prior=LV[(LV["index"]==idx)&(LV.date<D)&(LV.date>=D-pd.Timedelta(days=14))]
    hipv,lopv=pivots(idx,D); allpv=hipv+lopv; tH=tproj(hipv); tL=tproj(lopv)
    cand=set()
    for _,r in prior.iterrows(): cand.add(r50(r.mid))
    for p in allpv: cand.add(r50(p))
    for _,r in past.tail(5).iterrows(): cand.add(r50(r.H)); cand.add(r50(r.Lo))
    for m in (100,250,500,1000):
        base=round(C/m)*m
        for k in range(-3,4):
            if abs(base+k*m-C)<=C*0.03: cand.add(r50(base+k*m))
    if tH: cand.add(r50(tH))
    if tL: cand.add(r50(tL))
    rows=[]
    for k in cand:
        if k<=0: continue
        pr=prior[abs(prior.mid-k)<=50]; ages=[(pd.Timestamp(D)-t).days for t in pr.date]
        touch=sum(1 for p in allpv if abs(p-k)<=40)
        rows.append(dict(idx=idx,date=pd.Timestamp(D),level=k,C=C,above=int(k>C),
            dist=(k-C)/C*100,adist=abs(k-C)/C*100,
            prior_wt=sum(np.exp(-a/6) for a in ages),prior_cnt=len(pr),prior_recent=(min(ages) if ages else 99),
            prior_res=int((pr.kind=="resistance").sum()),prior_sup=int((pr.kind=="support").sum()),
            touch=touch,is_pivot=int(touch>0),
            is_prevH=int(abs(k-pv.H)<=50),is_prevL=int(abs(k-pv.Lo)<=50),
            is5H=int(abs(k-d5.H.max())<=50),is5L=int(abs(k-d5.Lo.min())<=50),
            is10H=int(abs(k-d10.H.max())<=50),is10L=int(abs(k-d10.Lo.min())<=50),
            near_trendH=int(tH is not None and abs(k-tH)<=60),near_trendL=int(tL is not None and abs(k-tL)<=60),
            r100=int(k%100==0),r250=int(k%250==0),r500=int(k%500==0),r1000=int(k%1000==0)))
    df=pd.DataFrame(rows); df["near_rank"]=df.groupby("above").adist.rank()
    if label:
        act=LV[(LV["index"]==idx)&(LV.date==D)]
        aR=act[act.kind=="resistance"].mid.values; aS=act[act.kind=="support"].mid.values
        df["y"]=df.apply(lambda r:int(len(aR) and (np.abs(aR-r.level)<=50).any()) if r.above
                         else int(len(aS) and (np.abs(aS-r.level)<=50).any()),axis=1)
    return df
def train_upto(D):
    rows=[feats(i,d) for i in IDXS for d in sorted(LV[LV["index"]==i].date.unique()) if d<pd.Timestamp(D)]
    X=pd.concat([r for r in rows if r is not None and len(r)],ignore_index=True)
    m=GradientBoostingClassifier(n_estimators=250,max_depth=3,learning_rate=0.04,subsample=0.8,random_state=0)
    m.fit(X[FEATS],X.y); return m
def predict(D,topk=3):
    m=train_upto(D); out={}
    for idx in IDXS:
        f=feats(idx,D,label=False)
        if f is None or not len(f): continue
        f["p"]=m.predict_proba(f[FEATS])[:,1]
        R=f[f.above==1].sort_values("p",ascending=False).head(topk)
        S=f[f.above==0].sort_values("p",ascending=False).head(topk)
        out[idx]=(f.C.iloc[0],R[["level","p"]].values,S[["level","p"]].values)
    return out

def score_stats():
    """Aggregate recall of logged predictions vs his ingested actuals (past sessions only)."""
    logf=ROOT/"data"/"anuj_levels"/"predicted_log.csv"
    if not logf.exists(): return None
    P=pd.read_csv(logf); A=LV.copy(); A["d"]=A.date.dt.strftime("%Y-%m-%d")
    common=sorted(set(P.date)&set(A.d))
    if not common: return None
    agg={2:[0,0],3:[0,0]}; bydate=[]
    for d in common:
        dh=dt=0
        for idx in IDXS:
            for kind in ("resistance","support"):
                act=A[(A.d==d)&(A["index"]==idx)&(A.kind==kind)].mid.tolist()
                pr=P[(P.date==d)&(P["index"]==idx)&(P.kind==kind)].sort_values("rank")
                for K in (2,3):
                    pk=pr.head(K).level.tolist()
                    for av in act:
                        hit=any(abs(p-av)<=50 for p in pk); agg[K][0 if hit else 1]+=1
                        if K==3: dh+=int(hit); dt+=1
        bydate.append((d,dh,dt))
    r=lambda k:(agg[k][0]/(agg[k][0]+agg[k][1])*100) if (agg[k][0]+agg[k][1]) else 0.0
    return dict(n=len(common), r2=r(2), r3=r(3), bydate=bydate)


def write_prediction(D=None, topk=3):
    """Write today's predicted levels to a dated markdown file + append a scored log."""
    refresh_history()                              # self-sufficient: pull fresh candles first
    D = pd.Timestamp(D) if D else next_session()
    preds = predict(D, topk=topk)
    outdir = ROOT/"data"/"anuj_levels"; outdir.mkdir(parents=True, exist_ok=True)
    md = [f"# Synthetic-Anuj predicted levels \u2014 session {D.date()}",
          f"_generated {pd.Timestamp.now():%Y-%m-%d %H:%M} \u00b7 pre-video \u00b7 NIFTY + BankNifty only_", "",
          "> Honest accuracy (walk-forward, \u00b150pts): his level in **top-3 ~57%**, top-4 ~65%, exact ~44%.",
          "> This is a ranked **shortlist**, not an exact clone \u2014 the final pick is his discretion.", ""]
    _st = score_stats()
    if _st:
        _l = _st["bydate"][-1]
        md += [f"**Track record** (last {_st['n']} scored sessions): recall@3 **{_st['r3']:.0f}%** \u00b7 recall@2 {_st['r2']:.0f}%"
               f" \u2014 most recent {_l[0]}: caught {_l[1]}/{_l[2]} of his levels.", ""]
    logrows = []
    for idx,(C,R,S) in preds.items():
        md.append(f"## {idx}  (prev close ~{C:.0f})")
        md.append("| # | RESISTANCE | conf | \u2003 | # | SUPPORT | conf |")
        md.append("|--:|--:|--:|--|--:|--:|--:|")
        for i in range(max(len(R), len(S))):
            rr = f"{int(R[i][0])} | {R[i][1]*100:.0f}%" if i < len(R) else " | "
            ss = f"{int(S[i][0])} | {S[i][1]*100:.0f}%" if i < len(S) else " | "
            md.append(f"| {i+1} | {rr} | \u2003 | {i+1} | {ss} |")
        md.append("")
        for rank,(l,p) in enumerate(R,1): logrows.append((str(D.date()),idx,"resistance",rank,int(l),round(float(p),3)))
        for rank,(l,p) in enumerate(S,1): logrows.append((str(D.date()),idx,"support",rank,int(l),round(float(p),3)))
    (outdir/"predicted_today.md").write_text("\n".join(md))
    logf = outdir/"predicted_log.csv"
    new = pd.DataFrame(logrows, columns=["date","index","kind","rank","level","prob"])
    if logf.exists():
        old = pd.read_csv(logf); old = old[old.date != str(D.date())]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(logf, index=False)
    print(f"[{pd.Timestamp.now():%Y-%m-%d %H:%M}] wrote predicted_today.md + {len(logrows)} log rows for session {D.date()}")


def score_log():
    """Live verification: of HIS actual levels, how many did our top-K shortlist catch
    (recall) — same metric as --backtest, so live vs backtest are comparable."""
    outdir=ROOT/"data"/"anuj_levels"; logf=outdir/"predicted_log.csv"
    if not logf.exists(): print("no predicted_log.csv yet"); return
    P=pd.read_csv(logf); A=LV.copy(); A["d"]=A.date.dt.strftime("%Y-%m-%d")
    common=sorted(set(P.date)&set(A.d))
    if not common: print("no overlap with his actual levels yet (predictions await his videos)"); return
    agg={K:[0,0] for K in (2,3)}; bydate=[]
    for d in common:
        dh=0; dt=0
        for idx in IDXS:
            for kind in ("resistance","support"):
                act=A[(A.d==d)&(A["index"]==idx)&(A.kind==kind)].mid.tolist()
                preds=P[(P.date==d)&(P["index"]==idx)&(P.kind==kind)].sort_values("rank")
                for K in (2,3):
                    pk=preds.head(K).level.tolist()
                    for av in act:
                        hit=any(abs(p-av)<=50 for p in pk); agg[K][0 if hit else 1]+=1
                        if K==3: dh+=hit; dt+=1
        bydate.append((d,dh,dt))
    print(f"LIVE recall over {len(common)} scored sessions (his levels caught by our shortlist, +/-50pts):")
    for K in (2,3):
        tp,fn=agg[K]; print(f"  recall@{K}: {tp}/{tp+fn} = {tp/(tp+fn)*100:.0f}%   (backtest said ~{ {2:44,3:57}[K] }%)")
    print("  by session (recall@3):")
    for d,h,t in bydate: print(f"    {d}: {h}/{t}")


if __name__=="__main__":
    args = sys.argv[1:]
    if "--backtest" in args:
        allrows=[feats(i,d) for i in IDXS for d in sorted(LV[LV["index"]==i].date.unique())]
        X=pd.concat([r for r in allrows if r is not None and len(r)],ignore_index=True)
        tr=X[X.date<pd.Timestamp("2026-07-15")]
        m=GradientBoostingClassifier(n_estimators=250,max_depth=3,learning_rate=0.04,subsample=0.8,random_state=0).fit(tr[FEATS],tr.y)
        X["p"]=m.predict_proba(X[FEATS])[:,1]; te=X[X.date>=pd.Timestamp("2026-07-15")]
        def rk(sub,K):
            tp=fn=0
            for (idx,D),g in sub.groupby(["idx","date"]):
                act=LV[(LV["index"]==idx)&(LV.date==D)]
                for ab,kd in [(1,"resistance"),(0,"support")]:
                    a=act[act.kind==kd].mid.tolist(); pred=g[g.above==ab].sort_values("p",ascending=False).head(K).level.tolist()
                    for av in a: tp+= any(abs(p-av)<=50 for p in pred); fn+= not any(abs(p-av)<=50 for p in pred)
            return tp/(tp+fn) if tp+fn else 0
        print("TEST recall@K (unseen dates):", {K:round(rk(te,K)*100) for K in (2,3,4,5,6)})
    elif "--write" in args:
        d=[a for a in args if not a.startswith("--")]
        write_prediction(d[0] if d else None)
    elif "--score" in args:
        score_log()
    else:
        d=[a for a in args if not a.startswith("--")]
        D=pd.Timestamp(d[0]) if d else next_session()
        print(f"\n=== SYNTHETIC ANUJ \u2014 predicted levels for session {D.date()} (pre-video) ===")
        for idx,(C,R,S) in predict(D).items():
            print(f"\n{idx}  (prev close ~{C:.0f})")
            print("  RESISTANCE:  "+"  ".join(f"{int(l)} ({p*100:.0f}%)" for l,p in R))
            print("  SUPPORT:     "+"  ".join(f"{int(l)} ({p*100:.0f}%)" for l,p in S))
