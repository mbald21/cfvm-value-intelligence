"""Reproducible CFVM baseline. Not the original v0.3 trained model."""
import io,json,pathlib,requests
from datetime import datetime, timezone
import numpy as np,pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss,accuracy_score
ROOT=pathlib.Path(__file__).parent
DATA=ROOT/"data"
DATA.mkdir(exist_ok=True)
BASE="https://github.com/sportsdataverse/sportsdataverse-data/releases/download/espn_cfb_schedules/cfb_schedule_{year}.csv"
CURRENT_SOURCE="https://raw.githubusercontent.com/sportsdataverse/cfbfastR-data/main/schedules/csv/cfb_schedules_2026.csv"
def download(start=2015,end=2026):
    log=[]
    for year in range(start,end+1):
        p=DATA/f"cfb_schedule_{year}.csv"
        if year!=2026 and p.exists() and p.stat().st_size>500:
            log.append({"year":year,"status":"cached"});continue
        try:
            source=CURRENT_SOURCE if year==2026 else BASE.format(year=year)
            r=requests.get(source,timeout=35,headers={"User-Agent":"CFVM-Research/0.5"})
            r.raise_for_status()
            df=pd.read_csv(io.BytesIO(r.content))
            if year==2026:
                required={"game_id","start_date","home_team","away_team","home_points","away_points","completed"}
                if not required.issubset(df.columns):raise ValueError("2026 schedule missing required columns")
                status=df["completed"].astype(str).str.lower().isin(["true","1","yes"])
                df["season"]=2026
                df["game_date"]=df["start_date"]
                df["home_score"]=pd.to_numeric(df["home_points"],errors="coerce").where(status)
                df["away_score"]=pd.to_numeric(df["away_points"],errors="coerce").where(status)
                df.to_csv(p,index=False)
            else:
                if not {"game_id","game_date","home_team","away_team"}.issubset(df.columns):raise ValueError("Unexpected historical source schema")
            if year!=2026:p.write_bytes(r.content)
            log.append({"year":year,"status":"downloaded","rows":len(df),"source":source,"retrieved_utc":datetime.now(timezone.utc).isoformat()})
        except Exception as e:log.append({"year":year,"status":"error","detail":str(e)[:200]})
    (DATA/"download_log.json").write_text(json.dumps(log,indent=2))
    return log
def load():
    fs=[]
    for p in sorted(DATA.glob("cfb_schedule_*.csv")):
        try:fs.append(pd.read_csv(p))
        except Exception:pass
    if not fs:return pd.DataFrame()
    d=pd.concat(fs,ignore_index=True)
    d["game_date"]=pd.to_datetime(d["game_date"],utc=True,errors="coerce")
    d["season"]=pd.to_numeric(d["season"],errors="coerce")
    for c in ("home_score","away_score"):d[c]=pd.to_numeric(d.get(c),errors="coerce")
    return d.dropna(subset=["game_id","game_date","home_team","away_team"]).sort_values("game_date").drop_duplicates("game_id",keep="last")
def features(d):
    elo={};last_year=None;rows=[]
    for day,games in d.groupby(d.game_date.dt.date,sort=True):
        year=int(games.season.dropna().iloc[0]) if games.season.notna().any() else None
        if year!=last_year:
            elo={k:1500+(v-1500)*.75 for k,v in elo.items()};last_year=year
        changes=[]
        for _,g in games.iterrows():
            home,away=str(g.home_team),str(g.away_team)
            h,a=elo.get(home,1500.),elo.get(away,1500.)
            neutral=str(g.get("neutral_site","")).lower() in ("true","1","yes")
            prob=1/(1+10**(-(h+(0 if neutral else 55)-a)/400))
            complete=pd.notna(g.home_score) and pd.notna(g.away_score)
            rows.append({"game_id":str(g.game_id),"game_date":g.game_date.isoformat(),"season":year,"home_team":home,"away_team":away,"home_score":g.home_score,"away_score":g.away_score,"elo_diff":h-a,"home_field":int(not neutral),"completed":bool(complete)})
            if complete:
                result=1 if g.home_score>g.away_score else (0 if g.home_score<g.away_score else .5)
                changes.append((home,away,20*(result-prob)))
        for home,away,delta in changes:
            elo[home]=elo.get(home,1500)+delta;elo[away]=elo.get(away,1500)-delta
    return pd.DataFrame(rows)
def train():
    d=load()
    if d.empty:raise RuntimeError("No data. Run: python data_pipeline.py download")
    f=features(d);f.to_csv(DATA/"features.csv",index=False)
    hist=f[f.completed & f.season.between(2015,2025) & (f.home_score!=f.away_score)].copy()
    if len(hist)<100:raise RuntimeError("Not enough historical games")
    hist["label"]=(hist.home_score>hist.away_score).astype(int)
    tests=[]
    for year in range(2018,2026):
        tr=hist[hist.season<year];te=hist[hist.season==year]
        if len(tr)<100 or len(te)<20 or tr.label.nunique()<2:continue
        model=LogisticRegression(max_iter=1000).fit(tr[["elo_diff","home_field"]],tr.label)
        pred=model.predict_proba(te[["elo_diff","home_field"]])[:,1]
        tests.append({"season":year,"games":len(te),"brier":float(brier_score_loss(te.label,pred)),"accuracy":float(accuracy_score(te.label,pred>=.5))})
    model=LogisticRegression(max_iter=1000).fit(hist[["elo_diff","home_field"]],hist.label)
    future_dates=pd.to_datetime(f.game_date,utc=True,errors="coerce")>pd.Timestamp.now(tz="UTC")
    diagnostics={"source_2026_rows":int((d.season==2026).sum()),"features_2026_rows":int((f.season==2026).sum()),"uncompleted_2026_rows":int(((~f.completed)&(f.season==2026)).sum()),"future_dated_2026_rows":int((future_dates&(f.season==2026)).sum()),"future_uncompleted_2026_rows":int((future_dates&(~f.completed)&(f.season==2026)).sum()),"latest_2026_dates":f.loc[f.season==2026,"game_date"].tail(5).tolist(),"current_utc":pd.Timestamp.now(tz="UTC").isoformat()}
    upcoming=f[(~f.completed)&(f.season>=2026)&future_dates].copy()
    if len(upcoming):upcoming["home_win_probability"]=model.predict_proba(upcoming[["elo_diff","home_field"]])[:,1]
    else:upcoming["home_win_probability"]=pd.Series(dtype=float)
    upcoming[["game_id","game_date","home_team","away_team","home_win_probability"]].to_csv(DATA/"predictions.csv",index=False)
    report={"model":"CFVM v0.4 Elo/logistic baseline, NOT original v0.3","train_games":len(hist),"historical_evaluation":tests,"upcoming_predictions":len(upcoming),"profitability_validated":False,"forecast_source":CURRENT_SOURCE,"generated_utc":datetime.now(timezone.utc).isoformat(),"schedule_diagnostics":diagnostics}
    (DATA/"model_report.json").write_text(json.dumps(report,indent=2))
    return report
if __name__=="__main__":
    import sys
    if len(sys.argv)>1 and sys.argv[1]=="download":print(json.dumps(download(),indent=2))
    elif len(sys.argv)>1 and sys.argv[1]=="train":print(json.dumps(train(),indent=2))
    else:print("Usage: python data_pipeline.py download && python data_pipeline.py train")
