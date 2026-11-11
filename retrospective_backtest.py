"""Retrospective 2026 evaluation with frozen 2015–2025 model coefficients.
No historical odds or actual recorded pick claims.
"""
import json
from datetime import datetime,timezone
from pathlib import Path
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss,accuracy_score
from data_pipeline import download,load,features
DATA=Path(__file__).parent/"data"

def run(start="2026-10-20",end="2026-11-09",refresh=True):
    if refresh:
        log=download(2026,2026)
        if any(x.get("status")=="error" for x in log):
            return {"status":"download_failed","log":log}
    d=load()
    if d.empty:return {"status":"no_schedule"}
    f=features(d)
    historic=f[f.completed & f.season.between(2015,2025) & (f.home_score!=f.away_score)].dropna(subset=["elo_diff","home_field"])
    if len(historic)<500:return {"status":"insufficient_training_games","count":len(historic)}
    y=(historic.home_score>historic.away_score).astype(int)
    model=LogisticRegression(max_iter=1000).fit(historic[["elo_diff","home_field"]],y)
    dates=pd.to_datetime(f.game_date,utc=True,errors="coerce")
    start_utc=pd.Timestamp(start,tz="UTC")
    end_utc=pd.Timestamp(end,tz="UTC")+pd.Timedelta(days=1)
    test=f[(f.season==2026)&f.completed&(f.home_score!=f.away_score)&(dates>=start_utc)&(dates<end_utc)].dropna(subset=["elo_diff","home_field"]).copy()
    if test.empty:return {"status":"no_completed_games","start":start,"end":end}
    test["home_probability"]=model.predict_proba(test[["elo_diff","home_field"]])[:,1]
    test["predicted_winner"]=test.apply(lambda r:r.home_team if r.home_probability>=.5 else r.away_team,axis=1)
    test["actual_winner"]=test.apply(lambda r:r.home_team if r.home_score>r.away_score else r.away_team,axis=1)
    test["correct"]=test.predicted_winner==test.actual_winner
    test["confidence"]=test.home_probability.where(test.home_probability>=.5,1-test.home_probability)
    test["week"]=pd.to_datetime(test.game_date,utc=True).dt.strftime("%Y-%m-%d")
    test["bucket"]=pd.cut(test.confidence,[.5,.6,.7,.8,.9,1.0001],labels=["50–60%","60–70%","70–80%","80–90%","90–100%"],include_lowest=True)
    cols=["game_id","game_date","away_team","home_team","away_score","home_score","predicted_winner","actual_winner","home_probability","confidence","correct","bucket"]
    DATA.mkdir(exist_ok=True)
    test[cols].to_csv(DATA/"retrospective_3weeks_games.csv",index=False)
    bins=test.groupby("bucket",observed=True).agg(games=("correct","size"),wins=("correct","sum")).reset_index()
    summary={"status":"ok","start":start,"end":end,"games":len(test),"correct":int(test.correct.sum()),"incorrect":int((~test.correct).sum()),"accuracy":float(accuracy_score(test.actual_winner,test.predicted_winner)),"brier":float(brier_score_loss((test.home_score>test.away_score).astype(int),test.home_probability)),"by_confidence":bins.to_dict("records"),"generated_utc":datetime.now(timezone.utc).isoformat(),"method":"Logistic model trained on completed 2015–2025 games only; Elo features are computed sequentially and use results from games before each matchup. Retrospective reconstructed forecast, not an archived prediction.","betting_roi":"Unavailable: sportsbook quotes and actual picks were not archived.","caution":"2026 Elo strength ratings evolve using earlier 2026 game results, as they would have before each kickoff. Verify schedule completeness and game dates before using this report."}
    (DATA/"retrospective_3weeks_report.json").write_text(json.dumps(summary,indent=2))
    return summary

if __name__=="__main__":print(json.dumps(run(),indent=2))
