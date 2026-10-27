"""Chronological model competition using only currently available pregame features.
No sportsbook ROI claims: historical timestamped odds are not yet available.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

DATA=Path(__file__).parent/"data"
FEATURES=["elo_diff","home_field"]

def candidates():
    return {
        "Elo + logistic (current)": LogisticRegression(max_iter=1000),
        "Random forest": RandomForestClassifier(n_estimators=180,max_depth=5,min_samples_leaf=30,random_state=42,n_jobs=-1),
        "Gradient boosting": HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=10,min_samples_leaf=40,l2_regularization=10,random_state=42),
    }

def evaluate():
    path=DATA/"features.csv"
    if not path.exists():
        return {"status":"missing_features","detail":"Train the baseline first to generate historical features."}
    df=pd.read_csv(path)
    required=set(FEATURES+["season","completed","home_score","away_score"])
    if not required.issubset(df.columns):
        return {"status":"missing_columns","detail":sorted(required-set(df.columns))}
    df["season"]=pd.to_numeric(df["season"],errors="coerce")
    df["home_score"]=pd.to_numeric(df["home_score"],errors="coerce")
    df["away_score"]=pd.to_numeric(df["away_score"],errors="coerce")
    df["completed"]=df["completed"].astype(str).str.lower().isin(["true","1","yes"])
    hist=df[df["completed"] & df["season"].between(2015,2025) & (df.home_score!=df.away_score)].dropna(subset=FEATURES+["home_score","away_score"]).copy()
    hist["won"]=(hist.home_score>hist.away_score).astype(int)
    if len(hist)<500:return {"status":"insufficient_data","games":len(hist)}
    seasons=range(2018,2026)
    rows=[]
    for season in seasons:
        train=hist[hist.season<season]
        test=hist[hist.season==season]
        if len(train)<100 or len(test)<20 or train.won.nunique()!=2:continue
        xtr=train[FEATURES];ytr=train.won
        xte=test[FEATURES];yte=test.won
        forecasts={}
        for name,model in candidates().items():
            model.fit(xtr,ytr)
            forecasts[name]=np.clip(model.predict_proba(xte)[:,1],.001,.999)
        forecasts["Equal-weight ensemble"]=np.mean(np.column_stack(list(forecasts.values())),axis=1)
        for name,probs in forecasts.items():
            rows.append({"Season":int(season),"Model":name,"Games":len(test),"Correct":int(accuracy_score(yte,probs>=.5)*len(test)),"Brier total":float(np.square(probs-yte.to_numpy()).sum()),"Log loss total":float(log_loss(yte,probs,labels=[0,1])*len(test))})
    if not rows:return {"status":"no_test_seasons"}
    frame=pd.DataFrame(rows)
    summary=[]
    for name,group in frame.groupby("Model"):
        games=int(group.Games.sum())
        summary.append({"Model":name,"Test games":games,"Win pick accuracy":round(group.Correct.sum()/games,4),"Brier score":round(group["Brier total"].sum()/games,4),"Log loss":round(group["Log loss total"].sum()/games,4)})
    summary=sorted(summary,key=lambda r:r["Brier score"])
    result={"status":"ok","features":FEATURES,"method":"Train on all seasons before each test year (2018-2025); never train on the tested year.","winner_by_brier":summary[0]["Model"],"leaderboard":summary,"season_results":rows,"limitations":"Only pregame Elo difference and home-field indicator are available. No injuries, EPA, rosters, odds-history backtest, verified profit or proven betting edge. Ensemble averages three models without tuning."}
    (DATA/"model_competition.json").write_text(json.dumps(result,indent=2))
    return result

if __name__=="__main__":
    print(json.dumps(evaluate(),indent=2))
