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
    advanced_cols=["home_off_epa_prior","away_off_epa_prior","home_success_prior","away_success_prior"]
    advanced_path=DATA/"advanced_features.csv"
    if advanced_path.exists():
        adv=pd.read_csv(advanced_path,dtype={"game_id":str})
        hist["game_id"]=hist["game_id"].astype(str)
        hist=hist.merge(adv[["game_id"]+advanced_cols].drop_duplicates("game_id"),on="game_id",how="left")
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
        if all(c in hist.columns for c in advanced_cols):
            tr_adv=train.dropna(subset=advanced_cols)
            te_adv=test.dropna(subset=advanced_cols)
            if len(tr_adv)>=500 and len(te_adv)>=20 and tr_adv.won.nunique()==2:
                enhanced=HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=10,min_samples_leaf=40,l2_regularization=10,random_state=42)
                enhanced.fit(tr_adv[FEATURES+advanced_cols],tr_adv.won)
                p=np.clip(enhanced.predict_proba(te_adv[FEATURES+advanced_cols])[:,1],.001,.999)
                rows.append({"Season":int(season),"Model":"EPA + success + Elo (experimental)","Games":len(te_adv),"Correct":int(accuracy_score(te_adv.won,p>=.5)*len(te_adv)),"Brier total":float(np.square(p-te_adv.won.to_numpy()).sum()),"Log loss total":float(log_loss(te_adv.won,p,labels=[0,1])*len(te_adv))})
                # Matched-sample baseline: score the original models on exactly these games.
                mask=test.index.isin(te_adv.index)
                for base_name,base_probs in forecasts.items():
                    subset=base_probs[mask]
                    actual=te_adv.won.to_numpy()
                    rows.append({"Season":int(season),"Model":base_name+" (same EPA games)","Games":len(te_adv),"Correct":int(accuracy_score(actual,subset>=.5)*len(te_adv)),"Brier total":float(np.square(subset-actual).sum()),"Log loss total":float(log_loss(actual,subset,labels=[0,1])*len(te_adv))})
        for name,probs in forecasts.items():
            rows.append({"Season":int(season),"Model":name,"Games":len(test),"Correct":int(accuracy_score(yte,probs>=.5)*len(test)),"Brier total":float(np.square(probs-yte.to_numpy()).sum()),"Log loss total":float(log_loss(yte,probs,labels=[0,1])*len(test))})
    if not rows:return {"status":"no_test_seasons"}
    frame=pd.DataFrame(rows)
    summary=[]
    for name,group in frame.groupby("Model"):
        games=int(group.Games.sum())
        summary.append({"Model":name,"Test games":games,"Win pick accuracy":round(group.Correct.sum()/games,4),"Brier score":round(group["Brier total"].sum()/games,4),"Log loss":round(group["Log loss total"].sum()/games,4)})
    summary=sorted(summary,key=lambda r:r["Brier score"])
    result={"status":"ok","features":FEATURES,"method":"Train on all seasons before each test year (2018-2025); never train on the tested year.","winner_by_brier":min((r for r in summary if "same EPA games" not in r["Model"] and "experimental" not in r["Model"]),key=lambda r:r["Brier score"])["Model"],"leaderboard":summary,"season_results":rows,"limitations":"Baseline uses Elo and venue. Optional experimental model uses pregame EPA and success-rate averages, if uploaded. Different coverage means scores are not strictly comparable. No verified betting ROI."}
    (DATA/"model_competition.json").write_text(json.dumps(result,indent=2))
    return result

if __name__=="__main__":
    print(json.dumps(evaluate(),indent=2))
