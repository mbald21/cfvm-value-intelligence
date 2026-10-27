"""Leakage-safe optional team-game efficiency inputs.
Upload CSV with game_id,team,epa_per_play,success_rate.
One row per team per COMPLETED game. Historical features use only earlier games.
"""
from pathlib import Path
import pandas as pd
DATA=Path(__file__).parent/"data"
COLUMNS=["home_off_epa_prior","away_off_epa_prior","home_success_prior","away_success_prior"]

def build():
    path=DATA/"advanced_team_games.csv"
    schedule=DATA/"features.csv"
    if not path.exists() or not schedule.exists():
        return {"status":"missing_data","detail":"Upload historical team-game EPA and success-rate data, then train the baseline."}
    raw=pd.read_csv(path,dtype={"game_id":str})
    needed={"game_id","team","epa_per_play","success_rate"}
    if not needed.issubset(raw.columns):return {"status":"missing_columns","detail":sorted(needed-set(raw.columns))}
    games=pd.read_csv(schedule,dtype={"game_id":str})
    raw=raw.merge(games[["game_id","game_date","completed"]].drop_duplicates("game_id"),on="game_id",how="inner")
    raw=raw[raw.completed.astype(str).str.lower().isin(["true","1","yes"])].copy()
    raw["game_date"]=pd.to_datetime(raw.game_date,utc=True,errors="coerce")
    for c in ("epa_per_play","success_rate"):raw[c]=pd.to_numeric(raw[c],errors="coerce")
    raw=raw.dropna(subset=["game_date","epa_per_play","success_rate"])
    raw=raw.sort_values(["game_date","game_id"])
    # Aggregate by date before shifting: games played on the same day cannot inform one another.
    daily=raw.groupby(["team","game_date"],as_index=False).agg(epa_per_play=("epa_per_play","mean"),success_rate=("success_rate","mean"))
    daily=daily.sort_values(["team","game_date"])
    daily["prior_epa"]=daily.groupby("team")["epa_per_play"].transform(lambda x:x.shift().expanding(min_periods=3).mean())
    daily["prior_success"]=daily.groupby("team")["success_rate"].transform(lambda x:x.shift().expanding(min_periods=3).mean())
    # asof joins require globally sorted dates, not team-first sorting
    lookup=daily[["team","game_date","prior_epa","prior_success"]].sort_values("game_date")
    result=games.copy()
    result["game_date"]=pd.to_datetime(result.game_date,utc=True,errors="coerce")
    for side in ("home","away"):
        left=result[["game_id","game_date",f"{side}_team"]].rename(columns={f"{side}_team":"team"}).sort_values("game_date")
        right=lookup.rename(columns={"prior_epa":f"{side}_off_epa_prior","prior_success":f"{side}_success_prior"})
        merged=pd.merge_asof(left,right,on="game_date",by="team",direction="backward",allow_exact_matches=False)
        result=result.merge(merged[["game_id",f"{side}_off_epa_prior",f"{side}_success_prior"]],on="game_id",how="left")
    result[["game_id"]+COLUMNS].to_csv(DATA/"advanced_features.csv",index=False)
    coverage=int(result[COLUMNS].notna().all(axis=1).sum())
    return {"status":"ok","covered_games":coverage,"total_games":len(result),"method":"Each team uses the average EPA and success rate from at least three games strictly before kickoff date. Same-day and future results excluded."}
