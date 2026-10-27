"""Fetch official SportsDataverse release assets; normalize ESPN team-game statistics."""
from pathlib import Path
from datetime import datetime,timezone
import io,json,re
import pandas as pd
import requests
DATA=Path(__file__).parent/"data"
ROOT="https://github.com/sportsdataverse/sportsdataverse-data/releases/download"

def norm(x):
    return re.sub(r"[^a-z0-9]","",str(x).lower())

def download_efficiency(start=2018,end=2025):
    schedule=DATA/"features.csv"
    if not schedule.exists():return {"status":"missing_features","detail":"Train the historical baseline first."}
    games=pd.read_csv(schedule,dtype={"game_id":str})
    games=games[["game_id","home_team","away_team"]].drop_duplicates("game_id")
    game_lookup={str(g.game_id):(str(g.home_team),str(g.away_team)) for _,g in games.iterrows()}
    records=[];logs=[]
    for year in range(start,end+1):
        datasets={}
        for label,tag,stem in (("offense","espn_cfb_adv_team","adv_team"),("success","espn_cfb_adv_situational","adv_situational")):
            url=f"{ROOT}/{tag}/{stem}_{year}.parquet"
            try:
                r=requests.get(url,timeout=75,headers={"User-Agent":"CFVM-Research/1.0"})
                r.raise_for_status()
                data=pd.read_parquet(io.BytesIO(r.content))
                required={"game_id","pos_team"}
                metric="EPA_per_play" if label=="offense" else "EPA_success_rate"
                if not required.union({metric}).issubset(data.columns):
                    raise ValueError("Missing required columns: "+str(sorted(required.union({metric})-set(data.columns))))
                data["game_id"]=data["game_id"].astype(str)
                datasets[label]=data[["game_id","pos_team",metric]].rename(columns={metric:label})
                logs.append({"season":year,"dataset":label,"status":"ok","rows":len(data),"source":url})
            except Exception as exc:
                logs.append({"season":year,"dataset":label,"status":"error","detail":str(exc)[:180],"source":url})
        if len(datasets)!=2:continue
        merged=datasets["offense"].merge(datasets["success"],on=["game_id","pos_team"],how="inner")
        for _,r in merged.iterrows():
            teams=game_lookup.get(str(r.game_id))
            if not teams:continue
            matched=[t for t in teams if norm(r.pos_team)==norm(t) or norm(r.pos_team).startswith(norm(t)) and len(norm(t))>=4]
            if len(matched)!=1:continue
            records.append({"game_id":str(r.game_id),"team":matched[0],"epa_per_play":r.offense,"success_rate":r.success})
    result={"status":"ok" if records else "no_matched_rows","seasons_requested":[start,end],"matched_team_games":len(records),"logs":logs,"generated_utc":datetime.now(timezone.utc).isoformat()}
    (DATA/"efficiency_download_log.json").write_text(json.dumps(result,indent=2))
    if records:
        pd.DataFrame(records).drop_duplicates(["game_id","team"]).to_csv(DATA/"advanced_team_games.csv",index=False)
        from advanced_features import build
        result["feature_build"]=build()
    return result
