"""TheOddsAPI NCAAF odds adapter. API key must live in Streamlit Secrets."""
import requests
import pandas as pd
import re
from datetime import datetime,timezone

def _norm(s):
    return re.sub(r"[^a-z0-9]","",str(s).lower().replace("university","").replace("college",""))

def fetch_odds(predictions,key):
    status={"provider":"TheOddsAPI","status":"not_configured","events":0,"matched_games":0,"moneyline_quotes":0}
    if not key:return pd.DataFrame(),status
    try:
        r=requests.get("https://api.theoddsapi.com/odds/",headers={"x-api-key":key},params={"sport_key":"americanfootball_ncaaf","markets":"h2h","oddsFormat":"american","regions":"us"},timeout=18)
        if r.status_code in (401,403,404):
            status["status"]="access_restricted"
            status["detail"]="NCAAF requires an eligible paid plan; check your provider account."
            return pd.DataFrame(),status
        r.raise_for_status()
        payload=r.json()
        events=payload.get("data",[]) if isinstance(payload,dict) else payload
        if not isinstance(events,list):raise ValueError("Unexpected provider response structure")
        status["events"]=len(events)
        rows=[]
        for event in events:
            home=event.get("home_team","")
            away=event.get("away_team","")
            matches=predictions[(predictions.home_team.map(_norm)==_norm(home))&(predictions.away_team.map(_norm)==_norm(away))]
            if len(matches)!=1:continue
            game=matches.iloc[0]
            for book in event.get("books",[]):
                if book.get("market")!="h2h":continue
                outcomes=book.get("outcomes",[])
                hp=next((o.get("price") for o in outcomes if _norm(o.get("name"))==_norm(home)),None)
                ap=next((o.get("price") for o in outcomes if _norm(o.get("name"))==_norm(away)),None)
                if hp is None or ap is None:continue
                rows.append({"game_id":str(game.game_id),"book":book.get("book","unknown"),"home_american":hp,"away_american":ap,"quote_utc":book.get("updated_at") or datetime.now(timezone.utc).isoformat()})
        status["matched_games"]=len(set(x["game_id"] for x in rows))
        status["moneyline_quotes"]=len(rows)
        status["status"]="ok" if rows else "no_matched_quotes"
        return pd.DataFrame(rows),status
    except requests.HTTPError as exc:
        status["status"]="provider_error"
        status["detail"]=f"HTTP {exc.response.status_code}" if exc.response is not None else "HTTP request failed"
        return pd.DataFrame(),status
    except Exception as exc:
        status["status"]="provider_error"
        status["detail"]=str(exc)[:150]
        return pd.DataFrame(),status
