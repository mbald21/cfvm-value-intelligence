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
        status["http_status"]=r.status_code
        if r.status_code in (401,403,404):
            status["status"]={401:"authentication_failed",403:"access_denied",404:"endpoint_not_found"}[r.status_code]
            status["detail"]="Check the key, subscription activation, and sport permissions. HTTP "+str(r.status_code)
            return pd.DataFrame(),status
        r.raise_for_status()
        payload=r.json()
        events=payload.get("data",[]) if isinstance(payload,dict) else payload
        if not isinstance(events,list):raise ValueError("Unexpected provider response structure")
        status["events"]=len(events)
        status["sample_event_keys"]=list(events[0].keys())[:25] if events else []
        status["sample_team_names"]=[{"home":e.get("home_team"),"away":e.get("away_team"),"teams":e.get("teams")} for e in events[:3]]
        status["sample_market_keys"]=[]
        rows=[]
        pred_index={}
        for _,g in predictions.iterrows():
            pred_index.setdefault((_norm(g.home_team),_norm(g.away_team)),[]).append(g)
        event_matches=0
        offers_seen=0
        for event in events:
            home=event.get("home_team") or event.get("homeTeam") or ""
            away=event.get("away_team") or event.get("awayTeam") or ""
            teams=event.get("teams") or []
            if (not home or not away) and len(teams)==2:
                home,away=teams[0],teams[1]
            matches=pred_index.get((_norm(home),_norm(away)),[])
            if len(matches)!=1:
                matches=pred_index.get((_norm(away),_norm(home)),[])
                if len(matches)!=1:continue
            event_matches+=1
            game=matches[0]
            books=event.get("books") or event.get("bookmakers") or []
            if isinstance(books,dict):books=list(books.values())
            for book in books:
                markets=book.get("markets") or [book]
                if isinstance(markets,dict):markets=list(markets.values())
                for market in markets:
                    if len(status["sample_market_keys"])<3:
                        status["sample_market_keys"].append({"book_keys":list(book.keys())[:12],"market_keys":list(market.keys())[:12]})
                    market_type=market.get("key") or market.get("market") or market.get("market_key")
                    if market_type not in ("h2h","moneyline","moneylines"):continue
                    offers_seen+=1
                    outcomes=market.get("outcomes") or market.get("odds") or []
                    if isinstance(outcomes,dict):
                        outcomes=[{"name":k,"price":v} for k,v in outcomes.items()]
                    hp=next((o.get("price",o.get("odds")) for o in outcomes if _norm(o.get("name") or o.get("team"))==_norm(game.home_team)),None)
                    ap=next((o.get("price",o.get("odds")) for o in outcomes if _norm(o.get("name") or o.get("team"))==_norm(game.away_team)),None)
                    if hp is None or ap is None:continue
                    rows.append({"game_id":str(game.game_id),"book":book.get("title") or book.get("key") or book.get("book") or "unknown","home_american":hp,"away_american":ap,"quote_utc":market.get("last_update") or book.get("last_update") or book.get("updated_at") or datetime.now(timezone.utc).isoformat()})
        status["matched_events"]=event_matches
        status["moneyline_markets_seen"]=offers_seen
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
