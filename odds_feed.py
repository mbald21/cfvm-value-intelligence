"""Best-effort public ESPN scoreboard odds. Unofficial endpoint; coverage varies."""
import re
from datetime import datetime, timedelta, timezone
import pandas as pd
import requests

URL="https://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard"

def norm(s):
    return re.sub(r"[^a-z0-9]","",str(s).lower().replace("university","").replace("college",""))

def _american(value):
    if value is None:return None
    try:
        x=float(str(value).replace("+","").replace("−","-"))
        return x if abs(x)>=100 else None
    except (TypeError,ValueError):return None

def fetch_public_odds(predictions):
    """Return (quotes dataframe, diagnostics). Never invent prices or books."""
    diagnostics={"provider":"ESPN public scoreboard (unofficial)","status":"no_data","events":0,"priced_events":0,"matched_games":0,"retrieved_utc":datetime.now(timezone.utc).isoformat()}
    if predictions.empty:return pd.DataFrame(),diagnostics
    start=datetime.now(timezone.utc).date()
    events=[]
    errors=[]
    for offset in range(0,15):
        day=start+timedelta(days=offset)
        try:
            response=requests.get(URL,params={"dates":f"{day:%Y%m%d}","limit":300},timeout=12,headers={"User-Agent":"CFVM-Research/0.9"})
            response.raise_for_status()
            events.extend(response.json().get("events",[]))
        except Exception as exc:
            errors.append(f"{day}: {str(exc)[:100]}")
    diagnostics["date_requests"]=15
    diagnostics["failed_date_requests"]=len(errors)
    if not events and errors:
        diagnostics["status"]="source_error"
        diagnostics["detail"]="; ".join(errors[:3])
        return pd.DataFrame(),diagnostics
    diagnostics["partial_errors"]=errors[:3]
    events=list({str(e.get("id",i)):e for i,e in enumerate(events)}.values())
    diagnostics["events"]=len(events)
    diagnostics["events_with_odds"]=0
    diagnostics["events_with_both_moneylines"]=0
    diagnostics["matched_by_id"]=0
    diagnostics["matched_by_team_names"]=0
    diagnostics["odds_offers"]=0
    diagnostics["offers_with_two_moneylines"]=0
    diagnostics["offers_missing_moneylines"]=0
    diagnostics["sample_odds_fields"]=[]
    diagnostics["unmatched_example"]=[]
    diagnostics["market_samples"]=[]
    diagnostics["offers_with_spread"]=0
    diagnostics["offers_with_total"]=0
    rows=[]
    for event in events:
        competition=(event.get("competitions") or [{}])[0]
        teams=competition.get("competitors") or []
        home=next((x for x in teams if x.get("homeAway")=="home"),None)
        away=next((x for x in teams if x.get("homeAway")=="away"),None)
        if not home or not away:continue
        offers=competition.get("odds") or []
        if not offers:continue
        diagnostics["priced_events"]+=1
        diagnostics["events_with_odds"]+=1
        def names(t):
            d=t.get("team") or {}
            return [d.get("displayName"),d.get("shortDisplayName"),d.get("name"),d.get("abbreviation")]
        home_names={norm(x) for x in names(home) if x}
        away_names={norm(x) for x in names(away) if x}
        event_id=str(event.get("id",""))
        candidates=predictions[predictions.game_id.astype(str)==event_id]
        if len(candidates)==1:diagnostics["matched_by_id"]+=1
        else:
            candidates=predictions[predictions.home_team.map(norm).isin(home_names)&predictions.away_team.map(norm).isin(away_names)]
            if len(candidates)==1:diagnostics["matched_by_team_names"]+=1
        if len(candidates)!=1 and len(diagnostics["unmatched_example"])<3:
            diagnostics["unmatched_example"].append({"event_id":event_id,"home":(home.get("team") or {}).get("displayName"),"away":(away.get("team") or {}).get("displayName")})
        gid=str(candidates.iloc[0]["game_id"]) if len(candidates)==1 else None
        for offer in offers:
            diagnostics["odds_offers"]+=1
            if offer.get("spread") is not None:diagnostics["offers_with_spread"]+=1
            if offer.get("overUnder") is not None:diagnostics["offers_with_total"]+=1
            if len(diagnostics["market_samples"])<5:
                diagnostics["market_samples"].append({"game":f"{(away.get(chr(116)+chr(101)+chr(97)+chr(109)) or {}).get(chr(100)+chr(105)+chr(115)+chr(112)+chr(108)+chr(97)+chr(121)+chr(78)+chr(97)+chr(109)+chr(101))} @ {(home.get(chr(116)+chr(101)+chr(97)+chr(109)) or {}).get(chr(100)+chr(105)+chr(115)+chr(112)+chr(108)+chr(97)+chr(121)+chr(78)+chr(97)+chr(109)+chr(101))}","spread":offer.get("spread"),"total":offer.get("overUnder"),"details":offer.get("details"),"provider":(offer.get("provider") or {}).get("name") if isinstance(offer.get("provider"),dict) else None})
            if len(diagnostics["sample_odds_fields"])<2:
                diagnostics["sample_odds_fields"].append({"offer_fields":list(offer.keys()),"home_odds_fields":list((offer.get("homeTeamOdds") or {}).keys()),"away_odds_fields":list((offer.get("awayTeamOdds") or {}).keys())})
            h=offer.get("homeTeamOdds") or {}
            a=offer.get("awayTeamOdds") or {}
            home_price=_american(h.get("moneyLine") or h.get("moneyline"))
            away_price=_american(a.get("moneyLine") or a.get("moneyline"))
            if home_price is None or away_price is None:
                diagnostics["offers_missing_moneylines"]+=1
                continue
            diagnostics["offers_with_two_moneylines"]+=1
            if gid is None:continue
            diagnostics["events_with_both_moneylines"]+=1
            provider=offer.get("provider") or {}
            book=provider.get("name") if isinstance(provider,dict) else str(provider)
            rows.append({"game_id":gid,"book":book or "ESPN-listed provider","home_american":home_price,"away_american":away_price,"quote_utc":diagnostics["retrieved_utc"],"source":"ESPN scoreboard (unofficial; snapshot retrieval time)"})
    diagnostics["matched_games"]=len(set(x["game_id"] for x in rows))
    diagnostics["status"]="ok" if rows else "no_matching_moneylines"
    return pd.DataFrame(rows),diagnostics
