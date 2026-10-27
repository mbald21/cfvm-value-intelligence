import json
from datetime import datetime, timezone, timedelta
import sqlite3
from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title="CFVM | Value Intelligence", page_icon="🏈", layout="wide")
DATA=Path(__file__).parent/"data"
DATA.mkdir(exist_ok=True)
st.title("🏈 Better Betting with Baldwin")
st.subheader("College Football Value Intelligence")
st.info("CFVM v1.7.1 • Syntax repair")
st.caption("Private research prototype • independently modeled probabilities • not verified betting advice")
st.warning("Research prototype: forecasts refresh when the app is visited after 24 hours; no background scheduler or durable cloud database. Historical profitability is unverified.")
report_path=DATA/"model_report.json"
try:
    previous=json.loads(report_path.read_text()) if report_path.exists() else {}
    last=datetime.fromisoformat(previous.get("generated_utc","2000-01-01T00:00:00+00:00"))
    if last.tzinfo is None:last=last.replace(tzinfo=timezone.utc)
    if datetime.now(timezone.utc)-last>timedelta(hours=24):
        from data_pipeline import download,train
        with st.spinner("Refreshing schedule and forecasts (runs on visits after 24 hours)..."):
            logs=download()
            failures=[x for x in logs if x["status"]=="error"]
            if failures:st.warning("Automatic refresh skipped: source errors for "+", ".join(str(x["year"]) for x in failures))
            else:train()
except Exception as exc:
    st.warning(f"Automatic refresh unavailable: {exc}. Manual refresh remains available.")
with st.sidebar:
    st.header("Model controls")
    st.caption("This starter does not contain the original CFVM v0.3 trained model.")
    if st.button("Download schedules & train baseline",type="primary"):
        from data_pipeline import download, train
        with st.spinner("Downloading historical football data and training a chronological Elo baseline; this may take several minutes..."):
            try:
                logs=download()
                failed=[r for r in logs if r["status"]=="error"]
                if failed: st.error(f"Failed seasons: {', '.join(str(r['year']) for r in failed)}. Training withheld until data is complete.")
                else:
                    report=train()
                    st.success(f"Trained on {report['train_games']:,} completed games.")
                    st.rerun()
            except Exception as exc: st.error(f"Training failed: {exc}")
    log_path=DATA/"download_log.json"
    if log_path.exists():
        with st.expander("Download diagnostics — latest run"):
            st.dataframe(pd.DataFrame(json.loads(log_path.read_text())),hide_index=True,use_container_width=True)
    report_path=DATA/"model_report.json"
    if report_path.exists():
        st.json(json.loads(report_path.read_text()))
    st.divider()
    st.caption("Optional manual odds upload. Public ESPN odds are also checked automatically, but availability is not guaranteed.")
    st.download_button("Download odds CSV template",data="game_id,book,home_american,away_american,quote_utc\n",file_name="cfvm_moneyline_template.csv",mime="text/csv")
    upload=st.file_uploader("Moneyline quotes CSV",type="csv")
    if upload:
        try:
            q=pd.read_csv(upload,dtype={"game_id":str})
            if {"game_id","book","home_american","away_american","quote_utc"}.issubset(q.columns):
                q.to_csv(DATA/"quotes.csv",index=False)
                st.success("Quotes saved on this app instance.")
            else:st.error("Required columns: game_id,book,home_american,away_american,quote_utc")
        except Exception as exc:st.error(str(exc))

pred_path=DATA/"predictions.csv"
pred=pd.read_csv(pred_path,dtype={"game_id":str}) if pred_path.exists() else pd.DataFrame()
quotes_path=DATA/"quotes.csv"
from theodds_feed import fetch_odds
try:
    provider_key=st.secrets.get("THEODDSAPI_KEY","")
except Exception:
    provider_key=""
if provider_key and not pred.empty:
    @st.cache_data(ttl=600,show_spinner=False)
    def fetch_paid_quotes(games_key):
        return fetch_odds(pred,provider_key)
    provider_q,provider_status=fetch_paid_quotes(tuple(pred.game_id.astype(str)))
else:
    provider_q=pd.DataFrame()
    provider_status={"status":"not_configured"}
with st.expander("Dedicated moneyline provider — TheOddsAPI",expanded=True):
    if st.button("Retry TheOddsAPI now",key="retry_paid_odds"):
        st.cache_data.clear()
        st.rerun()
    if provider_status["status"]=="not_configured":
        st.info("Add THEODDSAPI_KEY to Streamlit Secrets to connect. NCAAF requires an eligible provider plan.")
    elif provider_status["status"] in ("access_restricted","access_denied","authentication_failed","endpoint_not_found"):
        st.error(f"TheOddsAPI request failed: {provider_status.get('status')} (HTTP {provider_status.get('http_status','unknown')}). {provider_status.get('detail','')}")
    elif provider_status["status"]=="ok":
        st.success(f"Retrieved {provider_status['moneyline_quotes']} sportsbook quotes across {provider_status['matched_games']} games.")
    else:
        st.info(f"Provider status: {provider_status['status']}. {provider_status.get('detail','')}")
    st.caption(f"Provider HTTP status: {provider_status.get('http_status','not requested')} • Events: {provider_status.get('events',0)} • Matched events: {provider_status.get('matched_events',0)} • Moneyline markets: {provider_status.get('moneyline_markets_seen',0)} • Quoted games: {provider_status.get('matched_games',0)}")
    with st.expander("Provider data diagnostics",expanded=False):
        st.json({k:v for k,v in provider_status.items() if k not in ("api_key","key")})
board=pd.DataFrame()
public_q=pd.DataFrame()
if not pred.empty:
    from odds_feed import fetch_public_odds
    if st.button("🔄 Check ESPN odds now",key="refresh_espn_odds"):
        st.cache_data.clear()
        st.rerun()
    @st.cache_data(ttl=300,show_spinner=False)
    def get_public_quotes(games_key):
        return fetch_public_odds(pred)
    with st.spinner("Checking public college football moneylines..."):
        public_q,odds_diag=get_public_quotes(("espn-daily-v5",datetime.now(timezone.utc).strftime("%Y%m%d"),tuple(pred.game_id.astype(str))))
    st.subheader("Sportsbook odds connection")
    c1,c2,c3=st.columns(3)
    c1.metric("ESPN events found",odds_diag.get("events",0))
    c2.metric("Events with odds entries",odds_diag.get("priced_events",0))
    c3.metric("Matched moneyline games",odds_diag.get("matched_games",0))
    if odds_diag.get("status")=="source_error":
        st.error("ESPN request failed; open technical details below.")
    elif not odds_diag.get("priced_events"):
        st.info("ESPN returned no odds entries in the checked schedule window. No betting prices can be displayed from this feed.")
    elif not odds_diag.get("matched_games"):
        st.info("ESPN returned odds entries, but no complete home/away moneyline pair could be linked to a CFVM forecast.")
    with st.expander("Technical feed diagnostics (optional)",expanded=False):
        st.json(odds_diag)
        st.caption("Unofficial ESPN source. Retrieval time is not necessarily the sportsbook quote time.")
    if odds_diag.get("priced_events",0)>0 and odds_diag.get("offers_with_two_moneylines",0)==0:
        st.info("ESPN has betting odds entries, but none contain complete home-and-away moneylines in the fields currently checked. These may be spread or total markets.")
    elif odds_diag.get("offers_with_two_moneylines",0)>0 and odds_diag.get("matched_games",0)==0:
        st.info("Complete moneyline pairs exist, but they do not match CFVM games yet. See unmatched examples in technical diagnostics.")
    if odds_diag.get("market_samples"):
        st.markdown("**Sample ESPN markets (not moneyline opportunities)**")
        samples=pd.DataFrame(odds_diag["market_samples"])
        st.dataframe(samples,hide_index=True,use_container_width=True)
    m1,m2,m3=st.columns(3)
    m1.metric("Spread offers",odds_diag.get("offers_with_spread",0))
    m2.metric("Total offers",odds_diag.get("offers_with_total",0))
    m3.metric("Complete moneyline offers",odds_diag.get("offers_with_two_moneylines",0))
    st.caption("Spreads and totals cannot be substituted for moneyline odds in expected-value calculations.")
    if not public_q.empty:
        st.success(f"Matched moneylines for {odds_diag['matched_games']} games.")
    else:
        st.warning("No matched public moneylines yet. Check the event and odds counts above. Predictions remain available.")
if not pred.empty and (quotes_path.exists() or not public_q.empty or not provider_q.empty):
    q=pd.read_csv(quotes_path,dtype={"game_id":str}) if quotes_path.exists() else pd.DataFrame()
    if not public_q.empty:q=pd.concat([q,public_q],ignore_index=True)
    if not provider_q.empty:q=pd.concat([q,provider_q],ignore_index=True)
    if {"game_id","book","home_american","away_american","quote_utc"}.issubset(q.columns):
        q["quote_utc"]=pd.to_datetime(q["quote_utc"],utc=True,errors="coerce")
        q=q[q["quote_utc"].notna()].copy()
        joined=pred.merge(q,on="game_id",how="inner")
        st.caption(f"Quote rows with valid timestamps: {len(q)} • Forecast-quote comparisons: {len(joined)}")
        rows=[]
        for _,g in joined.iterrows():
            for side in ("home","away"):
                try:
                    odds=float(g[f"{side}_american"])
                    if -100<odds<100:continue
                    decimal=1+odds/100 if odds>0 else 1+100/abs(odds)
                    prob=float(g.home_win_probability) if side=="home" else 1-float(g.home_win_probability)
                    ev=prob*decimal-1
                    if ev>0:rows.append({"Game":f"{g.away_team} @ {g.home_team}","Pick":g.home_team if side=="home" else g.away_team,"Book":g["book"],"American odds":odds,"Model probability":round(prob,4),"Estimated ROI":round(ev,4),"Quote time (UTC)":g["quote_utc"]})
                except (ValueError,TypeError):continue
        board=pd.DataFrame(rows)
        if joined.empty:
            st.error("No quote rows joined to forecasts. Check game IDs and timestamps.")
        elif board.empty:
            st.warning("Quotes joined, but no positive-EV opportunities passed validation. Inspect odds formats and probability values.")
            with st.expander("Comparison diagnostics"):
                st.dataframe(joined[["away_team","home_team","book","home_american","away_american","home_win_probability"]].head(15),hide_index=True,use_container_width=True)
        if not board.empty:board=board.sort_values("Estimated ROI",ascending=False)
a,b,c=st.columns(3)
a.metric("Upcoming game forecasts",len(pred))
b.metric("Odds comparisons",len(board))
c.metric("Validated profitable strategies",0)
st.subheader("Top 10 model-implied value opportunities")
if board.empty:
    st.warning("No sportsbook comparisons available yet — this is NOT a prediction failure.")
    st.write("Forecasts are available below. ESPN may offer only spreads/totals for these games. A valid moneyline leaderboard requires actual two-sided moneyline prices, not inferred odds.")
else:
    st.dataframe(board.head(10),hide_index=True,use_container_width=True)
    st.caption("Illustrative expected return, not realized ROI. Check quote age, market availability, sportsbook limits and model calibration before acting.")
st.subheader("🏈 Upcoming game forecasts and win probabilities")
if pred.empty:
    st.warning("No upcoming forecasts were generated. This is a schedule/data issue, not proof that there are no upcoming games.")
    st.info("Open Download diagnostics in the sidebar. Check the 2026 row for downloaded status, number of rows, or an error. Then send that row to me.")
else:
    pred["game_date"]=pd.to_datetime(pred["game_date"],utc=True,errors="coerce")
    pred=pred.sort_values("game_date")
    st.subheader("Next scheduled games")
    preview=pred.head(20).copy()
    preview["Home win %"]=(preview["home_win_probability"]*100).round(1)
    preview["Away win %"]=(100-preview["Home win %"]).round(1)
    st.dataframe(preview[["game_date","away_team","home_team","Away win %","Home win %"]],hide_index=True,use_container_width=True)
    search=st.text_input("Search a team")
    matches=pred[pred.home_team.str.contains(search,case=False,na=False)|pred.away_team.str.contains(search,case=False,na=False)] if search else pred
    st.dataframe(matches,hide_index=True,use_container_width=True)
    if len(matches):
        game=st.selectbox("Select a game",options=list(matches.game_id),format_func=lambda gid: (lambda r:f"{r.away_team} @ {r.home_team}")(matches[matches.game_id==gid].iloc[0]))
        conn=sqlite3.connect(DATA/"journal.sqlite")
        conn.execute("CREATE TABLE IF NOT EXISTS notes (game_id TEXT PRIMARY KEY, note TEXT)")
        prior=conn.execute("SELECT note FROM notes WHERE game_id=?",(str(game),)).fetchone()
        note=st.text_area("Research notes",value=prior[0] if prior else "",key=str(game))
        if st.button("Save notes"):
            conn.execute("INSERT OR REPLACE INTO notes(game_id,note) VALUES (?,?)",(str(game),note))
            conn.commit()
            st.success("Saved locally. Notes may be lost when this cloud instance restarts.")
        conn.close()
st.divider()
st.caption("CFVM v0.4 starter — baseline forecasts, not the previously reported v0.3 model. No historical executable ROI has been established.")
