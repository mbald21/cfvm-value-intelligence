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
st.info("CFVM v3.1 • One game, one ranked opportunity")
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
with st.expander("Live sportsbook connection · TheOddsAPI",expanded=False):
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
public_q=pd.DataFrame()  # Paid TheOddsAPI is the primary and only automatic odds source.
if not pred.empty and (quotes_path.exists() or not public_q.empty or not provider_q.empty):
    q=pd.read_csv(quotes_path,dtype={"game_id":str}) if quotes_path.exists() else pd.DataFrame()
    if not provider_q.empty:q=pd.concat([q,provider_q],ignore_index=True)
    if {"game_id","book","home_american","away_american","quote_utc"}.issubset(q.columns):
        raw_quote_times=q["quote_utc"].astype(str).copy()
        q["quote_utc"]=pd.to_datetime(q["quote_utc"],utc=True,errors="coerce",format="mixed")
        invalid_times=int(q["quote_utc"].isna().sum())
        st.caption(f"Raw sportsbook quotes: {len(q)} • Invalid quote timestamps: {invalid_times}")
        if invalid_times:
            with st.expander("Invalid timestamp samples"):
                st.write(raw_quote_times[q["quote_utc"].isna()].head(5).tolist())
        q=q[q["quote_utc"].notna()].copy()
        joined=pred.merge(q,on="game_id",how="inner")
        st.caption(f"Quote rows with valid timestamps: {len(q)} • Forecast-quote comparisons: {len(joined)}")
        rows=[]
        skipped_prices=0
        with st.expander("Raw moneyline price diagnostics"):
            st.dataframe(joined[["home_team","away_team","book","home_american","away_american"]].head(12),hide_index=True,use_container_width=True)
        for _,g in joined.iterrows():
            for side in ("home","away"):
                try:
                    odds=float(g[f"{side}_american"])
                    if 1.0<odds<50.0:
                        decimal=odds
                        american=round((odds-1)*100 if odds>=2 else -100/(odds-1))
                    elif odds>=100 or odds<=-100:
                        decimal=1+odds/100 if odds>0 else 1+100/abs(odds)
                        american=odds
                    else:
                        skipped_prices+=1
                        continue
                    prob=float(g.home_win_probability) if side=="home" else 1-float(g.home_win_probability)
                    ev=prob*decimal-1
                    rows.append({"Game ID":str(g.game_id),"Game":f"{g.away_team} @ {g.home_team}","Pick":g.home_team if side=="home" else g.away_team,"Book":g["book"],"American odds":american,"Model probability":round(prob,4),"Estimated ROI":round(ev,4),"Quote time (UTC)":g["quote_utc"]})
                except (ValueError,TypeError):continue
        board=pd.DataFrame(rows)
        st.caption(f"Calculated betting sides: {len(board)} • Skipped invalid prices: {skipped_prices}")
        if joined.empty:
            st.error("No quote rows joined to forecasts. Check game IDs and timestamps.")
        elif board.empty:
            st.warning("Quotes joined, but no valid comparison rows passed validation. Inspect odds formats and probability values.")
            with st.expander("Comparison diagnostics"):
                st.dataframe(joined[["away_team","home_team","book","home_american","away_american","home_win_probability"]].head(15),hide_index=True,use_container_width=True)
        if not board.empty:board=board.sort_values("Estimated ROI",ascending=False)
# The intelligence view intentionally shows unique games, not ten near-duplicate sportsbook quotes.
st.markdown("""
<style>
.block-container {max-width: 1380px; padding-top: 2rem;}
[data-testid="stMetric"] {background: #121e32; border: 1px solid #2b405d; padding: 16px; border-radius: 14px;}
[data-testid="stMetric"] label, [data-testid="stMetric"] div {color: #f1f5fa !important;}
.hero {padding: 30px; border-radius: 20px; background: linear-gradient(120deg,#10233c,#173e57); color: #f7fbff; margin-bottom: 22px;}
.hero h2 {color:#ffffff; font-size:2rem; margin:0 0 10px;}
.hero p {color:#d9e7f2; line-height:1.6;}
.kicker {font-size:.8rem; letter-spacing:.13em; color:#83d6d2; font-weight:700;}
.pick {border:1px solid #2d4961; border-radius:15px; padding:18px; background:#12243a; color:#f5f8fc; min-height:180px;}
.pick .rank {color:#8edbd1; font-size:.8rem; font-weight:700;}
.pick .team {font-size:1.25rem; font-weight:750; margin:10px 0;}
.pick .detail {color:#d0dce9; font-size:.9rem; line-height:1.8;}
.pick .edge {color:#87e4b3; font-weight:750; font-size:1.2rem;}
</style>
""",unsafe_allow_html=True)
st.markdown("""<div class="hero"><div class="kicker">COLLEGE FOOTBALL VALUE INTELLIGENCE</div>
<h2>Where our forecast disagrees with the sportsbook.</h2>
<p>CFVM independently estimates each team's chance of winning from historical results, evolving Elo team strength, and home-field advantage. It then compares that probability against real moneyline prices from TheOddsAPI to identify potential mispricing. The model does not use sportsbook odds to generate its forecasts.</p>
<p><b>01 · Forecast</b> &nbsp; Team strength + venue &nbsp;&nbsp; <b>02 · Compare</b> &nbsp; Live sportsbook prices &nbsp;&nbsp; <b>03 · Explain</b> &nbsp; Win probability, implied probability and expected value</p></div>""",unsafe_allow_html=True)
if not board.empty:
    board=board.copy()
    board["Estimated ROI"]=pd.to_numeric(board["Estimated ROI"],errors="coerce")
    board["Model probability"]=pd.to_numeric(board["Model probability"],errors="coerce")
    board["Market implied probability"]=board["American odds"].apply(lambda o: (100/(float(o)+100)) if float(o)>0 else abs(float(o))/(abs(float(o))+100))
    board["Edge (pp)"]=100*(board["Model probability"]-board["Market implied probability"])
    board["Estimated EV %"]=100*board["Estimated ROI"]
    positive=board[board["Estimated ROI"]>0].sort_values("Estimated ROI",ascending=False)
    top=positive.drop_duplicates(subset=["Game ID"],keep="first").head(10)
else:
    positive=pd.DataFrame()
    top=pd.DataFrame()
k1,k2,k3,k4=st.columns(4)
k1.metric("Upcoming forecasts",len(pred))
k2.metric("Games with moneylines",board["Game"].nunique() if not board.empty else 0)
k3.metric("Positive EV comparisons",len(positive))
k4.metric("Top model-implied EV",f"{100*positive['Estimated ROI'].max():.1f}%" if not positive.empty else "—")
st.caption("A positive EV is a model estimate, not guaranteed profit. Multiple sportsbooks can quote the same game.")
st.markdown("## Top 10 opportunities")
st.caption("One pick per game, using its highest model-implied EV sportsbook quote. Expand a card to compare other books.")
if top.empty:
    st.info("No positive model-implied opportunities are available from current moneylines.")
else:
    feature_path=DATA/"features.csv"
    if feature_path.exists():
        features=pd.read_csv(feature_path,dtype={"game_id":str},usecols=["game_id","elo_diff","home_field"])
        details=pred[["game_id","home_team","away_team"]].merge(features,on="game_id",how="left")
        detail_index={(str(r.away_team),str(r.home_team)):r for _,r in details.iterrows()}
    else:
        detail_index={}
    items=top.to_dict("records")
    for offset in range(0,len(items),2):
        cols=st.columns(2,gap="medium")
        for idx,col in enumerate(cols):
            if offset+idx>=len(items):continue
            r=items[offset+idx]
            rank=offset+idx+1
            odds=f"{int(round(r['American odds'])):+d}"
            with col:
                st.markdown(f"""<div class="pick"><div class="rank">#{rank:02d} · MODEL-IMPLIED VALUE</div>
<div class="team">{r['Pick']}</div>
<div class="detail">{r['Game']}<br>{r['Book']} · Moneyline <b>{odds}</b></div>
<div class="detail">Model win chance: <b>{100*r['Model probability']:.1f}%</b></div>
<div class="edge">Estimated EV {r['Estimated EV %']:+.1f}%</div></div>""",unsafe_allow_html=True)
                with st.expander("Why this pick? · Model and market"):
                    st.write(f"Model win probability: **{100*r['Model probability']:.1f}%**")
                    st.write(f"Sportsbook implied probability (before removing vig): **{100*r['Market implied probability']:.1f}%**")
                    st.write(f"Model minus implied probability: **{r['Edge (pp)']:+.1f} percentage points**")
                    st.write(f"Estimated return per $100 staked: **${100*r['Estimated ROI']:+.2f}** (expectation, not guaranteed)")
                    away,home=r["Game"].split(" @ ",1)
                    detail=detail_index.get((away,home))
                    if detail is not None:
                        if pd.notna(detail.elo_diff):
                            st.write(f"Pre-game Elo strength difference (home minus away): **{detail.elo_diff:+.0f}**")
                        st.write("Venue adjustment: **Home-field advantage**" if detail.home_field==1 else "Venue adjustment: **Neutral site**")
                    st.caption("Current baseline inputs: historical game results, pre-game Elo difference, and home-field indicator. Injuries, weather, rosters and player availability are not modeled.")
                    alternatives=board[board["Game"]==r["Game"]].sort_values("Estimated ROI",ascending=False).head(8)
                    st.dataframe(alternatives[["Pick","Book","American odds","Model probability","Estimated EV %"]],hide_index=True,use_container_width=True)
st.markdown("## All upcoming games")
st.caption("Search the complete forecast schedule; sportsbook prices appear when available.")
st.subheader("Upcoming game forecasts and win probabilities")
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
