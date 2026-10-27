import json
import sqlite3
from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title="CFVM | Value Intelligence", page_icon="🏈", layout="wide")
DATA=Path(__file__).parent/"data"
DATA.mkdir(exist_ok=True)
st.title("🏈 CFVM | Value Intelligence")
st.caption("Dashboard build: v0.5 — October 27, 2026")
st.caption("Private research prototype • independently modeled probabilities • not verified betting advice")
st.warning("Initial prototype: no automated refresh or durable cloud database yet. Historical profitability is unverified.")
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
    upload=st.file_uploader("Optional moneyline quotes CSV",type="csv")
    if upload:
        try:
            q=pd.read_csv(upload,dtype={"game_id":str})
            if {"game_id","book","home_american","away_american"}.issubset(q.columns):
                q.to_csv(DATA/"quotes.csv",index=False)
                st.success("Quotes saved on this app instance.")
            else:st.error("Required columns: game_id,book,home_american,away_american")
        except Exception as exc:st.error(str(exc))

pred_path=DATA/"predictions.csv"
pred=pd.read_csv(pred_path,dtype={"game_id":str}) if pred_path.exists() else pd.DataFrame()
quotes_path=DATA/"quotes.csv"
board=pd.DataFrame()
if not pred.empty and quotes_path.exists():
    q=pd.read_csv(quotes_path,dtype={"game_id":str})
    if {"game_id","book","home_american","away_american"}.issubset(q.columns):
        joined=pred.merge(q,on="game_id",how="inner")
        rows=[]
        for _,g in joined.iterrows():
            for side in ("home","away"):
                try:
                    odds=float(g[f"{side}_american"])
                    if -100<odds<100:continue
                    decimal=1+odds/100 if odds>0 else 1+100/abs(odds)
                    prob=float(g.home_win_probability) if side=="home" else 1-float(g.home_win_probability)
                    ev=prob*decimal-1
                    if ev>0:rows.append({"Game":f"{g.away_team} @ {g.home_team}","Pick":g.home_team if side=="home" else g.away_team,"Book":g["book"],"American odds":odds,"Model probability":round(prob,4),"Estimated ROI":round(ev,4)})
                except (ValueError,TypeError):continue
        board=pd.DataFrame(rows)
        if not board.empty:board=board.sort_values("Estimated ROI",ascending=False)
a,b,c=st.columns(3)
a.metric("Predicted games",len(pred))
b.metric("Positive-EV hypotheses",len(board))
c.metric("Validated profitable strategies",0)
st.subheader("Top 10 pricing discrepancies")
if board.empty:st.info("No comparisons yet. Train the baseline and import independently sourced sportsbook quotes.")
else:
    st.dataframe(board.head(10),hide_index=True,use_container_width=True)
    st.caption("Illustrative model EV only. Verify sportsbook availability, timestamp, and accuracy.")
st.subheader("Game explorer")
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
