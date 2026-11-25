"""Role-based NSE OHLC forecast web application."""
from __future__ import annotations
import base64, hashlib, hmac, os, secrets, sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

st.set_page_config(page_title="Project 24", page_icon="📈", layout="wide")
FEATURES = ["Open","High","Low","Close","Close_lag1","Close_lag2","Close_lag3","Open_lag1","High_lag1","Low_lag1","Volume","MA5","MA10"]
TARGETS = ["Open","High","Low","Close"]
ALGORITHMS = ["XGBoost", "Gradient Boosting", "Histogram Gradient Boosting", "Ridge Regression"]
INTERVALS = {"1 day (next trading day)":("1d","5y"),"1 hour (next hour)":("1h","730d"),"30 minutes (next bar)":("30m","60d"),"15 minutes (next bar)":("15m","60d"),"5 minutes (next bar)":("5m","60d")}
DB_PATH = Path(os.getenv("APP_DB_PATH", Path(__file__).parent / "data" / "marketforecast.db"))
NSE_LIST = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"

def style():
    background = base64.b64encode((Path(__file__).parent / "assets" / "market-analytics.svg").read_bytes()).decode("ascii")
    st.markdown(f"<style>.stApp{{background-image:linear-gradient(rgba(3,14,29,.68),rgba(3,14,29,.82)),url('data:image/svg+xml;base64,{background}');background-size:cover;background-position:center;background-attachment:fixed}}</style>", unsafe_allow_html=True)
    st.markdown("""<style>
    .stApp{background-color:#061120}
    [data-testid="stHeader"]{background:transparent}.block-container{max-width:1440px;width:100%;padding:2rem clamp(1rem,4vw,4rem)}
    [data-testid="stToolbar"]{visibility:hidden}
    .hero{color:#f6fbff;text-align:center;font-size:clamp(1.8rem,5vw,3.5rem);font-weight:800;margin:1rem}.sub{color:#c2d6e9;text-align:center;margin-bottom:1.4rem}
    [data-testid="stMetric"], [data-testid="stForm"]{background:linear-gradient(145deg,rgba(28,48,75,.97),rgba(8,20,38,.97));border:1px solid rgba(93,173,204,.3);border-radius:18px;padding:1rem;box-shadow:0 12px 28px rgba(0,0,0,.35),inset 0 1px rgba(255,255,255,.09)}
    [data-testid="stDataFrame"]{border:1px solid rgba(93,173,204,.25);border-radius:14px;overflow:hidden;box-shadow:0 12px 28px rgba(0,0,0,.25)}
    @media(max-width:640px){.block-container{padding:1rem .8rem 3rem}[data-testid="stMetric"]{padding:.8rem}.stButton button{width:100%;min-height:44px} [data-testid="stTabs"] button{font-size:.85rem}}
    </style>""", unsafe_allow_html=True)

def now(): return datetime.now(timezone.utc).isoformat(timespec="seconds")
def con():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c=sqlite3.connect(DB_PATH,check_same_thread=False); c.row_factory=sqlite3.Row; return c
def digest(password,salt=None):
    salt=salt or secrets.token_hex(16)
    return salt+"$"+hashlib.pbkdf2_hmac("sha256",password.encode(),salt.encode(),310000).hex()
def valid(password,stored):
    try:
        salt,_=stored.split("$",1); return hmac.compare_digest(digest(password,salt),stored)
    except ValueError: return False

def setup_db():
    with con() as c:
        c.executescript("""CREATE TABLE IF NOT EXISTS accounts(id INTEGER PRIMARY KEY,name TEXT UNIQUE COLLATE NOCASE,email TEXT UNIQUE COLLATE NOCASE,password TEXT,role TEXT,status TEXT,created TEXT,approved TEXT);
        CREATE TABLE IF NOT EXISTS reset_requests(id INTEGER PRIMARY KEY,email TEXT,requested TEXT,done TEXT);
        CREATE TABLE IF NOT EXISTS activity(id INTEGER PRIMARY KEY,account_id INTEGER,username TEXT,event TEXT,stock TEXT,interval TEXT,occurred_at TEXT);""")
        email=os.getenv("ADMIN_EMAIL","").strip().lower(); pwd=os.getenv("ADMIN_PASSWORD","")
        if email and pwd and not c.execute("SELECT 1 FROM accounts WHERE email=?",(email,)).fetchone():
            c.execute("INSERT INTO accounts(name,email,password,role,status,created,approved) VALUES(?,?,?,?,?,?,?)",(os.getenv("ADMIN_NAME","Administrator"),email,digest(pwd),"admin","active",now(),now()))

def signup(name,email,password):
    with con() as c:
        first=c.execute("SELECT COUNT(*) FROM accounts").fetchone()[0]==0
        try:
            c.execute("INSERT INTO accounts(name,email,password,role,status,created,approved) VALUES(?,?,?,?,?,?,?)",(name.strip(),email.strip().lower(),digest(password),"admin" if first else "user","active" if first else "pending",now(),now() if first else None))
            return True, "Your new account has been created. Please sign in." if first else "Your new account has been created and is awaiting administrator approval."
        except sqlite3.IntegrityError: return False,"That name or email address is already registered."

def login(identifier,password):
    with con() as c: r=c.execute("SELECT * FROM accounts WHERE lower(email)=lower(?) OR lower(name)=lower(?)",(identifier,identifier)).fetchone()
    if not r or not valid(password,r["password"]): return None,"Invalid username or password."
    if r["status"]=="pending": return None,"Your account is waiting for administrator approval."
    if r["status"]!="active": return None,"This account is not active."
    return dict(r),None

def log_activity(account,event,stock=None,interval=None):
    with con() as c:
        c.execute("INSERT INTO activity(account_id,username,event,stock,interval,occurred_at) VALUES(?,?,?,?,?,?)",(account["id"],account["name"],event,stock,interval,now()))

@st.cache_data(ttl=86400,show_spinner=False)
def nse_stocks():
    d=pd.read_csv(NSE_LIST); d.columns=[str(x).strip() for x in d.columns]
    d=d.dropna(subset=["SYMBOL","NAME OF COMPANY"]).drop_duplicates("SYMBOL")
    d["ticker"]=d.SYMBOL.astype(str).str.strip()+".NS"; d["label"]=d["NAME OF COMPANY"].astype(str).str.strip()+" — "+d.SYMBOL.astype(str).str.strip()
    return d.sort_values("label")[["ticker","label"]]

def prices(ticker,interval,period):
    d=yf.download(ticker,interval=interval,period=period,auto_adjust=False,progress=False,multi_level_index=False)
    if d.empty: raise ValueError("Yahoo Finance returned no data.")
    return d.rename_axis("Date").reset_index()[["Date","Open","High","Low","Close","Volume"]].dropna()

def features(raw):
    d=raw.copy(); d.Date=pd.to_datetime(d.Date,errors="coerce"); d=d.dropna(subset=["Date"]).sort_values("Date").set_index("Date")
    for col in ["Open","High","Low","Close","Volume"]: d[col]=pd.to_numeric(d[col],errors="coerce")
    for col,n in [("Close_lag1",1),("Close_lag2",2),("Close_lag3",3)]: d[col]=d.Close.shift(n)
    d["Open_lag1"]=d.Open.shift(1); d["High_lag1"]=d.High.shift(1); d["Low_lag1"]=d.Low.shift(1); d["MA5"]=d.Close.rolling(5).mean(); d["MA10"]=d.Close.rolling(10).mean()
    return d[~d.index.duplicated(keep="last")]

def model(name):
    if name=="XGBoost": return XGBRegressor(n_estimators=300,learning_rate=.05,max_depth=5,subsample=.8,colsample_bytree=.8,random_state=42,objective="reg:squarederror",n_jobs=1)
    if name=="Gradient Boosting": return GradientBoostingRegressor(n_estimators=300,learning_rate=.05,max_depth=5,random_state=42)
    if name=="Histogram Gradient Boosting": return HistGradientBoostingRegressor(max_iter=150,learning_rate=.05,max_depth=5,random_state=42)
    if name=="Ridge Regression": return make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    raise ValueError(f"Unknown algorithm: {name}")

def predict(d,name):
    d=d.dropna(subset=FEATURES).copy()
    if len(d)<30: raise ValueError("At least 30 complete bars are needed.")
    for t in TARGETS: d["Target_"+t]=d[t].shift(-1)
    train=d.dropna(subset=["Target_"+t for t in TARGETS]); last=d.iloc[[-1]]; out={}
    for t in TARGETS:
        m=model(name); m.fit(train[FEATURES],train["Target_"+t]); out[t]=float(m.predict(last[FEATURES])[0])
    out["High"]=max(out["High"],out["Open"],out["Close"]); out["Low"]=min(out["Low"],out["Open"],out["Close"])
    return out,d

def validate(d,name):
    d=d.dropna(subset=FEATURES).copy(); rows=[]
    for i in range(max(30,len(d)-6),len(d)-1):
        s=d.iloc[:i+1].copy()
        for t in TARGETS:s["Target_"+t]=s[t].shift(-1)
        train=s.dropna(subset=["Target_"+t for t in TARGETS]); p={}
        for t in TARGETS:
            m=model(name);m.fit(train[FEATURES],train["Target_"+t]);p[t]=float(m.predict(s.iloc[[-1]][FEATURES])[0])
        a=d.iloc[i+1]; rows.append({"Date":a.name,**{"Forecast "+t:p[t] for t in TARGETS},**{"Actual "+t:float(a[t]) for t in TARGETS}})
    r=pd.DataFrame(rows)
    if not r.empty:r["Close deviation %"]=(r["Forecast Close"]-r["Actual Close"]).abs()/r["Actual Close"].replace(0,np.nan)*100
    return r

@st.cache_data(ttl=300, show_spinner=False)
def forecast(ticker,interval,period,forced=None):
    d=features(prices(ticker,interval,period)); results={}
    candidates=[forced] if forced else ALGORITHMS
    with ThreadPoolExecutor(max_workers=min(len(candidates), max(1, os.cpu_count() or 1))) as pool:
        futures={pool.submit(validate,d,name):name for name in candidates}
        for future in as_completed(futures):
            name=futures[future]
            try:
                v=future.result()
                if len(v)==5 and np.isfinite(v["Close deviation %"].mean()): results[name]=v
            except Exception: pass
    if not results: raise ValueError("Not enough history to validate a forecast.")
    best=forced or min(results,key=lambda x:results[x]["Close deviation %"].mean())
    p,history=predict(d,best)
    return p,history,results[best],best,{k:v["Close deviation %"].mean() for k,v in results.items()}

def auth_screen():
    st.markdown("<div class='hero'>◈ PROJECT 24</div><div class='sub'>NSE OHLC forecasting</div>",unsafe_allow_html=True)
    a,b,c=st.tabs(["Sign in","New sign up","Forgot password"])
    with a:
        with st.form("signin"):
            user=st.text_input("Email or username"); password=st.text_input("Password",type="password"); submit=st.form_submit_button("Sign in",use_container_width=True)
        if submit:
            account,error=login(user,password)
            if error:st.error(error)
            else:log_activity(account,"login");st.session_state.account=account;st.rerun()
    with b:
        with st.form("register",clear_on_submit=True):
            name=st.text_input("Name / username"); email=st.text_input("Email ID"); p1=st.text_input("Password",type="password");p2=st.text_input("Reconfirm password",type="password")
            st.markdown("**Terms and risk disclosure**\n\nProject 24 provides market data and model-generated estimates for information and education only. Forecasts are uncertain, may be inaccurate or delayed, and are not investment, financial, tax or legal advice, nor a recommendation to buy, sell or hold any security. Past performance and recent validation results do not guarantee future results. You are responsible for independently verifying data and making your own decisions, including seeking qualified professional advice when appropriate. Trading can result in loss of capital. The service may be interrupted or unavailable; no accuracy or availability is guaranteed. Your account activity and stock searches are recorded for administration and security. By continuing, you acknowledge these risks and accept responsibility for your decisions, subject to applicable law and rights that cannot be waived.")
            accepted=st.checkbox("I have read and agree to the terms and risk disclosure")
            submit=st.form_submit_button("Create account",use_container_width=True)
        if submit:
            if not accepted:st.error("Please accept the terms to create an account.")
            elif len(name.strip())<3 or "@" not in email or len(p1)<8:st.error("Provide a name, valid email, and a password of at least 8 characters.")
            elif p1!=p2:st.error("Passwords do not match.")
            else:
                ok,msg=signup(name,email,p1); (st.success if ok else st.error)(msg)
    with c:
        with st.form("forgot",clear_on_submit=True):
            email=st.text_input("Registered email address");submit=st.form_submit_button("Request password reset",use_container_width=True)
        if submit:
            with con() as db: db.execute("INSERT INTO reset_requests(email,requested) VALUES(?,?)",(email.lower(),now()))
            st.success("Your request has been sent to the administrator.")

def user_screen(account):
    st.title(f"Welcome, {account['name']}")
    st.caption("The system automatically uses the model with the lowest recent validation deviation.")
    try: stock=nse_stocks()
    except Exception as e:st.error(f"Could not load NSE directory: {e}");return
    labels=dict(zip(stock.ticker,stock.label)); one,two=st.columns([2,1])
    with one: chosen=st.multiselect("Select up to 10 NSE stocks",list(labels),format_func=lambda x:labels[x],max_selections=10,placeholder="Search by company or symbol")
    with two: choice=st.selectbox("Forecast interval",list(INTERVALS));go=st.button("Forecast selected stocks",type="primary",use_container_width=True,disabled=not chosen)
    if not go:return
    interval,period=INTERVALS[choice]; output=[]; checks=[]; bar=st.progress(0)
    for i,t in enumerate(chosen,1):
        try:
            log_activity(account,"forecast",labels[t],choice)
            p,h,v,_,_=forecast(t,interval,period);last=h.iloc[-1]
            output.append({"Stock":labels[t],"Forecast Open":p["Open"],"Forecast High":p["High"],"Forecast Low":p["Low"],"Forecast Close":p["Close"],"Expected Close Change %":(p["Close"]-last.Close)/last.Close*100,"Last 5 Days Deviation %":v["Close deviation %"].mean()})
            v.insert(0,"Stock",labels[t]);checks.append(v)
        except Exception as e:output.append({"Stock":labels[t],"Status":"Unavailable: "+str(e)})
        bar.progress(i/len(chosen),text=f"Forecasting {i} of {len(chosen)}")
    bar.empty();st.subheader("Forecast results")
    result_df=pd.DataFrame(output); numeric=[x for x in result_df.columns if x not in {"Stock","Status"}]
    st.dataframe(result_df.style.format({x:"{:.2f}" for x in numeric}),use_container_width=True,hide_index=True)
    if checks:
        st.subheader("Last five days validation")
        check_df=pd.concat(checks,ignore_index=True)
        st.dataframe(check_df.style.format({x:"{:.2f}" for x in check_df.columns if x not in {"Stock","Date"}}),use_container_width=True,hide_index=True)

def admin_screen():
    forecast_tab,accounts_tab,activity_tab,controls_tab=st.tabs(["Forecast workspace","Account approvals","Activity dashboard","Admin controls"])
    with forecast_tab:
        stock=nse_stocks();labels=dict(zip(stock.ticker,stock.label))
        ticker=st.selectbox("NSE stock",list(labels),format_func=lambda x:labels[x]); interval_text=st.selectbox("Forecast interval",list(INTERVALS),key="admin_interval"); selected=st.selectbox("Forecast model",["Automatic best model"]+ALGORITHMS)
        if st.button("Run administrator forecast",type="primary"):
            log_activity(st.session_state.account,"forecast",labels[ticker],interval_text)
            with st.spinner("Running forecast…"):p,h,v,name,scores=forecast(ticker,*INTERVALS[interval_text],None if selected=="Automatic best model" else selected)
            last=h.iloc[-1];st.success(f"Model used: {name}")
            for col,key in zip(st.columns(4),TARGETS):col.metric("Forecast "+key,f"{p[key]:,.2f}")
            st.metric("Expected close change",f"{(p['Close']-last.Close)/last.Close*100:+.2f}%")
            if selected=="Automatic best model":st.dataframe(pd.DataFrame({"Algorithm":scores.keys(),"Close deviation %":scores.values()}).style.format({"Close deviation %":"{:.2f}%"}),hide_index=True)
            fig=go.Figure();fig.add_trace(go.Scatter(x=h.tail(5).index,y=h.tail(5).Close,name="Actual close",mode="lines+markers",line={"color":"#16803a"}));fig.add_trace(go.Scatter(x=v.Date,y=v["Forecast Close"],name="Forecast close",mode="lines+markers",line={"color":"#f58220","dash":"dot"}));st.plotly_chart(fig,use_container_width=True)
            st.subheader("Last five days validation");st.dataframe(v.style.format({x:"{:.2f}" for x in v.columns if x!="Date"}),hide_index=True,use_container_width=True)
    with accounts_tab:
        st.caption("Approve new accounts as users or administrators. Password reset requests appear below.")
        with con() as db: pending=pd.read_sql_query("SELECT id,name,email,created FROM accounts WHERE status='pending' ORDER BY created",db); resets=pd.read_sql_query("SELECT email,requested FROM reset_requests WHERE done IS NULL ORDER BY requested",db)
        if pending.empty:st.info("No accounts are waiting for approval.")
        else:
            st.dataframe(pending[["name","email","created"]],hide_index=True,use_container_width=True); options={f"{r['name']} — {r['email']}":int(r.id) for _,r in pending.iterrows()}
            with st.form("approve"):
                target=st.selectbox("Pending account",list(options));role=st.radio("Approve as",["user","admin"],horizontal=True);approve=st.form_submit_button("Approve");reject=st.form_submit_button("Reject")
            if approve or reject:
                with con() as db:db.execute("UPDATE accounts SET status=?,role=?,approved=? WHERE id=?",("active" if approve else "rejected",role,now(),options[target]))
                st.rerun()
        st.subheader("Password reset requests");st.dataframe(resets if not resets.empty else pd.DataFrame({"Status":["No requests"]}),hide_index=True,use_container_width=True)
        with st.form("reset"):
            email=st.text_input("User email");password=st.text_input("New temporary password",type="password");submit=st.form_submit_button("Set new password")
        if submit:
            with con() as db:
                changed=db.execute("UPDATE accounts SET password=? WHERE lower(email)=lower(?)",(digest(password),email)).rowcount
                db.execute("UPDATE reset_requests SET done=? WHERE lower(email)=lower(?) AND done IS NULL",(now(),email.lower()))
            (st.success if changed else st.error)("Password reset." if changed else "No account found.")
        st.divider()
        st.subheader("Delete a user account")
        st.caption("Deleting an account prevents future sign-ins. Its activity history is retained.")
        with con() as db:
            users=pd.read_sql_query("SELECT id, name, email, status FROM accounts WHERE role='user' ORDER BY name",db)
        if users.empty:
            st.info("There are no user accounts to delete.")
        else:
            user_options={f"{row['name']} — {row['email']} ({row['status']})":int(row['id']) for _,row in users.iterrows()}
            with st.form("delete_user"):
                target_user=st.selectbox("User account",list(user_options))
                confirm_delete=st.checkbox("I understand this will permanently delete this user account")
                delete_user=st.form_submit_button("Delete user account",type="secondary")
            if delete_user:
                if not confirm_delete:
                    st.error("Please confirm the deletion first.")
                else:
                    with con() as db:
                        record=db.execute("SELECT email FROM accounts WHERE id=? AND role='user'",(user_options[target_user],)).fetchone()
                        if record:
                            db.execute("DELETE FROM reset_requests WHERE lower(email)=lower(?)",(record['email'],))
                            db.execute("DELETE FROM accounts WHERE id=? AND role='user'",(user_options[target_user],))
                    if record:
                        st.success("The user account and its password reset requests were deleted. Activity history remains.")
                    else:
                        st.error("That user account is no longer available.")
    with activity_tab:
        st.caption("Recorded sign-ins, explicit sign-outs and forecast requests. Times are UTC. Closing a browser may not record a sign-out.")
        with con() as db: activity=pd.read_sql_query("SELECT username AS User,event AS Event,stock AS Stock,interval AS Interval,occurred_at AS 'Time (UTC)' FROM activity ORDER BY id DESC",db)
        st.dataframe(activity,hide_index=True,use_container_width=True)
    with controls_tab:
        st.caption("Application controls")
        if st.button("Clear data cache"):
            st.cache_data.clear();st.success("Data cache cleared.")
        if st.button("Rerun application"):st.rerun()
        st.caption("Deployment and automatic code reload are managed by the hosting environment. The Streamlit toolbar is hidden for all accounts.")

def main():
    style();setup_db()
    if "account" not in st.session_state:auth_screen();return
    account=st.session_state.account
    with con() as db:
        current=db.execute("SELECT role,status FROM accounts WHERE id=?",(account['id'],)).fetchone()
    if not current or current['status']!='active':
        del st.session_state.account
        st.warning("Your account is no longer active.")
        st.stop()
    account['role']=current['role']
    if st.button("Sign out"):log_activity(account,"logout");del st.session_state.account;st.rerun()
    admin_screen() if account["role"]=="admin" else user_screen(account)
if __name__=="__main__":main()

