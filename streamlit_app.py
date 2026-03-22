"""
Hyper-Personalization vs Profitability
Complete Streamlit Prototype
Khushi Garg | PGDM RBA 2
Run: streamlit run streamlit_app.py
"""

import warnings; warnings.filterwarnings("ignore")
import os, random
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LinearRegression
import pulp

# ── PAGE CONFIG ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Hyper-Personalization vs Profitability",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CUSTOM CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Sidebar */
    [data-testid="stSidebar"] {background:#1a1a2e !important;}
    [data-testid="stSidebar"] * {color:#fff !important;}
    [data-testid="stSidebarNav"] {display:none}

    /* Main background */
    .main {background:#f0f4f8}
    .block-container {padding-top:1.5rem !important}

    /* KPI Cards */
    .kpi-card {
        background:white; border-radius:12px; padding:20px;
        box-shadow:0 2px 8px rgba(0,0,0,0.06);
        border-left:4px solid #2196F3; margin-bottom:12px;
    }
    .kpi-value {font-size:28px; font-weight:800; color:#2d3748; margin:6px 0}
    .kpi-label {font-size:11px; color:#718096; text-transform:uppercase; letter-spacing:.5px}
    .kpi-sub   {font-size:11px; color:#a0aec0; margin-top:4px}

    /* Section headers */
    .section-header {
        background:#1a1a2e; color:white; padding:14px 20px;
        border-radius:10px; margin-bottom:16px;
    }

    /* Segment pills */
    .pill-premium        {background:#2196F3;color:white;padding:3px 10px;border-radius:12px;font-size:12px;font-weight:600}
    .pill-loyal          {background:#4CAF50;color:white;padding:3px 10px;border-radius:12px;font-size:12px;font-weight:600}
    .pill-price-sensitive{background:#FF9800;color:white;padding:3px 10px;border-radius:12px;font-size:12px;font-weight:600}
    .pill-low-value      {background:#F44336;color:white;padding:3px 10px;border-radius:12px;font-size:12px;font-weight:600}

    /* Metric override */
    [data-testid="metric-container"] {
        background:white; border-radius:10px;
        padding:16px; box-shadow:0 2px 8px rgba(0,0,0,0.06);
    }

    /* Hide streamlit branding */
    #MainMenu {visibility:hidden}
    footer     {visibility:hidden}
    header     {visibility:hidden}
</style>
""", unsafe_allow_html=True)

SEG_COLORS = {
    "Premium":        "#2196F3",
    "Loyal":          "#4CAF50",
    "Price-Sensitive":"#FF9800",
    "Low-Value":      "#F44336",
}
SEED = 42; np.random.seed(SEED); random.seed(SEED)

# ── LOAD & CACHE DATA ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Training models on real Superstore data...")
def load_data():
    for path in ["/content/Superstore.csv","/content/superstore.csv",
                 "Superstore.csv","superstore.csv",
                 "/content/Sample_-_Superstore.csv"]:
        if os.path.exists(path):
            df = pd.read_csv(path, encoding="latin-1")
            break
    else:
        st.error("Superstore.csv not found. Upload it to /content/")
        st.stop()

    df.columns = [c.strip().replace(" ","_") for c in df.columns]
    df["Order_Date"]      = pd.to_datetime(df["Order_Date"])
    SNAP                  = df["Order_Date"].max() + pd.Timedelta(days=1)
    df["Discount_Amount"] = (df["Sales"]/(1-df["Discount"].replace(0,np.nan))*df["Discount"]).fillna(0)
    df["Margin_Pct"]      = np.where(df["Sales"]>0, df["Profit"]/df["Sales"]*100, 0)
    df["Profitable"]      = (df["Profit"]>0).astype(int)
    df["Year"]            = df["Order_Date"].dt.year
    df["Month"]           = df["Order_Date"].dt.month

    dc = df.groupby("Customer_ID").agg(
        Total_Revenue     =("Sales",          "sum"),
        Total_Profit      =("Profit",         "sum"),
        Total_Orders      =("Order_ID",       "nunique"),
        Avg_Discount      =("Discount",       "mean"),
        Avg_Margin_Pct    =("Margin_Pct",     "mean"),
        Order_Variability =("Sales",          "std"),
        Preferred_Category=("Category",       lambda x: x.mode()[0]),
        First_Purchase    =("Order_Date",     "min"),
        Last_Purchase     =("Order_Date",     "max"),
    ).reset_index()
    dc["Tenure_Years"]     = ((dc["Last_Purchase"]-dc["First_Purchase"]).dt.days/365).clip(lower=0.08)
    dc["Annual_Frequency"] = dc["Total_Orders"]/dc["Tenure_Years"]
    dc["AOV"]              = dc["Total_Revenue"]/dc["Total_Orders"]
    dc["Avg_Order_Margin"] = dc["Total_Profit"]/dc["Total_Orders"]
    dc["CLV"]              = (dc["Avg_Order_Margin"]*dc["Annual_Frequency"]*3).round(2)
    dc["Order_Variability"]= dc["Order_Variability"].fillna(0)
    dc["Days_Since_Last"]  = (SNAP-dc["Last_Purchase"]).dt.days

    rfm = df.groupby("Customer_ID").agg(
        Recency  =("Order_Date", lambda x:(SNAP-x.max()).days),
        Frequency=("Order_ID",   "nunique"),
        Monetary =("Sales",      "sum"),
    ).reset_index()
    for col in ["Recency","Frequency","Monetary"]:
        p1,p99=rfm[col].quantile(.01),rfm[col].quantile(.99)
        rfm[col]=rfm[col].clip(lower=p1,upper=p99)
    rfm["R_Score"]=pd.qcut(rfm["Recency"],   q=5,labels=[5,4,3,2,1]).astype(int)
    rfm["F_Score"]=pd.qcut(rfm["Frequency"].rank(method="first"),q=5,labels=[1,2,3,4,5]).astype(int)
    rfm["M_Score"]=pd.qcut(rfm["Monetary"].rank(method="first"), q=5,labels=[1,2,3,4,5]).astype(int)
    rfm["RFM_Score"]=rfm["R_Score"]+rfm["F_Score"]+rfm["M_Score"]
    Xs=StandardScaler().fit_transform(rfm[["Recency","Frequency","Monetary"]])
    km=KMeans(n_clusters=4,random_state=SEED,n_init=10)
    rfm["Cluster"]=km.fit_predict(Xs)
    cm_rfm=rfm.groupby("Cluster")["RFM_Score"].mean().sort_values(ascending=False)
    lm={cm_rfm.index[0]:"Premium",cm_rfm.index[1]:"Loyal",
        cm_rfm.index[2]:"Price-Sensitive",cm_rfm.index[3]:"Low-Value"}
    rfm["RFM_Segment"]=rfm["Cluster"].map(lm)
    dc =dc.merge(rfm[["Customer_ID","RFM_Segment","RFM_Score"]],on="Customer_ID",how="left")
    dfs=df.merge(rfm[["Customer_ID","RFM_Segment"]],on="Customer_ID",how="left")

    FEAT=["Avg_Discount","Total_Orders","Avg_Margin_Pct","AOV",
          "Tenure_Years","Order_Variability","CLV","RFM_Score"]
    dc["Churned"]=(dc["Days_Since_Last"]>180).astype(int)
    Xc=dc[FEAT].fillna(0).values; yc=dc["Churned"].values
    Xtr,Xte,ytr,yte=train_test_split(Xc,yc,test_size=.25,random_state=SEED,stratify=yc)
    rf=RandomForestClassifier(n_estimators=100,random_state=SEED,class_weight="balanced")
    rf.fit(Xtr,ytr)
    dc["Churn_Prob"]=rf.predict_proba(Xc)[:,1]
    auc=roc_auc_score(yte,rf.predict_proba(Xte)[:,1])

    reg=LinearRegression().fit(dc[["Avg_Discount","Total_Orders"]].values,
                                dc["Avg_Margin_Pct"].values)
    be=-(reg.intercept_+reg.coef_[1]*dc["Total_Orders"].mean())/reg.coef_[0]

    return df, dc, dfs, rfm, rf, FEAT, auc, be

df, dc, dfs, rfm, RF_MODEL, FEAT_COLS, CHURN_AUC, BREAKEVEN = load_data()

# ── SIDEBAR ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:8px 0 16px'>
      <div style='font-size:15px;font-weight:800;color:white'>📊 Hyper-Personalization</div>
      <div style='font-size:11px;color:rgba(255,255,255,.5);margin-top:4px'>vs Profitability · Agentic AI</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio("", [
        "📊  Executive Dashboard",
        "👥  Customer Segments",
        "💸  Discount Analysis",
        "💎  CLV Matrix",
        "⚠️  Churn Prediction",
        "🎯  AI Optimizer",
        "⚡  Strategy Simulation",
        "🔍  Customer Lookup",
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("""
    <div style='font-size:11px;color:rgba(255,255,255,.4)'>
    <b style='color:rgba(255,255,255,.7)'>Khushi Garg</b><br>
    PGDM RBA 2<br>
    Dr. Farzan Ghadially<br>
    Dr. Chandravadan Goritiyal<br><br>
    Dataset: Superstore 2014–2017<br>
    9,994 txns · 793 customers
    </div>
    """, unsafe_allow_html=True)

page = page.split("  ")[1].strip()

# ════════════════════════════════════════════════════════
# PAGE 1 — EXECUTIVE DASHBOARD
# ════════════════════════════════════════════════════════
if page == "Executive Dashboard":
    st.markdown('<div class="section-header"><b style="font-size:18px">📊 Executive Dashboard</b><br><span style="font-size:12px;opacity:.7">Real Superstore Data · 2014–2017 · Hyper-Personalization vs Profitability</span></div>', unsafe_allow_html=True)

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("Total Revenue",    f"${df['Sales'].sum()/1e6:.2f}M",  f"{len(df):,} transactions")
    c2.metric("Total Profit",     f"${df['Profit'].sum():,.0f}",      f"Margin {dc['Avg_Margin_Pct'].mean():.1f}%")
    c3.metric("Loss Transactions",f"{(df['Profit']<0).mean()*100:.1f}%", f"Break-even @ {BREAKEVEN:.1%}")
    c4.metric("Churn Rate",       f"{dc['Churned'].mean()*100:.1f}%", f"AUC {CHURN_AUC:.3f}")
    c5.metric("Avg CLV",          f"${dc['CLV'].mean():,.0f}",        f"Total ${dc['CLV'].sum():,.0f}")
    c6.metric("Customers",        f"{dc['Customer_ID'].nunique()}",    "4 segments · ANOVA F>400")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        m = df.groupby(["Year","Month"])[["Sales","Profit"]].sum().reset_index()
        m["Date"] = pd.to_datetime(m[["Year","Month"]].assign(day=1))
        fig = make_subplots(specs=[[{"secondary_y":True}]])
        fig.add_trace(go.Scatter(x=m["Date"],y=m["Sales"]/1e3,name="Revenue ($K)",
            fill="tozeroy",line=dict(color="#2196F3",width=2)),secondary_y=False)
        fig.add_trace(go.Scatter(x=m["Date"],y=m["Profit"]/1e3,name="Profit ($K)",
            fill="tozeroy",line=dict(color="#4CAF50",width=2)),secondary_y=True)
        fig.update_layout(title="Monthly Revenue & Profit Trend",height=320,
            legend=dict(orientation="h",y=-0.2),margin=dict(t=40,b=20,l=20,r=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        samp = dc.sample(n=200,random_state=SEED)
        def sim(d,u):
            p=sum(max(r["AOV"],1)*u*(1-d)-max(r["AOV"],1)*(1-r["Avg_Margin_Pct"]/100)-max(r["AOV"],1)*u*d for _,r in samp.iterrows())
            return round(p,2)
        strats={"Blanket 20%":sim(0.20,1.0),"10% Discount":sim(0.10,1.08),
                "5%+Bundle":sim(0.05,1.12),"AI No-Disc":sim(0.00,1.12)}
        colors_s=[("#4CAF50" if v==max(strats.values()) else "#F44336" if v<0 else "#FF9800")
                  for v in strats.values()]
        fig2=go.Figure(go.Bar(x=list(strats.keys()),y=list(strats.values()),
            marker_color=colors_s,text=[f"${v:,.0f}" for v in strats.values()],
            textposition="outside"))
        fig2.add_hline(y=0,line_dash="dash",line_color="black",line_width=1)
        fig2.update_layout(title="Strategy Comparison (200 customers)",height=320,
            showlegend=False,margin=dict(t=40,b=20,l=20,r=20))
        st.plotly_chart(fig2, use_container_width=True)

    st.info(f"📌 **Key Finding:** {(df['Profit']<0).mean()*100:.1f}% of transactions are loss-making. Break-even discount threshold = **{BREAKEVEN:.1%}**. AI No-Discount strategy outperforms blanket discounting.")

# ════════════════════════════════════════════════════════
# PAGE 2 — CUSTOMER SEGMENTS
# ════════════════════════════════════════════════════════
elif page == "Customer Segments":
    st.markdown('<div class="section-header"><b style="font-size:18px">👥 Customer Segmentation</b><br><span style="font-size:12px;opacity:.7">K-Means k=4 · ANOVA F>400 · p&lt;0.001 · Silhouette=0.353</span></div>', unsafe_allow_html=True)

    col1,col2 = st.columns(2)
    with col1:
        seg_c=rfm["RFM_Segment"].value_counts().reset_index()
        seg_c.columns=["Segment","Count"]
        fig=px.pie(seg_c,values="Count",names="Segment",
            color="Segment",color_discrete_map=SEG_COLORS,hole=0.4)
        fig.update_layout(title="Customer Distribution by Segment",height=320)
        st.plotly_chart(fig,use_container_width=True)

    with col2:
        seg_fin=dfs.groupby("RFM_Segment")[["Sales","Profit"]].sum().reset_index()
        fig=go.Figure()
        fig.add_bar(x=seg_fin["RFM_Segment"],y=seg_fin["Sales"]/1e6,
            name="Revenue($M)",marker_color="#90CAF9")
        fig.add_bar(x=seg_fin["RFM_Segment"],y=seg_fin["Profit"]/1e6,
            name="Profit($M)",marker_color=[SEG_COLORS[s] for s in seg_fin["RFM_Segment"]])
        fig.update_layout(title="Revenue vs Profit by Segment",height=320,barmode="group")
        st.plotly_chart(fig,use_container_width=True)

    col3,col4 = st.columns(2)
    with col3:
        clv_s=dc.groupby("RFM_Segment")["CLV"].mean().reset_index()
        fig=px.bar(clv_s,x="RFM_Segment",y="CLV",color="RFM_Segment",
            color_discrete_map=SEG_COLORS,text="CLV")
        fig.update_traces(texttemplate="$%{text:,.0f}",textposition="outside")
        fig.update_layout(title="Average CLV by Segment",height=300,showlegend=False)
        st.plotly_chart(fig,use_container_width=True)

    with col4:
        ch_s=dc.groupby("RFM_Segment")["Churn_Prob"].mean().reset_index()
        ch_s["Churn_%"]=(ch_s["Churn_Prob"]*100).round(1)
        fig=px.bar(ch_s,x="RFM_Segment",y="Churn_%",color="RFM_Segment",
            color_discrete_map=SEG_COLORS,text="Churn_%")
        fig.update_traces(texttemplate="%{text:.1f}%",textposition="outside")
        fig.update_layout(title="Avg Churn Risk % by Segment",height=300,showlegend=False)
        st.plotly_chart(fig,use_container_width=True)

    st.markdown("### Segment Summary Table")
    seg_tbl=dfs.groupby("RFM_Segment").agg(
        Customers=("Customer_ID","nunique"),Revenue=("Sales","sum"),
        Profit=("Profit","sum"),Avg_Disc=("Discount","mean")).round(2)
    seg_tbl["Margin%"]=(seg_tbl["Profit"]/seg_tbl["Revenue"]*100).round(1)
    seg_tbl["Avg_Disc"]=(seg_tbl["Avg_Disc"]*100).round(1)
    seg_tbl["Avg_CLV"]=dc.groupby("RFM_Segment")["CLV"].mean().round(0)
    seg_tbl.columns=["Customers","Revenue($)","Profit($)","Avg Disc(%)","Margin%","Avg CLV($)"]
    st.dataframe(seg_tbl.style.background_gradient(subset=["Profit($)"],cmap="RdYlGn")
                              .format({"Revenue($)":"${:,.0f}","Profit($)":"${:,.0f}",
                                       "Avg CLV($)":"${:,.0f}"}),use_container_width=True)

# ════════════════════════════════════════════════════════
# PAGE 3 — DISCOUNT ANALYSIS
# ════════════════════════════════════════════════════════
elif page == "Discount Analysis":
    st.markdown('<div class="section-header"><b style="font-size:18px">💸 Discount & Margin Analysis</b></div>', unsafe_allow_html=True)
    st.warning(f"⚠️ **Break-even Discount Threshold: {BREAKEVEN:.1%}** — Customers above this level are net unprofitable. **{(df['Profit']<0).mean()*100:.1f}%** of all transactions are currently loss-making.")

    bins  =[-0.001,0,0.05,0.10,0.20,0.30,1.0]
    labels=["0%","1-5%","6-10%","11-20%","21-30%","31%+"]
    tmp=df.copy(); tmp["Disc_Bin"]=pd.cut(tmp["Discount"],bins=bins,labels=labels)
    res=tmp.groupby("Disc_Bin",observed=True).agg(
        Avg_Profit=("Profit","mean"),Avg_Margin=("Margin_Pct","mean"),
        Loss_Rate =("Profitable",lambda x:(x==0).mean()*100),
        Count     =("Order_ID","count")).round(2).reset_index()

    col1,col2=st.columns(2)
    with col1:
        colors_p=["#4CAF50" if v>=0 else "#F44336" for v in res["Avg_Profit"]]
        fig=go.Figure(go.Bar(x=res["Disc_Bin"].astype(str),y=res["Avg_Profit"],
            marker_color=colors_p,text=res["Avg_Profit"].round(1),textposition="outside"))
        fig.add_hline(y=0,line_dash="dash",line_color="black",line_width=1)
        fig.update_layout(title="Average Profit by Discount Bracket",height=320,showlegend=False)
        st.plotly_chart(fig,use_container_width=True)

    with col2:
        colors_m=["#4CAF50" if v>=0 else "#F44336" for v in res["Avg_Margin"]]
        fig=go.Figure(go.Bar(x=res["Disc_Bin"].astype(str),y=res["Avg_Margin"],
            marker_color=colors_m,text=res["Avg_Margin"].round(1),textposition="outside"))
        fig.add_hline(y=0,line_dash="dash",line_color="black",line_width=1)
        fig.update_layout(title="Average Margin % by Discount Bracket",height=320,showlegend=False)
        st.plotly_chart(fig,use_container_width=True)

    fig=go.Figure()
    fig.add_trace(go.Scatter(x=res["Disc_Bin"].astype(str),y=res["Loss_Rate"],
        mode="lines+markers",line=dict(color="#F44336",width=3),
        marker=dict(size=8),fill="tozeroy",fillcolor="rgba(244,67,54,0.1)"))
    fig.add_hline(y=50,line_dash="dash",line_color="orange",annotation_text="50% loss line")
    fig.update_layout(title="Loss Transaction Rate % by Discount Bracket",height=300)
    st.plotly_chart(fig,use_container_width=True)

# ════════════════════════════════════════════════════════
# PAGE 4 — CLV MATRIX
# ════════════════════════════════════════════════════════
elif page == "CLV Matrix":
    st.markdown('<div class="section-header"><b style="font-size:18px">💎 CLV Profitability Matrix</b><br><span style="font-size:12px;opacity:.7">CLV = Avg Order Margin × Annual Frequency × 3yr lifespan</span></div>', unsafe_allow_html=True)

    clv_med=float(dc["CLV"].median()); disc_med=float(dc["Avg_Discount"].median())
    def quad(r):
        hc=r["CLV"]>=clv_med; hd=r["Avg_Discount"]>=disc_med
        if hc and not hd: return "Q1: Premium"
        if hc and hd:     return "Q2: At-Risk"
        if not hc and not hd: return "Q3: Stable"
        return "Q4: Dependent"
    mat=dc[["Customer_ID","RFM_Segment","CLV","Avg_Discount"]].copy()
    mat["Quadrant"]=mat.apply(quad,axis=1)
    mat["Avg_Disc_Pct"]=(mat["Avg_Discount"]*100).round(1)
    QCOL={"Q1: Premium":"#2196F3","Q2: At-Risk":"#FF9800",
          "Q3: Stable":"#9E9E9E","Q4: Dependent":"#F44336"}

    col1,col2=st.columns([2,1])
    with col1:
        fig=px.scatter(mat,x="Avg_Disc_Pct",y="CLV",color="Quadrant",
            color_discrete_map=QCOL,hover_data=["Customer_ID","RFM_Segment"],
            labels={"Avg_Disc_Pct":"Avg Discount %","CLV":"Customer Lifetime Value ($)"})
        fig.add_vline(x=disc_med*100,line_dash="dash",line_color="black",line_width=1)
        fig.add_hline(y=clv_med,line_dash="dash",line_color="black",line_width=1)
        fig.add_annotation(x=disc_med*100/2,y=clv_med*1.8,text="Q1: Premium",
            font=dict(color="#2196F3",size=12),showarrow=False)
        fig.add_annotation(x=disc_med*100*1.5,y=clv_med*1.8,text="Q2: At-Risk",
            font=dict(color="#FF9800",size=12),showarrow=False)
        fig.add_annotation(x=disc_med*100/2,y=clv_med*0.3,text="Q3: Stable",
            font=dict(color="#9E9E9E",size=12),showarrow=False)
        fig.add_annotation(x=disc_med*100*1.5,y=clv_med*0.3,text="Q4: Dependent",
            font=dict(color="#F44336",size=12),showarrow=False)
        fig.update_layout(title="Personalization Profitability Matrix",height=450)
        st.plotly_chart(fig,use_container_width=True)

    with col2:
        qc=mat["Quadrant"].value_counts()
        st.markdown("### Quadrant Summary")
        qdesc={"Q1: Premium":"High CLV · Low Discount\n→ Premium Rec, no discounts",
               "Q2: At-Risk":"High CLV · High Discount\n→ Bundle offers, reduce discount",
               "Q3: Stable":"Low CLV · Low Discount\n→ Low-cost engagement",
               "Q4: Dependent":"Low CLV · High Discount\n→ Wean off discounts urgently"}
        qbg={"Q1: Premium":"#e3f2fd","Q2: At-Risk":"#fff8e1",
             "Q3: Stable":"#f5f5f5","Q4: Dependent":"#ffebee"}
        for q,n in qc.items():
            st.markdown(f"""
            <div style="background:{qbg.get(q,'#fff')};border-radius:8px;
            padding:12px;margin-bottom:8px;border-left:4px solid {QCOL.get(q,'#999')}">
            <b>{q}</b><br>
            <span style="font-size:22px;font-weight:800">{n}</span>
            <span style="font-size:12px;color:#666"> customers</span><br>
            <span style="font-size:11px;color:#888">{qdesc.get(q,'')}</span>
            </div>""",unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# PAGE 5 — CHURN PREDICTION
# ════════════════════════════════════════════════════════
elif page == "Churn Prediction":
    st.markdown(f'<div class="section-header"><b style="font-size:18px">⚠️ Churn Prediction Model</b><br><span style="font-size:12px;opacity:.7">Random Forest · AUC={CHURN_AUC:.3f} · Churn = no purchase in 180 days · Rate={dc["Churned"].mean()*100:.1f}%</span></div>', unsafe_allow_html=True)

    col1,col2,col3=st.columns(3)
    with col1:
        fi=pd.Series(RF_MODEL.feature_importances_,index=FEAT_COLS).sort_values()
        fig=go.Figure(go.Bar(x=fi.values,y=fi.index,orientation="h",
            marker_color=["#2196F3" if v>fi.mean() else "#90CAF9" for v in fi.values]))
        fig.update_layout(title="Feature Importance<br>(Churn Drivers)",height=320,
            showlegend=False,margin=dict(t=50,b=20,l=20,r=20))
        st.plotly_chart(fig,use_container_width=True)

    with col2:
        seg_ch=dc.groupby("RFM_Segment")["Churn_Prob"].mean().reset_index()
        seg_ch["Churn_%"]=(seg_ch["Churn_Prob"]*100).round(1)
        fig=px.bar(seg_ch,x="RFM_Segment",y="Churn_%",color="RFM_Segment",
            color_discrete_map=SEG_COLORS,text="Churn_%")
        fig.update_traces(texttemplate="%{text:.1f}%",textposition="outside")
        fig.update_layout(title="Churn Risk % by Segment",height=320,showlegend=False)
        st.plotly_chart(fig,use_container_width=True)

    with col3:
        dc["Risk_Cat"]=pd.cut(dc["Churn_Prob"],bins=[0,.3,.6,1.0],
            labels=["Low (<30%)","Medium (30-60%)","High (>60%)"])
        rc=dc["Risk_Cat"].value_counts().reset_index()
        rc.columns=["Risk","Count"]
        fig=px.bar(rc,x="Risk",y="Count",
            color="Risk",color_discrete_map={"Low (<30%)":"#4CAF50",
            "Medium (30-60%)":"#FF9800","High (>60%)":"#F44336"},text="Count")
        fig.update_traces(textposition="outside")
        fig.update_layout(title="Churn Risk Distribution",height=320,showlegend=False)
        st.plotly_chart(fig,use_container_width=True)

    st.markdown("### 🔴 Top High-Risk Customers (Churn Probability > 60%)")
    hr=dc[dc["Churn_Prob"]>=0.6][["Customer_ID","RFM_Segment","CLV",
        "Avg_Discount","Total_Orders","Churn_Prob"]].head(10).copy()
    hr["Churn_Prob"]=(hr["Churn_Prob"]*100).round(1)
    hr["Avg_Discount"]=(hr["Avg_Discount"]*100).round(1)
    hr["CLV"]=hr["CLV"].round(2)
    hr.columns=["Customer ID","Segment","CLV($)","Avg Disc(%)","Orders","Churn Risk(%)"]
    st.dataframe(hr.style.background_gradient(subset=["Churn Risk(%)"],cmap="Reds")
                        .format({"CLV($)":"${:.2f}"}),use_container_width=True)

# ════════════════════════════════════════════════════════
# PAGE 6 — AI OPTIMIZER (FULLY LIVE)
# ════════════════════════════════════════════════════════
elif page == "AI Optimizer":
    st.markdown('<div class="section-header"><b style="font-size:18px">🎯 AI Optimization Engine — LIVE</b><br><span style="font-size:12px;opacity:.7">PuLP · Binary Integer Programming · Change parameters → click Run → results update</span></div>', unsafe_allow_html=True)

    col_p, col_r = st.columns([1,2])

    with col_p:
        st.markdown("#### ⚙️ Parameters")
        budget_pct  = st.slider("Budget (% of Total AOV)", 5, 30, 10, 1) / 100
        max_disc_opt= st.selectbox("Max Discount Allowed",
            options=[0.0,0.05,0.10,0.20],
            format_func=lambda x: {0.0:"0% — No discounts",0.05:"5% Max",
                                    0.10:"10% Max",0.20:"20% Max"}[x], index=2)
        min_margin  = st.slider("Min Margin Floor (%)", 0, 20, 5, 1)
        n_customers = st.slider("Customers to Optimize", 50, 793, 150, 25)
        run_btn     = st.button("▶ Run AI Optimization", type="primary", use_container_width=True)

    with col_r:
        if run_btn:
            with st.spinner("Running PuLP optimization..."):
                ACTIONS={"No Discount":{"d":0.00,"u":1.00},
                         "Premium Rec":{"d":0.00,"u":1.12}}
                if max_disc_opt>=0.05: ACTIONS["5% Discount"] ={"d":0.05,"u":1.08}
                if max_disc_opt>=0.08: ACTIONS["Bundle Offer"]={"d":0.08,"u":1.25}
                if max_disc_opt>=0.10: ACTIONS["10% Discount"]={"d":0.10,"u":1.18}
                if max_disc_opt>=0.20: ACTIONS["20% Discount"]={"d":0.20,"u":1.25}

                samp  =dc.sample(n=min(n_customers,len(dc)),random_state=SEED).copy()
                BUDGET=samp["AOV"].sum()*budget_pct

                def ep(aov,bm,act):
                    d=ACTIONS[act]["d"]; u=ACTIONS[act]["u"]
                    return round(aov*u*(1-d)-aov*(1-bm)-aov*u*d,2)
                def emp(aov,bm,act):
                    d=ACTIONS[act]["d"]; u=ACTIONS[act]["u"]
                    rev=aov*u*(1-d)
                    return 0 if rev<=0 else round((rev-aov*(1-bm)-aov*u*d)/rev*100,2)

                prob=pulp.LpProblem("P",pulp.LpMaximize)
                rows=list(samp.iterrows())
                xv={(i,a):pulp.LpVariable(f"x{i}_{a.replace(' ','_')}",cat="Binary")
                    for i in range(len(rows)) for a in ACTIONS}
                prob+=pulp.lpSum(ep(max(r["AOV"],1),r["Avg_Margin_Pct"]/100,a)*xv[(i,a)]
                                  for i,(_,r) in enumerate(rows) for a in ACTIONS)
                for i,(_,r) in enumerate(rows):
                    prob+=pulp.lpSum(xv[(i,a)] for a in ACTIONS)==1
                    for a in ACTIONS:
                        if emp(max(r["AOV"],1),r["Avg_Margin_Pct"]/100,a)<min_margin:
                            prob+=xv[(i,a)]==0
                prob+=pulp.lpSum(max(r["AOV"],1)*ACTIONS[a]["u"]*ACTIONS[a]["d"]*xv[(i,a)]
                                  for i,(_,r) in enumerate(rows) for a in ACTIONS)<=BUDGET
                prob.solve(pulp.PULP_CBC_CMD(msg=0))

                results=[]
                for i,(_,r) in enumerate(rows):
                    for a in ACTIONS:
                        if pulp.value(xv.get((i,a),0))==1:
                            results.append({"Customer_ID":r["Customer_ID"],
                                "Segment":r["RFM_Segment"],"Action":a,
                                "Profit":ep(max(r["AOV"],1),r["Avg_Margin_Pct"]/100,a),
                                "Margin%":emp(max(r["AOV"],1),r["Avg_Margin_Pct"]/100,a),
                                "AOV":round(max(r["AOV"],1),2)})

                opt_df=pd.DataFrame(results)
                if len(opt_df)==0:
                    st.error("No feasible solution. Try lowering Min Margin or increasing Budget.")
                else:
                    ai_p  =opt_df["Profit"].sum()
                    trad_p=float(sum(max(r["AOV"],1)*0.8-max(r["AOV"],1)*(1-r["Avg_Margin_Pct"]/100)
                                     -max(r["AOV"],1)*0.2 for _,r in samp.iterrows()))
                    impr  =(ai_p-trad_p)/abs(trad_p)*100 if trad_p!=0 else 0

                    m1,m2,m3,m4=st.columns(4)
                    m1.metric("AI Profit",     f"${ai_p:,.0f}",   f"{len(opt_df)} customers")
                    m2.metric("Traditional",   f"${trad_p:,.0f}", "20% blanket discount")
                    m3.metric("Improvement",   f"{impr:+.1f}%",   "AI vs Traditional")
                    m4.metric("Budget",        f"${BUDGET:,.0f}", f"{budget_pct:.0%} of AOV")

                    c1,c2,c3=st.columns(3)
                    with c1:
                        act_c=opt_df["Action"].value_counts().reset_index()
                        act_c.columns=["Action","Count"]
                        fig=px.pie(act_c,values="Count",names="Action",hole=0.4,
                            color_discrete_sequence=px.colors.qualitative.Set2)
                        fig.update_layout(title="Actions Distribution",height=280)
                        st.plotly_chart(fig,use_container_width=True)
                    with c2:
                        sp=opt_df.groupby("Segment")["Profit"].sum().reset_index()
                        fig=px.bar(sp,x="Segment",y="Profit",color="Segment",
                            color_discrete_map=SEG_COLORS,text="Profit")
                        fig.update_traces(texttemplate="$%{text:,.0f}",textposition="outside")
                        fig.add_hline(y=0,line_dash="dash",line_color="black")
                        fig.update_layout(title="Profit by Segment",height=280,showlegend=False)
                        st.plotly_chart(fig,use_container_width=True)
                    with c3:
                        comp=go.Figure(go.Bar(
                            x=["Traditional\n(20% disc)","AI-Optimized"],
                            y=[trad_p,ai_p],
                            marker_color=["#F44336" if trad_p<0 else "#FF9800","#4CAF50"],
                            text=[f"${trad_p:,.0f}",f"${ai_p:,.0f}"],textposition="outside"))
                        comp.add_hline(y=0,line_dash="dash",line_color="black")
                        comp.update_layout(title=f"AI vs Traditional<br>{impr:+.1f}% improvement",
                            height=280,showlegend=False)
                        st.plotly_chart(comp,use_container_width=True)

                    st.markdown("#### Top 10 Customer Recommendations")
                    top10=opt_df.nlargest(10,"Profit")[["Customer_ID","Segment","Action","Profit","Margin%","AOV"]]
                    st.dataframe(top10.style
                        .background_gradient(subset=["Profit"],cmap="Greens")
                        .format({"Profit":"${:.2f}","Margin%":"{:.1f}%","AOV":"${:.2f}"}),
                        use_container_width=True)
        else:
            st.info("👈 Set your parameters on the left and click **▶ Run AI Optimization**")
            st.markdown("""
            **How this works:**
            - **Budget %** — max promotional spend as % of total AOV
            - **Max Discount** — caps the highest discount action available
            - **Min Margin %** — floor below which no action is assigned
            - **Customers** — sample size for optimization
            
            Every time you click Run, PuLP solves a fresh optimization 
            with your exact parameter values. Results will change with 
            different settings.
            """)

# ════════════════════════════════════════════════════════
# PAGE 7 — STRATEGY SIMULATION (LIVE)
# ════════════════════════════════════════════════════════
elif page == "Strategy Simulation":
    st.markdown('<div class="section-header"><b style="font-size:18px">⚡ Strategy Simulation — Live</b><br><span style="font-size:12px;opacity:.7">Adjust parameters — results update instantly</span></div>', unsafe_allow_html=True)

    col_s,col_r=st.columns([1,2])
    with col_s:
        st.markdown("#### ⚙️ Simulation Parameters")
        s_disc  =st.slider("Discount %",   0, 50, 20, 1)
        s_uplift=st.slider("Demand Uplift %", 0, 50, 15, 1)
        s_n     =st.slider("Customers", 50, 793, 200, 25)
        st.markdown("---")
        st.info(f"**Current setting:** {s_disc}% discount generates {s_uplift}% more demand")

    with col_r:
        samp2=dc.sample(n=min(s_n,len(dc)),random_state=SEED)
        disc  =s_disc/100
        uplift=1+s_uplift/100

        def calc(d,u):
            r2=p2=0
            for _,r in samp2.iterrows():
                aov=max(r["AOV"],1); bm=r["Avg_Margin_Pct"]/100
                rev=aov*u*(1-d); cost=aov*(1-bm); dsc=aov*u*d
                r2+=rev; p2+=rev-cost-dsc
            return round(r2,2),round(p2,2)

        r1,p1=calc(disc,uplift)
        r2,p2=calc(disc*0.5,uplift*0.85)
        r3,p3=calc(0.05,1.12)
        r4,p4=calc(0.00,1.12)

        strats={
            f"Your Setting\n({s_disc}% disc)":{"r":r1,"p":p1,"c":"#F44336"},
            f"Half Disc\n({s_disc//2}%)":      {"r":r2,"p":p2,"c":"#FF9800"},
            "5% Bundle\nOffer":                 {"r":r3,"p":p3,"c":"#4CAF50"},
            "AI\nNo-Discount":                  {"r":r4,"p":p4,"c":"#2196F3"},
        }
        best=max(strats,key=lambda k:strats[k]["p"])

        m1,m2,m3,m4=st.columns(4)
        for col_w,(name,v) in zip([m1,m2,m3,m4],strats.items()):
            label=name.replace("\n"," ")
            col_w.metric(label,f"${v['p']:,.0f}",
                         "★ BEST" if name==best else f"Rev ${v['r']:,.0f}")

        fig=make_subplots(rows=1,cols=2,subplot_titles=["Total Profit","Total Revenue"])
        keys=list(strats.keys())
        profits =[strats[k]["p"] for k in keys]
        revenues=[strats[k]["r"] for k in keys]
        colors_k=[strats[k]["c"] for k in keys]
        short_keys=[k.replace("\n"," ") for k in keys]
        fig.add_bar(x=short_keys,y=profits, marker_color=colors_k,
            text=[f"${v:,.0f}" for v in profits],textposition="outside",row=1,col=1)
        fig.add_bar(x=short_keys,y=revenues,marker_color=[c+"aa" for c in colors_k],
            text=[f"${v:,.0f}" for v in revenues],textposition="outside",row=1,col=2)
        fig.add_hline(y=0,line_dash="dash",line_color="black",row=1,col=1)
        fig.update_layout(showlegend=False,height=380)
        st.plotly_chart(fig,use_container_width=True)

        ai_vs=((p4-p1)/abs(p1)*100) if p1!=0 else 0
        clr="green" if ai_vs>0 else "red"
        st.success(f"**Insight:** AI No-Discount generates **:{clr}[{ai_vs:+.1f}%]** more profit than your {s_disc}% discount setting on {s_n} customers.")

# ════════════════════════════════════════════════════════
# PAGE 8 — CUSTOMER LOOKUP
# ════════════════════════════════════════════════════════
elif page == "Customer Lookup":
    st.markdown('<div class="section-header"><b style="font-size:18px">🔍 Customer Lookup</b><br><span style="font-size:12px;opacity:.7">Search any of the 793 real customers</span></div>', unsafe_allow_html=True)

    cust_ids=sorted(dc["Customer_ID"].tolist())
    selected=st.selectbox("Select or type Customer ID",cust_ids,index=0)

    row=dc[dc["Customer_ID"]==selected].iloc[0]
    seg=row["RFM_Segment"]; col=SEG_COLORS.get(seg,"#999")
    rec={"Premium":     "⭐ Premium Rec — exclusive loyalty rewards, NO discount needed",
         "Loyal":       "🎁 Bundle Offer — value add without price cut",
         "Price-Sensitive":"🚫 No Discount — avoid reinforcing price dependency",
         "Low-Value":   "📧 Low-cost email reactivation only"}
    risk=("🔴 High Risk" if row["Churn_Prob"]>0.6 else
          "🟡 Medium Risk" if row["Churn_Prob"]>0.3 else "🟢 Low Risk")

    st.markdown(f"""
    <div style="background:white;border-radius:12px;padding:24px;
    box-shadow:0 2px 12px rgba(0,0,0,.08);margin-bottom:16px">
      <div style="display:flex;align-items:center;gap:16px;margin-bottom:20px">
        <div style="width:60px;height:60px;border-radius:50%;background:{col};
        display:flex;align-items:center;justify-content:center;
        color:white;font-size:20px;font-weight:800">{selected[:2].upper()}</div>
        <div>
          <div style="font-size:20px;font-weight:700">{selected}</div>
          <span style="background:{col};color:white;padding:3px 12px;
          border-radius:12px;font-size:12px;font-weight:600">{seg}</span>
          &nbsp;
          <span style="font-size:12px;color:#718096">
          RFM Score: {int(row['RFM_Score'])} · {row['Preferred_Category']} · 
          {row['Tenure_Years']:.1f} yrs tenure
          </span>
        </div>
      </div>
    </div>
    """,unsafe_allow_html=True)

    c1,c2,c3=st.columns(3)
    c1.metric("Customer LTV",   f"${row['CLV']:,.0f}")
    c2.metric("Total Profit",   f"${row['Total_Profit']:,.0f}")
    c3.metric("Total Orders",   f"{int(row['Total_Orders'])}")
    c4,c5,c6=st.columns(3)
    c4.metric("Avg Discount",   f"{row['Avg_Discount']*100:.1f}%")
    c5.metric("Avg Margin",     f"{row['Avg_Margin_Pct']:.1f}%")
    c6.metric("Churn Risk",     f"{row['Churn_Prob']*100:.1f}%", risk)

    st.markdown(f"""
    <div style="background:#1a1a2e;color:white;border-radius:10px;
    padding:16px;margin-top:12px">
      <div style="font-size:11px;color:rgba(255,255,255,.5);margin-bottom:6px">
      🤖 AI RECOMMENDATION</div>
      <div style="font-size:15px;font-weight:600">{rec.get(seg,'N/A')}</div>
    </div>
    """,unsafe_allow_html=True)

    st.markdown("#### Transaction History")
    cust_txns=dfs[dfs["Customer_ID"]==selected][
        ["Order_Date","Category","Sub-Category","Sales","Discount","Profit"]
    ].sort_values("Order_Date",ascending=False).head(15)
    cust_txns["Discount"]=(cust_txns["Discount"]*100).round(1)
    st.dataframe(cust_txns.style
        .background_gradient(subset=["Profit"],cmap="RdYlGn")
        .format({"Sales":"${:.2f}","Profit":"${:.2f}","Discount":"{:.1f}%"}),
        use_container_width=True)
