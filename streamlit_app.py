"""
Hyper-Personalization vs Profitability
Complete Streamlit Prototype — Clean Version
Khushi Garg | PGDM RBA 2
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

st.set_page_config(
    page_title="Hyper-Personalization vs Profitability",
    page_icon="📊", layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown("""
<style>
[data-testid="stSidebar"]{background:#1a1a2e !important}
[data-testid="stSidebar"] *{color:#fff !important}
[data-testid="stSidebarNav"]{display:none}
.main{background:#f0f4f8}
.block-container{padding-top:1.5rem !important}
.section-header{background:#1a1a2e;color:white;padding:14px 20px;
  border-radius:10px;margin-bottom:16px}
/* Force dark text in white content areas - targeted */
.stMarkdown p, .stMarkdown div, .stMarkdown span {color:#2d3748}
[data-testid="metric-container"] label {color:#718096 !important}
[data-testid="metric-container"] [data-testid="metric-value"] {color:#2d3748 !important}
#MainMenu{visibility:hidden}footer{visibility:hidden}header{visibility:hidden}
</style>
""", unsafe_allow_html=True)

SEG_COLORS = {"Premium":"#2196F3","Loyal":"#4CAF50","Price-Sensitive":"#FF9800","Low-Value":"#F44336"}
SEED = 42; np.random.seed(SEED); random.seed(SEED)

@st.cache_resource(show_spinner="Loading models — please wait...")
def load_data():
    for path in ["/content/Superstore.csv","/content/superstore.csv",
                 "Superstore.csv","superstore.csv"]:
        if os.path.exists(path):
            df = pd.read_csv(path, encoding="latin-1"); break
    else:
        st.error("Upload Superstore.csv"); st.stop()

    df.columns = [c.strip().replace(" ","_") for c in df.columns]
    df["Order_Date"] = pd.to_datetime(df["Order_Date"])
    SNAP = df["Order_Date"].max() + pd.Timedelta(days=1)
    df["Discount_Amount"] = (df["Sales"]/(1-df["Discount"].replace(0,np.nan))*df["Discount"]).fillna(0)
    df["Margin_Pct"] = np.where(df["Sales"]>0, df["Profit"]/df["Sales"]*100, 0)
    df["Profitable"] = (df["Profit"]>0).astype(int)
    df["Year"] = df["Order_Date"].dt.year
    df["Month"] = df["Order_Date"].dt.month

    dc = df.groupby("Customer_ID").agg(
        Total_Revenue=("Sales","sum"), Total_Profit=("Profit","sum"),
        Total_Orders=("Order_ID","nunique"), Avg_Discount=("Discount","mean"),
        Avg_Margin_Pct=("Margin_Pct","mean"), Order_Variability=("Sales","std"),
        Preferred_Category=("Category",lambda x:x.mode()[0]),
        First_Purchase=("Order_Date","min"), Last_Purchase=("Order_Date","max"),
    ).reset_index()
    dc["Tenure_Years"] = ((dc["Last_Purchase"]-dc["First_Purchase"]).dt.days/365).clip(lower=0.08)
    dc["Annual_Frequency"] = dc["Total_Orders"]/dc["Tenure_Years"]
    dc["AOV"] = dc["Total_Revenue"]/dc["Total_Orders"]
    dc["Avg_Order_Margin"] = dc["Total_Profit"]/dc["Total_Orders"]
    dc["CLV"] = (dc["Avg_Order_Margin"]*dc["Annual_Frequency"]*3).round(2)
    dc["Order_Variability"] = dc["Order_Variability"].fillna(0)
    dc["Days_Since_Last"] = (SNAP-dc["Last_Purchase"]).dt.days

    rfm = df.groupby("Customer_ID").agg(
        Recency=("Order_Date",lambda x:(SNAP-x.max()).days),
        Frequency=("Order_ID","nunique"), Monetary=("Sales","sum")).reset_index()
    for col in ["Recency","Frequency","Monetary"]:
        p1,p99=rfm[col].quantile(.01),rfm[col].quantile(.99)
        rfm[col]=rfm[col].clip(lower=p1,upper=p99)
    rfm["R_Score"]=pd.qcut(rfm["Recency"],q=5,labels=[5,4,3,2,1]).astype(int)
    rfm["F_Score"]=pd.qcut(rfm["Frequency"].rank(method="first"),q=5,labels=[1,2,3,4,5]).astype(int)
    rfm["M_Score"]=pd.qcut(rfm["Monetary"].rank(method="first"),q=5,labels=[1,2,3,4,5]).astype(int)
    rfm["RFM_Score"]=rfm["R_Score"]+rfm["F_Score"]+rfm["M_Score"]
    Xs=StandardScaler().fit_transform(rfm[["Recency","Frequency","Monetary"]])
    km=KMeans(n_clusters=4,random_state=SEED,n_init=10)
    rfm["Cluster"]=km.fit_predict(Xs)
    cm_rfm=rfm.groupby("Cluster")["RFM_Score"].mean().sort_values(ascending=False)
    lm={cm_rfm.index[0]:"Premium",cm_rfm.index[1]:"Loyal",
        cm_rfm.index[2]:"Price-Sensitive",cm_rfm.index[3]:"Low-Value"}
    rfm["RFM_Segment"]=rfm["Cluster"].map(lm)
    dc=dc.merge(rfm[["Customer_ID","RFM_Segment","RFM_Score"]],on="Customer_ID",how="left")
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

# SIDEBAR
with st.sidebar:
    st.markdown("<div style='padding:8px 0 16px'><div style='font-size:15px;font-weight:800'>📊 Hyper-Personalization</div><div style='font-size:11px;color:rgba(255,255,255,.5);margin-top:4px'>vs Profitability · Agentic AI</div></div>", unsafe_allow_html=True)
    page = st.radio("", ["📊  Executive Dashboard","👥  Customer Segments",
        "💸  Discount Analysis","💎  CLV Matrix","⚠️  Churn Prediction",
        "🎯  AI Optimizer","⚡  Strategy Simulation","🔍  Customer Lookup"],
        label_visibility="collapsed")
    st.markdown("---")
    st.markdown("<div style='font-size:11px;color:rgba(255,255,255,.4)'><b style='color:rgba(255,255,255,.7)'>Khushi Garg</b><br>PGDM RBA 2<br>Dr. Farzan Ghadially<br>Dr. Chandravadan Goritiyal<br><br>Superstore 2014-2017<br>9,994 txns · 793 customers</div>", unsafe_allow_html=True)

page = page.split("  ")[1].strip()

# ══ PAGE 1: EXECUTIVE DASHBOARD ══════════════════════════════════
if page == "Executive Dashboard":
    st.markdown('<div class="section-header"><b style="font-size:18px">📊 Executive Dashboard</b><br><span style="font-size:12px;opacity:.7">Real Superstore Data · 2014-2017</span></div>', unsafe_allow_html=True)

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("Total Revenue",    f"${df['Sales'].sum()/1e6:.2f}M", f"{len(df):,} transactions")
    c2.metric("Total Profit",     f"${df['Profit'].sum():,.0f}",    f"Margin {dc['Avg_Margin_Pct'].mean():.1f}%")
    c3.metric("Loss Transactions",f"{(df['Profit']<0).mean()*100:.1f}%", f"Break-even {BREAKEVEN:.1%}")
    c4.metric("Churn Rate",       f"{dc['Churned'].mean()*100:.1f}%", f"AUC {CHURN_AUC:.3f}")
    c5.metric("Avg CLV",          f"${dc['CLV'].mean():,.0f}",      f"Total ${dc['CLV'].sum():,.0f}")
    c6.metric("Customers",        f"{dc['Customer_ID'].nunique()}", "4 segments")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        m = df.groupby(["Year","Month"])[["Sales","Profit"]].sum().reset_index()
        m["Date"] = pd.to_datetime(m[["Year","Month"]].assign(day=1))
        fig = make_subplots(specs=[[{"secondary_y":True}]])
        fig.add_trace(go.Scatter(x=m["Date"],y=m["Sales"]/1e3,name="Revenue($K)",
            fill="tozeroy",line=dict(color="#2196F3",width=2)),secondary_y=False)
        fig.add_trace(go.Scatter(x=m["Date"],y=m["Profit"]/1e3,name="Profit($K)",
            fill="tozeroy",line=dict(color="#4CAF50",width=2)),secondary_y=True)
        fig.update_layout(title="Monthly Revenue & Profit Trend",height=320,
            legend=dict(orientation="h",y=-0.2),margin=dict(t=40,b=20,l=20,r=20))
        st.plotly_chart(fig,use_container_width=True)

    with col2:
        samp=dc.sample(n=200,random_state=SEED)
        def sim(d,u):
            return round(sum(max(r["AOV"],1)*u*(1-d)-max(r["AOV"],1)*(1-r["Avg_Margin_Pct"]/100)
                             -max(r["AOV"],1)*u*d for _,r in samp.iterrows()),2)
        sv={"Blanket 20%":sim(0.20,1.0),"10% Discount":sim(0.10,1.08),
            "5%+Bundle":sim(0.05,1.12),"AI No-Disc":sim(0.00,1.12)}
        bc=["#4CAF50" if v==max(sv.values()) else "#F44336" if v<0 else "#FF9800" for v in sv.values()]
        fig2=go.Figure(go.Bar(x=list(sv.keys()),y=list(sv.values()),
            marker_color=bc,text=[f"${v:,.0f}" for v in sv.values()],textposition="outside"))
        fig2.add_hline(y=0,line_dash="dash",line_color="black",line_width=1)
        fig2.update_layout(title="Strategy Comparison (200 customers)",height=320,
            showlegend=False,margin=dict(t=40,b=20,l=20,r=20))
        st.plotly_chart(fig2,use_container_width=True)
    st.info(f"📌 **Key Finding:** {(df['Profit']<0).mean()*100:.1f}% of transactions are loss-making. Break-even = **{BREAKEVEN:.1%}**. AI No-Discount outperforms all discount strategies.")

# ══ PAGE 2: SEGMENTS ═════════════════════════════════════════════
elif page == "Customer Segments":
    st.markdown('<div class="section-header"><b style="font-size:18px">👥 Customer Segmentation</b><br><span style="font-size:12px;opacity:.7">K-Means k=4 · ANOVA F>400 · p&lt;0.001</span></div>', unsafe_allow_html=True)

    col1,col2=st.columns(2)
    with col1:
        seg_c=rfm["RFM_Segment"].value_counts().reset_index()
        seg_c.columns=["Segment","Count"]
        fig=px.pie(seg_c,values="Count",names="Segment",color="Segment",
            color_discrete_map=SEG_COLORS,hole=0.4)
        fig.update_layout(title="Customer Distribution",height=320)
        st.plotly_chart(fig,use_container_width=True)
    with col2:
        sf=dfs.groupby("RFM_Segment")[["Sales","Profit"]].sum().reset_index()
        fig=go.Figure()
        fig.add_bar(x=sf["RFM_Segment"],y=sf["Sales"]/1e6,name="Revenue($M)",marker_color="#90CAF9")
        fig.add_bar(x=sf["RFM_Segment"],y=sf["Profit"]/1e6,name="Profit($M)",
            marker_color=[SEG_COLORS[s] for s in sf["RFM_Segment"]])
        fig.update_layout(title="Revenue vs Profit by Segment",height=320,barmode="group")
        st.plotly_chart(fig,use_container_width=True)

    col3,col4=st.columns(2)
    with col3:
        clv_s=dc.groupby("RFM_Segment")["CLV"].mean().reset_index()
        fig=px.bar(clv_s,x="RFM_Segment",y="CLV",color="RFM_Segment",
            color_discrete_map=SEG_COLORS,text=clv_s["CLV"].round(0))
        fig.update_traces(texttemplate="$%{text:,.0f}",textposition="outside")
        fig.update_layout(title="Average CLV by Segment",height=300,showlegend=False)
        st.plotly_chart(fig,use_container_width=True)
    with col4:
        ch_s=dc.groupby("RFM_Segment")["Churn_Prob"].mean().reset_index()
        ch_s["Pct"]=(ch_s["Churn_Prob"]*100).round(1)
        fig=px.bar(ch_s,x="RFM_Segment",y="Pct",color="RFM_Segment",
            color_discrete_map=SEG_COLORS,text="Pct")
        fig.update_traces(texttemplate="%{text:.1f}%",textposition="outside")
        fig.update_layout(title="Avg Churn Risk % by Segment",height=300,showlegend=False)
        st.plotly_chart(fig,use_container_width=True)

    st.markdown("### Segment Summary Table")
    st_tbl=dfs.groupby("RFM_Segment").agg(
        Customers=("Customer_ID","nunique"),Revenue=("Sales","sum"),
        Profit=("Profit","sum"),Avg_Disc=("Discount","mean")).round(2)
    st_tbl["Margin_Pct"]=(st_tbl["Profit"]/st_tbl["Revenue"]*100).round(1)
    st_tbl["Avg_Disc"]=(st_tbl["Avg_Disc"]*100).round(1)
    st_tbl["Avg_CLV"]=dc.groupby("RFM_Segment")["CLV"].mean().round(0)
    st_tbl.columns=["Customers","Revenue($)","Profit($)","Avg Disc(%)","Margin%","Avg CLV($)"]
    st.dataframe(st_tbl.style.format({
        "Revenue($)":"${:,.0f}","Profit($)":"${:,.0f}","Avg CLV($)":"${:,.0f}",
        "Avg Disc(%)":"{:.1f}%","Margin%":"{:.1f}%"}),use_container_width=True)

# ══ PAGE 3: DISCOUNT ANALYSIS ═════════════════════════════════════
elif page == "Discount Analysis":
    st.markdown('<div class="section-header"><b style="font-size:18px">💸 Discount & Margin Analysis</b></div>', unsafe_allow_html=True)
    st.warning(f"⚠️ **Break-even: {BREAKEVEN:.1%}** — Above this, customers are net unprofitable. **{(df['Profit']<0).mean()*100:.1f}%** of transactions are loss-making.")

    bins=[-0.001,0,0.05,0.10,0.20,0.30,1.0]
    labels=["0%","1-5%","6-10%","11-20%","21-30%","31%+"]
    tmp=df.copy(); tmp["Disc_Bin"]=pd.cut(tmp["Discount"],bins=bins,labels=labels)
    res=tmp.groupby("Disc_Bin",observed=True).agg(
        Avg_Profit=("Profit","mean"),Avg_Margin=("Margin_Pct","mean"),
        Loss_Rate=("Profitable",lambda x:(x==0).mean()*100),
        Count=("Order_ID","count")).round(2).reset_index()

    col1,col2=st.columns(2)
    with col1:
        cp=["#4CAF50" if v>=0 else "#F44336" for v in res["Avg_Profit"]]
        fig=go.Figure(go.Bar(x=res["Disc_Bin"].astype(str),y=res["Avg_Profit"],
            marker_color=cp,text=res["Avg_Profit"].round(1),textposition="outside"))
        fig.add_hline(y=0,line_dash="dash",line_color="black",line_width=1)
        fig.update_layout(title="Average Profit by Discount Bracket",height=320,showlegend=False)
        st.plotly_chart(fig,use_container_width=True)
    with col2:
        cm2=["#4CAF50" if v>=0 else "#F44336" for v in res["Avg_Margin"]]
        fig=go.Figure(go.Bar(x=res["Disc_Bin"].astype(str),y=res["Avg_Margin"],
            marker_color=cm2,text=res["Avg_Margin"].round(1),textposition="outside"))
        fig.add_hline(y=0,line_dash="dash",line_color="black",line_width=1)
        fig.update_layout(title="Average Margin % by Discount Bracket",height=320,showlegend=False)
        st.plotly_chart(fig,use_container_width=True)

    fig=go.Figure()
    fig.add_trace(go.Scatter(x=res["Disc_Bin"].astype(str),y=res["Loss_Rate"],
        mode="lines+markers",line=dict(color="#F44336",width=3),
        marker=dict(size=8),fill="tozeroy",fillcolor="rgba(244,67,54,0.1)"))
    fig.update_layout(title="Loss Transaction Rate % by Discount Bracket",height=300)
    st.plotly_chart(fig,use_container_width=True)

# ══ PAGE 4: CLV MATRIX ════════════════════════════════════════════
elif page == "CLV Matrix":
    st.markdown('<div class="section-header"><b style="font-size:18px">💎 CLV Profitability Matrix</b></div>', unsafe_allow_html=True)

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
    QCOL={"Q1: Premium":"#2196F3","Q2: At-Risk":"#FF9800","Q3: Stable":"#9E9E9E","Q4: Dependent":"#F44336"}

    col1,col2=st.columns([2,1])
    with col1:
        fig=px.scatter(mat,x="Avg_Disc_Pct",y="CLV",color="Quadrant",
            color_discrete_map=QCOL,hover_data=["Customer_ID","RFM_Segment"],
            labels={"Avg_Disc_Pct":"Avg Discount %","CLV":"Customer Lifetime Value ($)"})
        fig.add_vline(x=disc_med*100,line_dash="dash",line_color="black",line_width=1)
        fig.add_hline(y=clv_med,line_dash="dash",line_color="black",line_width=1)
        fig.update_layout(title="Personalization Profitability Matrix",height=450)
        st.plotly_chart(fig,use_container_width=True)
    with col2:
        qc=mat["Quadrant"].value_counts()
        st.markdown("### Quadrant Summary")
        qdesc={"Q1: Premium":"High CLV · Low Discount → Premium Rec",
               "Q2: At-Risk":"High CLV · High Discount → Bundle offers",
               "Q3: Stable":"Low CLV · Low Discount → Monitor",
               "Q4: Dependent":"Low CLV · High Discount → Wean off discounts"}
        qbg={"Q1: Premium":"#e3f2fd","Q2: At-Risk":"#fff8e1","Q3: Stable":"#f5f5f5","Q4: Dependent":"#ffebee"}
        for q,n in qc.items():
            bg=qbg.get(q,"#fff"); bc=QCOL.get(q,"#999"); desc=qdesc.get(q,"")
            st.markdown(f'''<div style="background:{bg};border-radius:8px;padding:14px;
            margin-bottom:10px;border-left:5px solid {bc}">
            <div style="font-size:14px;font-weight:700;color:#1a1a2e">{q}</div>
            <div style="font-size:26px;font-weight:800;color:#1a1a2e;margin:4px 0">{n}
            <span style="font-size:14px;font-weight:400;color:#444"> customers</span></div>
            <div style="font-size:11px;color:#555">{desc}</div>
            </div>''', unsafe_allow_html=True)

# ══ PAGE 5: CHURN PREDICTION ══════════════════════════════════════
elif page == "Churn Prediction":
    st.markdown(f'<div class="section-header"><b style="font-size:18px">⚠️ Churn Prediction</b><br><span style="font-size:12px;opacity:.7">Random Forest · AUC={CHURN_AUC:.3f} · Rate={dc["Churned"].mean()*100:.1f}%</span></div>', unsafe_allow_html=True)

    col1,col2,col3=st.columns(3)
    with col1:
        fi=pd.Series(RF_MODEL.feature_importances_,index=FEAT_COLS).sort_values()
        fig=go.Figure(go.Bar(x=fi.values,y=fi.index,orientation="h",
            marker_color=["#2196F3" if v>fi.mean() else "#90CAF9" for v in fi.values]))
        fig.update_layout(title="Feature Importance",height=320,showlegend=False,margin=dict(t=50,b=20,l=20,r=20))
        st.plotly_chart(fig,use_container_width=True)
    with col2:
        sc=dc.groupby("RFM_Segment")["Churn_Prob"].mean().reset_index()
        sc["Pct"]=(sc["Churn_Prob"]*100).round(1)
        fig=px.bar(sc,x="RFM_Segment",y="Pct",color="RFM_Segment",
            color_discrete_map=SEG_COLORS,text="Pct")
        fig.update_traces(texttemplate="%{text:.1f}%",textposition="outside")
        fig.update_layout(title="Churn Risk by Segment",height=320,showlegend=False)
        st.plotly_chart(fig,use_container_width=True)
    with col3:
        dc2=dc.copy()
        dc2["Risk_Cat"]=pd.cut(dc2["Churn_Prob"],bins=[0,.3,.6,1.0],labels=["Low","Medium","High"])
        rc=dc2["Risk_Cat"].value_counts().reset_index(); rc.columns=["Risk","Count"]
        fig=px.bar(rc,x="Risk",y="Count",color="Risk",
            color_discrete_map={"Low":"#4CAF50","Medium":"#FF9800","High":"#F44336"},text="Count")
        fig.update_traces(textposition="outside")
        fig.update_layout(title="Churn Risk Distribution",height=320,showlegend=False)
        st.plotly_chart(fig,use_container_width=True)

    st.markdown("### Top High-Risk Customers (Churn > 60%)")
    hr=dc[dc["Churn_Prob"]>=0.6][["Customer_ID","RFM_Segment","CLV","Avg_Discount",
        "Total_Orders","Churn_Prob"]].head(10).copy()
    hr["Churn_Prob"]=(hr["Churn_Prob"]*100).round(1)
    hr["Avg_Discount"]=(hr["Avg_Discount"]*100).round(1)
    hr["CLV"]=hr["CLV"].round(2)
    hr.columns=["Customer ID","Segment","CLV($)","Avg Disc(%)","Orders","Churn Risk(%)"]
    st.dataframe(hr.style.format({"CLV($)":"${:.2f}","Avg Disc(%)":"{:.1f}%","Churn Risk(%)":"{:.1f}%"}),
        use_container_width=True)

# ══ PAGE 6: AI OPTIMIZER (FULLY LIVE) ════════════════════════════
elif page == "AI Optimizer":
    st.markdown('<div class="section-header"><b style="font-size:18px">🎯 AI Optimization Engine — LIVE</b><br><span style="font-size:12px;opacity:.7">PuLP · Results change every time you click Run</span></div>', unsafe_allow_html=True)

    col_p,col_r=st.columns([1,2])
    with col_p:
        st.markdown("#### Parameters")
        budget_pct  = st.slider("Budget % of AOV", 5, 30, 10, 1) / 100
        max_disc_opt= st.selectbox("Max Discount",
            [0.0,0.05,0.10,0.20],
            format_func=lambda x:{0.0:"0% No discounts",0.05:"5% Max",0.10:"10% Max",0.20:"20% Max"}[x],
            index=2)
        min_margin  = st.slider("Min Margin Floor %", 0, 20, 5, 1)
        n_customers = st.slider("Customers", 50, 793, 150, 25)
        run_btn     = st.button("▶ Run AI Optimization", type="primary", use_container_width=True)

    with col_r:
        if run_btn:
            with st.spinner("Running PuLP optimization..."):
                ACTIONS={"No Discount":{"d":0.00,"u":1.00},"Premium Rec":{"d":0.00,"u":1.12}}
                if max_disc_opt>=0.05: ACTIONS["5% Discount"]={"d":0.05,"u":1.08}
                if max_disc_opt>=0.08: ACTIONS["Bundle Offer"]={"d":0.08,"u":1.25}
                if max_disc_opt>=0.10: ACTIONS["10% Discount"]={"d":0.10,"u":1.18}
                if max_disc_opt>=0.20: ACTIONS["20% Discount"]={"d":0.20,"u":1.25}

                samp=dc.sample(n=min(n_customers,len(dc)),random_state=SEED).copy()
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
                                "Margin_Pct":emp(max(r["AOV"],1),r["Avg_Margin_Pct"]/100,a),
                                "AOV":round(max(r["AOV"],1),2)})

                opt_df=pd.DataFrame(results)
                if len(opt_df)==0:
                    st.error("No feasible solution. Lower Min Margin or increase Budget.")
                else:
                    ai_p=opt_df["Profit"].sum()
                    trad_p=float(sum(max(r["AOV"],1)*0.8-max(r["AOV"],1)*(1-r["Avg_Margin_Pct"]/100)
                                     -max(r["AOV"],1)*0.2 for _,r in samp.iterrows()))
                    impr=(ai_p-trad_p)/abs(trad_p)*100 if trad_p!=0 else 0

                    m1,m2,m3,m4=st.columns(4)
                    m1.metric("AI Profit",   f"${ai_p:,.0f}",    f"{len(opt_df)} customers")
                    m2.metric("Traditional", f"${trad_p:,.0f}",  "20% blanket")
                    m3.metric("Improvement", f"{impr:+.1f}%",    "AI vs Traditional")
                    m4.metric("Budget",      f"${BUDGET:,.0f}",  f"{budget_pct:.0%} of AOV")

                    c1,c2,c3=st.columns(3)
                    with c1:
                        ac=opt_df["Action"].value_counts().reset_index()
                        ac.columns=["Action","Count"]
                        fig=px.pie(ac,values="Count",names="Action",hole=0.4,
                            color_discrete_sequence=px.colors.qualitative.Set2)
                        fig.update_layout(title="Actions Distribution",height=280)
                        st.plotly_chart(fig,use_container_width=True)
                    with c2:
                        sp=opt_df.groupby("Segment")["Profit"].sum().reset_index()
                        fig=px.bar(sp,x="Segment",y="Profit",color="Segment",
                            color_discrete_map=SEG_COLORS,text=sp["Profit"].round(0))
                        fig.update_traces(texttemplate="$%{text:,.0f}",textposition="outside")
                        fig.add_hline(y=0,line_dash="dash",line_color="black")
                        fig.update_layout(title="Profit by Segment",height=280,showlegend=False)
                        st.plotly_chart(fig,use_container_width=True)
                    with c3:
                        comp=go.Figure(go.Bar(
                            x=["Traditional","AI-Optimized"],y=[trad_p,ai_p],
                            marker_color=["#F44336" if trad_p<0 else "#FF9800","#4CAF50"],
                            text=[f"${trad_p:,.0f}",f"${ai_p:,.0f}"],textposition="outside"))
                        comp.add_hline(y=0,line_dash="dash",line_color="black")
                        comp.update_layout(title=f"AI vs Traditional ({impr:+.1f}%)",
                            height=280,showlegend=False)
                        st.plotly_chart(comp,use_container_width=True)

                    st.markdown("#### Top 10 Recommendations")
                    top10=opt_df.nlargest(10,"Profit")[["Customer_ID","Segment","Action","Profit","Margin_Pct","AOV"]]
                    st.dataframe(top10.style.format({"Profit":"${:.2f}","Margin_Pct":"{:.1f}%","AOV":"${:.2f}"}),
                        use_container_width=True)
        else:
            st.info("👈 Set parameters and click **▶ Run AI Optimization**. Results update every time.")

# ══ PAGE 7: STRATEGY SIMULATION (LIVE) ═══════════════════════════
elif page == "Strategy Simulation":
    st.markdown('<div class="section-header"><b style="font-size:18px">⚡ Strategy Simulation — Live</b><br><span style="font-size:12px;opacity:.7">Adjust parameters — results recalculate instantly</span></div>', unsafe_allow_html=True)

    col_s,col_r=st.columns([1,2])
    with col_s:
        st.markdown("#### Parameters")
        s_disc  =st.slider("Discount %",    0, 50, 20, 1)
        s_uplift=st.slider("Demand Uplift %", 0, 50, 15, 1)
        s_n     =st.slider("Customers",    50, 793, 200, 25)
        st.info(f"**{s_disc}% discount** generates **{s_uplift}% more** demand")

with col_r:
    samp2 = dc.sample(n=min(s_n, len(dc)), random_state=SEED)
    disc = s_disc / 100
    uplift = 1 + s_uplift / 100

    def calc(d, u):
        r2 = p2 = 0
        for _, r in samp2.iterrows():
            aov = max(r["AOV"], 1)
            bm = r["Avg_Margin_Pct"] / 100
            rev = aov * u * (1 - d)
            r2 += rev
            p2 += rev - aov * (1 - bm) - aov * u * d
        return round(r2, 2), round(p2, 2)

     r1, p1 = calc(disc, uplift)

            # ✅ FIXED LOGIC
            half_disc = disc * 0.5
            if disc == 0:
                r2, p2 = calc(0, uplift)
            else:
                r2, p2 = calc(half_disc, max(uplift * 0.85, 1.0))

            r3, p3 = calc(0.05, 1.12)
            r4, p4 = calc(0.00, 1.12)

        labels=[f"Your Setting ({s_disc}%)",f"Half Disc ({s_disc//2}%)","5% Bundle","AI No-Discount"]
        profits=[p1,p2,p3,p4]; revenues=[r1,r2,r3,r4]
        colors_strat=["#F44336","#FF9800","#4CAF50","#2196F3"]
        best_idx=profits.index(max(profits))

        m1,m2,m3,m4=st.columns(4)
        for col_w,lbl,pv,rv in zip([m1,m2,m3,m4],labels,profits,revenues):
            col_w.metric(lbl,f"${pv:,.0f}","★ BEST" if pv==max(profits) else f"Rev ${rv:,.0f}")

        fig=make_subplots(rows=1,cols=2,subplot_titles=["Total Profit","Total Revenue"])
        fig.add_bar(x=labels,y=profits,marker_color=colors_strat,
            text=[f"${v:,.0f}" for v in profits],textposition="outside",row=1,col=1)
        fig.add_bar(x=labels,y=revenues,marker_color=colors_strat,
            text=[f"${v:,.0f}" for v in revenues],textposition="outside",row=1,col=2)
        fig.add_hline(y=0,line_dash="dash",line_color="black",row=1,col=1)
        fig.update_layout(showlegend=False,height=380)
        st.plotly_chart(fig,use_container_width=True)

        ai_vs=((p4-p1)/abs(p1)*100) if p1!=0 else 0
        if ai_vs>0:
            st.success(f"✅ AI No-Discount generates **{ai_vs:+.1f}%** more profit than your {s_disc}% discount setting on {s_n} customers.")
        else:
            st.warning(f"⚠️ At low discount levels, strategies perform similarly. AI advantage is largest when discount > {BREAKEVEN:.0%}.")

# ══ PAGE 8: CUSTOMER LOOKUP ═══════════════════════════════════════
elif page == "Customer Lookup":
    st.markdown('<div class="section-header"><b style="font-size:18px">🔍 Customer Lookup</b><br><span style="font-size:12px;opacity:.7">Search any of the 793 real customers</span></div>', unsafe_allow_html=True)

    cust_ids=sorted(dc["Customer_ID"].tolist())
    selected=st.selectbox("Select Customer ID",cust_ids,index=0)

    row=dc[dc["Customer_ID"]==selected].iloc[0]
    seg=row["RFM_Segment"]; col=SEG_COLORS.get(seg,"#999")
    rec={"Premium":"⭐ Premium Rec — exclusive loyalty rewards, NO discount needed",
         "Loyal":"🎁 Bundle Offer — value add without price cut",
         "Price-Sensitive":"🚫 No Discount — avoid reinforcing price dependency",
         "Low-Value":"📧 Low-cost email reactivation only"}
    risk=("🔴 High Risk" if row["Churn_Prob"]>0.6 else
          "🟡 Medium Risk" if row["Churn_Prob"]>0.3 else "🟢 Low Risk")

    st.markdown(f'<div style="background:white;border-radius:12px;padding:24px;box-shadow:0 2px 12px rgba(0,0,0,.08);margin-bottom:16px"><div style="display:flex;align-items:center;gap:16px;margin-bottom:20px"><div style="width:60px;height:60px;border-radius:50%;background:{col};display:flex;align-items:center;justify-content:center;color:white;font-size:20px;font-weight:800">{selected[:2].upper()}</div><div><div style="font-size:20px;font-weight:700;color:#1a1a2e">{selected}</div><span style="background:{col};color:white;padding:3px 12px;border-radius:12px;font-size:12px;font-weight:600">{seg}</span> &nbsp; <span style="font-size:12px;color:#444">RFM Score: {int(row["RFM_Score"])} · {row["Preferred_Category"]} · {row["Tenure_Years"]:.1f} yrs</span></div></div></div>',unsafe_allow_html=True)

    c1,c2,c3=st.columns(3)
    c1.metric("Customer LTV",  f"${row['CLV']:,.0f}")
    c2.metric("Total Profit",  f"${row['Total_Profit']:,.0f}")
    c3.metric("Total Orders",  f"{int(row['Total_Orders'])}")
    c4,c5,c6=st.columns(3)
    c4.metric("Avg Discount",  f"{row['Avg_Discount']*100:.1f}%")
    c5.metric("Avg Margin",    f"{row['Avg_Margin_Pct']:.1f}%")
    c6.metric("Churn Risk",    f"{row['Churn_Prob']*100:.1f}%", risk)

    st.markdown(f'<div style="background:#1a1a2e;color:white;border-radius:10px;padding:16px;margin-top:12px"><div style="font-size:11px;color:rgba(255,255,255,.5);margin-bottom:6px">🤖 AI RECOMMENDATION</div><div style="font-size:15px;font-weight:600">{rec.get(seg,"N/A")}</div></div>',unsafe_allow_html=True)

    st.markdown("#### Transaction History")
    txns=dfs[dfs["Customer_ID"]==selected][
        ["Order_Date","Category","Sub-Category","Sales","Discount","Profit"]
    ].sort_values("Order_Date",ascending=False).head(15).copy()
    txns["Discount"]=(txns["Discount"]*100).round(1)
    st.dataframe(txns.style.format({"Sales":"${:.2f}","Profit":"${:.2f}","Discount":"{:.1f}%"}),
        use_container_width=True)
