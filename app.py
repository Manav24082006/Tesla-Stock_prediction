import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

st.set_page_config(
    page_title="Tesla Stock Price Predictor",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Custom styling
# -----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;750&display=swap');
html, body, [class*="css"] { font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
.stApp { background: #ffffff; color: #172238; }
.block-container { padding: 1.2rem 2.8rem 1rem; max-width: 1500px; }
[data-testid="stSidebar"] { background: linear-gradient(180deg,#fbfcff 0%,#f5f8fc 100%); border-right: 1px solid #e0e6ef; }
[data-testid="stSidebar"] > div:first-child { padding-top: 1.5rem; }
.tesla-logo { height: 76px; display:flex; align-items:center; justify-content:center; }
.tesla-symbol { color:#e82127; font-family:Arial,sans-serif; font-size:67px; font-weight:900; line-height:1; transform:scaleX(.85); }
.side-title { text-align:center; font-size:21px; font-weight:700; color:#172238; }
.side-subtitle { text-align:center; color:#44526a; font-size:14px; line-height:1.7; margin-bottom:1.3rem; }
.side-divider { height:1px; background:#d6deea; margin:1.3rem .5rem; }
.side-section-title { color:#e82127; font-size:15px; font-weight:700; margin-bottom:1rem; }
.detail { margin-bottom:14px; }
.detail-label { display:block; font-size:13px; color:#172238; margin-bottom:4px; }
.detail-value { font-size:14px; font-weight:600; }
.green { color:#159447!important; } .blue { color:#145dd7!important; }
.purple { color:#6b1fd1!important; } .orange { color:#ee7514!important; }
.red { color:#e82127!important; }
.side-info { margin-top:1.8rem; padding:15px 14px; border:1px solid #cde8da; border-radius:10px; background:linear-gradient(135deg,#f2fbf6,#e8f6ef); color:#19452e; font-size:13px; line-height:1.7; }
.side-chart { height:45px; margin-top:8px; background: linear-gradient(135deg,transparent 45%,#24a653 46%,transparent 48%), linear-gradient(160deg,transparent 40%,#24a653 41%,transparent 43%); opacity:.55; }
div.stButton > button { border-radius:10px; border:0; font-weight:600; text-align:left; }
div.stButton > button:hover { border-color:#e82127; color:#e82127; }
.hero { min-height:160px; display:flex; align-items:center; gap:28px; padding:22px 30px; border:1px solid #f8dfe2; border-radius:14px; background:linear-gradient(110deg,#fff0f3,#fff8f9); }
.hero-image { width:112px; height:112px; flex-shrink:0; display:flex; align-items:center; justify-content:center; border:1px solid #f1d9dd; border-radius:17px; background:white; box-shadow:0 8px 20px rgba(215,45,58,.08); font-size:55px; }
.hero-content { flex:1; } .hero-content h1 { margin:0 0 10px; font-size:38px; color:#172238; } .hero-content p { margin:0; color:#52617a; font-size:16px; line-height:1.55; }
.updated { width:215px; padding:15px 18px; display:flex; align-items:center; gap:15px; border:1px solid #f1dce0; border-radius:13px; background:white; }
.updated-icon { font-size:30px; color:#e82127; } .updated span { display:block; font-size:13px; color:#4b5870; margin-bottom:5px; } .updated strong { font-size:16px; color:#172238; }
.card { margin-top:17px; padding:25px 28px; border:1px solid #e1e7ef; border-radius:14px; background:white; box-shadow:0 4px 14px rgba(31,49,76,.055); }
.section-title { color:#e82127; font-size:23px; font-weight:700; margin:0 0 5px; } .section-subtitle { color:#53627b; font-size:14px; margin-bottom:20px; }
.pred-btn button { background:linear-gradient(135deg,#ed1c24,#ef2b31)!important; color:white!important; border:0!important; height:50px; }
.definition { padding:10px 14px; border:1px solid #cbdcf7; border-radius:8px; background:#f7faff; color:#52627e; font-size:13px; }
.prediction { margin-top:17px; padding:21px 39px; display:flex; align-items:center; gap:35px; border:1px solid #bde8cf; border-radius:13px; background:linear-gradient(100deg,#f6fdf8,#fcfffd); }
.pred-circle { width:96px; height:96px; flex-shrink:0; border-radius:50%; display:flex; align-items:center; justify-content:center; background:#ecfaf1; border:1px solid #d5f1df; color:#1c9d4b; font-size:38px; }
.prediction h3 { color:#159447; font-size:18px; margin:0 0 5px; } .pred-value { color:#169447; font-size:42px; font-weight:750; margin-bottom:6px; }
.success { display:inline-block; padding:4px 9px; border:1px solid #bde8cc; border-radius:5px; color:#159447; background:#ebfbf1; font-size:12px; }
.stat { min-height:105px; padding:17px; border:1px solid #e2e7ee; border-radius:12px; box-shadow:0 4px 13px rgba(25,46,76,.05); background:white; }
.stat-icon { font-size:30px; margin-bottom:8px; } .stat-value { font-size:21px; font-weight:700; } .stat-title { font-size:13px; color:#18253b; } .stat-subtitle { font-size:12px; color:#59677e; }
.bottom-card { min-height:135px; padding:19px 27px; border:1px solid #e1e7ef; border-radius:14px; background:white; box-shadow:0 4px 14px rgba(31,49,76,.05); }
.bottom-card.blue-bg { border-color:#cddff9; background:#f8fbff; } .bottom-title { font-size:18px; font-weight:700; color:#122653; }
.metric { text-align:center; border-right:1px solid #d9dee6; } .metric:last-child { border-right:0; } .metric-label { display:block; font-size:13px; margin-bottom:5px; } .metric-value { font-size:24px; font-weight:700; }
.page-header { padding:20px 24px; border:1px solid #e1e7ef; border-radius:14px; background:linear-gradient(110deg,#f8fbff,#fff); }
.page-header h1 { margin:0 0 5px; font-size:30px; color:#172238; } .page-header p { margin:0; color:#53627b; font-size:14px; }
.summary { padding:18px; border:1px solid #e2e7ee; border-radius:12px; background:white; box-shadow:0 4px 13px rgba(25,46,76,.05); }
.summary-number { display:block; font-size:27px; font-weight:750; } .summary-label { color:#59677e; font-size:13px; }
.chart-card { margin-top:17px; padding:22px 25px; border:1px solid #e1e7ef; border-radius:14px; background:white; box-shadow:0 4px 14px rgba(31,49,76,.05); }
.chart-title { font-size:20px; font-weight:700; color:#122653; } .chart-desc { font-size:13px; color:#59677e; }
.feature { padding:17px; border:1px solid #e1e7ef; border-radius:12px; background:white; } .feature-name { font-size:17px; font-weight:700; margin-bottom:5px; } .feature-desc { font-size:13px; color:#59677e; line-height:1.5; }
.target-badge { font-size:10px; background:#fff1f2; color:#e82127; padding:3px 6px; border-radius:5px; margin-left:5px; }
.about-hero { margin-top:17px; padding:25px; display:flex; gap:20px; align-items:center; border:1px solid #e1e7ef; border-radius:14px; background:#fafcff; }
.about-icon { font-size:42px; color:#6b1fd1; } .about-hero h2 { margin:0 0 7px; color:#172238; } .about-hero p { margin:0; color:#53627b; line-height:1.6; }
.about-card { padding:20px; border:1px solid #e1e7ef; border-radius:12px; background:white; min-height:150px; } .about-card h3 { color:#172238; margin:8px 0; font-size:17px; } .about-card p { color:#59677e; font-size:13px; line-height:1.6; }
.formula-card { margin-top:17px; padding:25px; border:1px solid #d9e8ff; border-radius:14px; background:#f8fbff; }
.formula-card h2 { color:#122653; margin-top:0; } .formula-note { color:#59677e; font-size:13px; line-height:1.6; }
.workflow { display:flex; align-items:center; justify-content:center; gap:18px; margin-top:18px; } .workflow-step { padding:16px 25px; border:1px solid #e1e7ef; border-radius:12px; background:white; text-align:center; }
.workflow-num { width:35px; height:35px; border-radius:50%; margin:auto auto 7px; display:flex; align-items:center; justify-content:center; color:white; font-weight:700; background:#145dd7; }
.workflow-step strong, .workflow-step span { display:block; } .workflow-step span { color:#59677e; font-size:12px; margin-top:4px; }
.performance-box { padding:20px; text-align:center; border:1px solid #e1e7ef; border-radius:12px; background:white; } .performance-box span, .performance-box small { display:block; color:#59677e; }
.performance-box strong { display:block; font-size:30px; margin:5px 0; } .footer { text-align:center; color:#465570; font-size:13px; padding:20px 0 5px; }
@media (max-width: 900px) { .block-container { padding:1rem; } .hero { flex-wrap:wrap; } .updated { width:100%; } .prediction { flex-wrap:wrap; padding:20px; } .workflow { flex-wrap:wrap; } }
</style>
""", unsafe_allow_html=True)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "TSLA.csv"
FEATURES = ["Open", "High", "Low", "Volume"]
TARGET = "Close"

# -----------------------------
# Exact preprocessing used in the project notebook
# -----------------------------
@st.cache_data
def _load_data():
    df = pd.read_csv(DATA_PATH)
    df.drop_duplicates(inplace=True)
    q1 = df["Close"].quantile(0.25)
    q3 = df["Close"].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    df = df[(df["Close"] >= lower) & (df["Close"] <= upper)].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df = df.dropna().reset_index(drop=True)
    return df, q1, q3, iqr, lower, upper

# Fix indentation generated above for decorators in a simple, explicit way.

def get_model_data():
    df, q1, q3, iqr, lower, upper = _load_data()
    X = df[FEATURES]
    y = df[TARGET]
    split_index = int(len(df) * 0.80)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    return df, X_train, X_test, y_train, y_test, split_index, (q1, q3, iqr, lower, upper)

@st.cache_resource
def train_linear_model():
    df, X_train, X_test, y_train, y_test, split_index, bounds = get_model_data()
    model = LinearRegression()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "MAE": mean_absolute_error(y_test, pred),
        "MSE": mean_squared_error(y_test, pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
        "R2": r2_score(y_test, pred),
    }
    return model, metrics, pred, X_train, X_test, y_train, y_test

@st.cache_data
def get_cv_results():
    df, X_train, X_test, y_train, y_test, split_index, bounds = get_model_data()
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "AdaBoost": AdaBoostRegressor(n_estimators=50, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "SVR": SVR(),
    }
    tscv = TimeSeriesSplit(n_splits=5)
    rows = []
    for name, model in models.items():
        scores = cross_val_score(
            model, X_train, y_train,
            cv=tscv,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1 if name != "SVR" else None
        )
        rmse = -scores
        rows.append({
            "Model": name,
            "CV AVG RMSE": rmse.mean(),
            "CV STD RMSE": rmse.std(),
        })
    return pd.DataFrame(rows)

# Load the actual project data/model once.
df, X_train, X_test, y_train, y_test, split_index, outlier_info = get_model_data()
model, model_metrics, test_predictions, _, _, _, _ = train_linear_model()

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown('<div class="tesla-logo"><div class="tesla-symbol">T</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="side-title">Tesla Price Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="side-subtitle">Predict Tesla Closing Price<br>using Linear Regression</div>', unsafe_allow_html=True)

    page = st.radio("Navigation", ["🏠  Predict Price", "📊  Dataset Overview", "ℹ️  About Model"], label_visibility="collapsed")

    st.markdown('<div class="side-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="side-section-title">Model Details</div>', unsafe_allow_html=True)
    details = [
        ("Algorithm", "Linear Regression", "green"),
        ("Target Variable", "Close", "blue"),
        ("Input Features", "Open, High, Low, Volume", "purple"),
        ("Clean Records", f"{len(df):,}", "orange"),
        ("Columns", f"{len(pd.read_csv(DATA_PATH).columns)}", "blue"),
    ]
    for label, value, color in details:
        st.markdown(f'<div class="detail"><span class="detail-label">{label}</span><strong class="detail-value {color}">{value}</strong></div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="side-info">
        <b>↗ Historical Data</b>
        <p>This application uses the cleaned Tesla dataset and a trained Linear Regression model to predict the closing price.</p>
        <div class="side-chart"></div>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------
# Prediction page
# -----------------------------
def prediction_page():
    latest = df.iloc[-1]
    st.markdown(f"""
    <div class="hero">
        <div class="hero-image">🚘</div>
        <div class="hero-content">
            <h1>Tesla Stock Price Predictor</h1>
            <p>Enter stock information below and predict the Tesla closing price<br>using the trained Linear Regression model.</p>
        </div>
        <div class="updated"><div class="updated-icon">📅</div><div><span>Latest Dataset Date</span><strong>{latest['Date'].strftime('%d %b %Y')}</strong></div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="section-title">📈 Enter Stock Information</div><div class="section-subtitle">Values are connected to the trained model. Defaults show the latest cleaned dataset row.</div></div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1: open_price = st.number_input("📈 Open Price ($)", value=float(latest["Open"]), step=0.01, format="%.2f")
    with c2: high_price = st.number_input("⬆️ High Price ($)", value=float(latest["High"]), step=0.01, format="%.2f")
    with c3: low_price = st.number_input("⬇️ Low Price ($)", value=float(latest["Low"]), step=0.01, format="%.2f")
    with c4: volume = st.number_input("⚡ Trading Volume", value=int(latest["Volume"]), step=100000, format="%d")

    st.markdown('<div class="pred-btn">', unsafe_allow_html=True)
    predict = st.button("✨  Predict Closing Price", use_container_width=False)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="definition"><b>Open:</b> Opening price &nbsp; • &nbsp; <b>High:</b> Highest daily price &nbsp; • &nbsp; <b>Low:</b> Lowest daily price &nbsp; • &nbsp; <b>Volume:</b> Shares traded</div>', unsafe_allow_html=True)

    if "prediction" not in st.session_state:
        st.session_state.prediction = float(model.predict(pd.DataFrame([[latest["Open"], latest["High"], latest["Low"], latest["Volume"]]], columns=FEATURES))[0])
    if predict:
        input_df = pd.DataFrame([[open_price, high_price, low_price, volume]], columns=FEATURES)
        st.session_state.prediction = float(model.predict(input_df)[0])

    prediction = st.session_state.prediction
    st.markdown(f"""
    <div class="prediction"><div class="pred-circle">📈</div><div><h3>Predicted Tesla Closing Price</h3><div class="pred-value">${prediction:.2f}</div><div class="success">✓ Prediction completed successfully!</div></div></div>
    """, unsafe_allow_html=True)

    st.write("")
    cols = st.columns(4)
    stats = [
        ("🗄️", f"{len(df):,}", "Clean Records", "After preprocessing", "purple"),
        ("📋", str(len(FEATURES)), "Input Features", "Used by model", "blue"),
        ("🎯", "Close", "Prediction Target", "Target variable", "orange"),
        ("🧠", "Linear Regression", "Machine Learning", "Algorithm", "red"),
    ]
    for col, (icon, value, title, subtitle, color) in zip(cols, stats):
        with col:
            st.markdown(f'<div class="stat"><div class="stat-icon">{icon}</div><div class="stat-value {color}">{value}</div><div class="stat-title">{title}</div><div class="stat-subtitle">{subtitle}</div></div>', unsafe_allow_html=True)

    st.write("")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="bottom-card"><div class="bottom-title">📊 Model Performance</div></div>', unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        for col, label, value, color in [(m1,"MAE",model_metrics["MAE"],"purple"),(m2,"MSE",model_metrics["MSE"],"blue"),(m3,"R² Score",model_metrics["R2"],"green")]:
            with col: st.markdown(f'<div class="metric"><span class="metric-label">{label}</span><strong class="metric-value {color}">{value:.4f}</strong></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="bottom-card blue-bg"><div class="bottom-title">ℹ️ How does this work?</div>
        <p style="color:#394a67;font-size:13px;line-height:1.65;margin-top:12px;">The model learns from <b>Open, High, Low and Volume</b>, using the first <b>80%</b> of the cleaned chronological dataset for training and the last <b>20%</b> for testing. The prediction button sends your values through the trained Linear Regression model.</p></div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="footer">♥ &nbsp; Built with Streamlit &nbsp; • &nbsp; Tesla Stock Price Prediction System</div>', unsafe_allow_html=True)

# -----------------------------
# Dataset page
# -----------------------------
def dataset_page():
    raw_df = pd.read_csv(DATA_PATH)
    st.markdown('<div class="page-header"><h1>📊 Dataset Overview</h1><p>Explore the Tesla historical stock market dataset used to train the prediction model.</p></div>', unsafe_allow_html=True)
    cols = st.columns(4)
    summary = [(f"{len(df):,}","Clean Records","purple"),(str(len(raw_df)),"Original Records","blue"),(str(len(FEATURES)),"Input Features","green"),("1","Target Variable","orange")]
    for col,(num,label,color) in zip(cols,summary):
        with col: st.markdown(f'<div class="summary"><span class="summary-number {color}">{num}</span><span class="summary-label">{label}</span></div>',unsafe_allow_html=True)

    trend = df.set_index("Date")[["Open","High","Low","Close"]].tail(250)
    st.markdown('<div class="chart-card"><div class="chart-title">📈 Tesla Price Trend</div><div class="chart-desc">Latest 250 cleaned observations</div></div>', unsafe_allow_html=True)
    st.line_chart(trend, height=300, use_container_width=True)

    averages = df[["Open","High","Low","Close"]].mean().rename_axis("Price Type").reset_index(name="Average")
    c1,c2 = st.columns(2)
    with c1:
        st.markdown('<div class="chart-card"><div class="chart-title">📊 Average Price Comparison</div><div class="chart-desc">Actual averages from the cleaned dataset</div></div>', unsafe_allow_html=True)
        st.bar_chart(averages.set_index("Price Type"), height=300)
    with c2:
        st.markdown('<div class="chart-card"><div class="chart-title">🥧 Dataset Distribution</div><div class="chart-desc">Chronological train/test split</div></div>', unsafe_allow_html=True)
        split = pd.DataFrame({"Records":[len(X_train),len(X_test)]},index=["Training","Testing"])
        st.bar_chart(split, height=220)
        st.markdown(f"<p style='text-align:center;color:#59677e'><b>{len(X_train):,}</b> Training &nbsp;&nbsp; <b>{len(X_test):,}</b> Testing</p>", unsafe_allow_html=True)

    st.markdown('<div class="chart-card"><div class="chart-title">🧩 Dataset Features</div><div class="chart-desc">Features used by the Linear Regression model to predict Close.</div></div>', unsafe_allow_html=True)
    features = [("Open","Opening price of Tesla stock for the trading day.","blue",False),("High","Highest price reached during the trading day.","green",False),("Low","Lowest price reached during the trading day.","red",False),("Volume","Total number of Tesla shares traded.","purple",False),("Close","Target variable representing the closing price.","orange",True)]
    fcols=st.columns(5)
    for col,(name,desc,color,target) in zip(fcols,features):
        with col:
            badge='<span class="target-badge">Target</span>' if target else ''
            st.markdown(f'<div class="feature"><div class="feature-name {color}">{name}{badge}</div><div class="feature-desc">{desc}</div></div>',unsafe_allow_html=True)

    st.markdown('<div class="chart-card"><div class="chart-title">📄 Sample Dataset</div><div class="chart-desc">Latest cleaned records used by the application.</div></div>', unsafe_allow_html=True)
    sample = df.tail(8).copy()
    sample["Date"] = sample["Date"].dt.strftime("%Y-%m-%d")
    st.dataframe(sample, use_container_width=True, hide_index=True)
    st.markdown('<div class="footer">♥ &nbsp; Built with Streamlit &nbsp; • &nbsp; Tesla Stock Price Prediction System</div>', unsafe_allow_html=True)

# -----------------------------
# About page + formulas
# -----------------------------
def about_page():
    st.markdown('<div class="page-header"><h1>🧠 About Model</h1><p>Learn the algorithms, mathematical formulas, preprocessing and evaluation metrics used in this Tesla prediction project.</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="about-hero"><div class="about-icon">🧠</div><div><h2>Linear Regression</h2><p>Linear Regression learns a relationship between the four input features and Tesla Close. The trained model is used directly by the prediction page.</p></div></div>', unsafe_allow_html=True)

    cards=[("🗄️","1. Historical Data","Historical Tesla records are cleaned, sorted by Date and prepared for modeling."),("🧹","2. Preprocessing","Duplicates, Close-price IQR outliers and missing values are handled before training."),("🧠","3. Model Training","Linear Regression is trained on the first 80% of the chronological data."),("🎯","4. Prediction","The trained model predicts Close from Open, High, Low and Volume.")]
    acols=st.columns(4)
    for col,(icon,title,text) in zip(acols,cards):
        with col: st.markdown(f'<div class="about-card"><div style="font-size:27px">{icon}</div><h3>{title}</h3><p>{text}</p></div>',unsafe_allow_html=True)

    # Core formula section
    st.markdown('<div class="formula-card"><h2>📐 1. Linear Regression Formula</h2>',unsafe_allow_html=True)
    st.latex(r"\hat{y} = \beta_0 + \beta_1X_1 + \beta_2X_2 + \beta_3X_3 + \beta_4X_4")
    st.markdown('<p class="formula-note"><b>In this project:</b> X₁ = Open, X₂ = High, X₃ = Low, X₄ = Volume, and ŷ = predicted Close. β₀ is the intercept and β₁…β₄ are learned coefficients.</p>',unsafe_allow_html=True)
    coef_df=pd.DataFrame({"Term":["Intercept","Open","High","Low","Volume"],"Learned coefficient":[model.intercept_,*model.coef_]})
    st.dataframe(coef_df.style.format({"Learned coefficient":"{:.8f}"}),use_container_width=True,hide_index=True)
    st.markdown('</div>',unsafe_allow_html=True)

    # Preprocessing formulas
    st.markdown('<div class="formula-card"><h2>🧹 2. IQR Outlier Detection</h2>',unsafe_allow_html=True)
    st.latex(r"IQR = Q_3 - Q_1")
    st.latex(r"Lower\ Bound = Q_1 - 1.5(IQR)")
    st.latex(r"Upper\ Bound = Q_3 + 1.5(IQR)")
    q1,q3,iqr,lower,upper=outlier_info
    st.markdown(f'<p class="formula-note">For the project Close column: Q₁ = <b>{q1:.4f}</b>, Q₃ = <b>{q3:.4f}</b>, IQR = <b>{iqr:.4f}</b>, Lower Bound = <b>{lower:.4f}</b>, Upper Bound = <b>{upper:.4f}</b>.</p>',unsafe_allow_html=True)
    st.markdown('</div>',unsafe_allow_html=True)

    # Evaluation formulas
    st.markdown('<div class="formula-card"><h2>📊 3. Model Evaluation Formulas</h2>',unsafe_allow_html=True)
    st.markdown('**Mean Absolute Error (MAE)**')
    st.latex(r"MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|")
    st.markdown('**Mean Squared Error (MSE)**')
    st.latex(r"MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2")
    st.markdown('**Root Mean Squared Error (RMSE)**')
    st.latex(r"RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}")
    st.markdown('**R² Score**')
    st.latex(r"R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}{\sum_{i=1}^{n}(y_i-\bar{y})^2}")
    st.markdown('<p class="formula-note">Here yᵢ is the actual Close, ŷᵢ is the predicted Close, ȳ is the mean actual Close, and n is the number of test observations.</p>',unsafe_allow_html=True)
    st.markdown('</div>',unsafe_allow_html=True)

    # Split and CV formulas
    st.markdown('<div class="formula-card"><h2>⏱️ 4. Train/Test Split & Time-Series Cross Validation</h2>',unsafe_allow_html=True)
    st.latex(r"Training\ Size = 0.80N")
    st.latex(r"Testing\ Size = 0.20N")
    st.markdown('<p class="formula-note">The data is sorted chronologically first. The first 80% is used for training and the final 20% is reserved for testing, so future observations are not used to train the test prediction.</p>',unsafe_allow_html=True)
    cv_df=get_cv_results()
    st.dataframe(cv_df.style.format({"CV AVG RMSE":"{:.4f}","CV STD RMSE":"{:.4f}"}),use_container_width=True,hide_index=True)
    st.markdown('</div>',unsafe_allow_html=True)

    # Advanced model formulas
    st.markdown('<div class="formula-card"><h2>🌲 5. Advanced Model Formulas</h2>',unsafe_allow_html=True)
    st.markdown('**Random Forest — average of tree predictions**')
    st.latex(r"\hat{y} = \frac{1}{T}\sum_{t=1}^{T}\hat{y}_t")
    st.markdown('**Gradient Boosting — additive weak learners**')
    st.latex(r"F_m(x)=F_{m-1}(x)+\eta h_m(x)")
    st.markdown('**AdaBoost — weighted combination of weak learners**')
    st.latex(r"F(x)=\sum_{m=1}^{M}\alpha_m h_m(x)")
    st.markdown('**SVR — ε-insensitive loss**')
    st.latex(r"L_\epsilon(y,f(x))=\max(0,|y-f(x)|-\epsilon)")
    st.markdown('<p class="formula-note">These formulas describe the core mathematical idea of the advanced models compared in the project. Their detailed optimization is handled by scikit-learn.</p>',unsafe_allow_html=True)
    st.markdown('</div>',unsafe_allow_html=True)

    st.markdown('<div class="chart-card"><div class="chart-title">⚡ Model Workflow</div><div class="workflow"><div class="workflow-step"><div class="workflow-num">1</div><strong>Clean</strong><span>Duplicates, IQR, missing values</span></div><div style="font-size:25px">→</div><div class="workflow-step"><div class="workflow-num" style="background:#6b1fd1">2</div><strong>Split</strong><span>80% chronological training</span></div><div style="font-size:25px">→</div><div class="workflow-step"><div class="workflow-num" style="background:#ee7514">3</div><strong>Train</strong><span>Linear Regression</span></div><div style="font-size:25px">→</div><div class="workflow-step"><div class="workflow-num" style="background:#159447">4</div><strong>Predict</strong><span>Close price</span></div></div></div>',unsafe_allow_html=True)

    st.markdown('<div class="chart-card"><div class="chart-title">Model Performance on Test Data</div></div>',unsafe_allow_html=True)
    pcols=st.columns(4)
    for col,label,value,desc,color in [(pcols[0],"MAE",model_metrics["MAE"],"Mean Absolute Error","purple"),(pcols[1],"MSE",model_metrics["MSE"],"Mean Squared Error","blue"),(pcols[2],"RMSE",model_metrics["RMSE"],"Root Mean Squared Error","orange"),(pcols[3],"R² Score",model_metrics["R2"],"Coefficient of Determination","green")]:
        with col: st.markdown(f'<div class="performance-box"><span>{label}</span><strong class="{color}">{value:.4f}</strong><small>{desc}</small></div>',unsafe_allow_html=True)

    st.markdown('<div class="footer">♥ &nbsp; Built with Streamlit &nbsp; • &nbsp; Tesla Stock Price Prediction System</div>',unsafe_allow_html=True)

# -----------------------------
# Router
# -----------------------------
if page.startswith("🏠"):
    prediction_page()
elif page.startswith("📊"):
    dataset_page()
else:
    about_page()
