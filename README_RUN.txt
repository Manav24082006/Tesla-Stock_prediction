TESLA PROJECT - CONNECTED STREAMLIT VERSION

This version connects the Streamlit frontend directly to the finalized ML workflow in Project.ipynb.
There is no separate REST/HTTP backend in the uploaded project; the ML model is executed by app.py.

NEW IN THIS VERSION
- Real Linear Regression prediction using TSLA.csv
- Exact notebook preprocessing: duplicate removal, Close IQR outlier removal, date sorting and missing-value removal
- Chronological 80/20 train-test split
- Dynamic MAE, MSE, RMSE and R2 metrics
- Dataset charts use the actual cleaned TSLA dataset
- About Model now contains the project's mathematical formulas:
  * Linear Regression
  * IQR outlier detection
  * MAE
  * MSE
  * RMSE
  * R2
  * 80/20 split
  * Time-series cross-validation explanation
  * Random Forest
  * Gradient Boosting
  * AdaBoost
  * SVR epsilon-insensitive loss
- Learned Linear Regression coefficients are displayed in the frontend
- 5-fold TimeSeriesSplit CV results are displayed in the frontend

RUN
1. Open a terminal in this folder.
2. Install dependencies:
   pip install -r requirements.txt
3. Start the app:
   streamlit run app.py
4. Open the Streamlit URL shown in the terminal.
