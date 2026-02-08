import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Sri Lanka GDP Growth Prediction", layout="wide")
st.title("🇱🇰 Sri Lanka GDP Growth Prediction")

@st.cache_resource
def load_artifacts():
    try:
        rf_model = joblib.load('random_forest_model.pkl')
        lr_model = joblib.load('linear_regression_model.pkl')
        scaler = joblib.load('scaler.pkl')
        trained_features = joblib.load('trained_features.pkl')
        return rf_model, lr_model, scaler, trained_features, None
    except Exception as e:
        return None, None, None, None, str(e)

@st.cache_data
def load_dataset():
    try:
        df = pd.read_csv('sri_lanka_economy_2003_2019_engineered.csv')
        return df, None
    except Exception as e:
        return None, str(e)

rf_model, lr_model, scaler, trained_features, error = load_artifacts()
if error:
    st.error(f"❌ Error loading models: {error}")
    st.stop()

df, error = load_dataset()
if error:
    st.error(f"❌ Error loading dataset: {error}")
    st.stop()

# Sidebar
st.sidebar.header("Model Selection")
model_choice = st.sidebar.selectbox("Choose Model", ["Random Forest", "Linear Regression"])

st.sidebar.markdown("---")
st.sidebar.header("Year Selection")
selected_year = st.sidebar.number_input("Select Year", min_value=2025, max_value=2050, value=2030, step=1)

st.sidebar.markdown("---")
show_forecast = st.sidebar.checkbox("Show forecast range")
if show_forecast:
    start_year = st.sidebar.number_input("Start Year", min_value=2025, max_value=2050, value=2025, step=1)
    end_year = st.sidebar.number_input("End Year", min_value=2025, max_value=2050, value=2035, step=1)

# Main inputs
st.subheader("📊 Input Economic Indicators")
col1, col2 = st.columns(2)

with col1:
    inflation = st.number_input("Inflation_Deflator (%)", value=5.0, step=0.1, format="%.2f")
    exchange_rate = st.number_input("ExchangeRate", value=180.0, step=1.0, format="%.2f")

with col2:
    unemployment = st.number_input("Unemployment (%)", value=4.5, step=0.1, format="%.2f")
    lending_rate = st.number_input("LendingRate (%)", value=3.0, step=0.1, format="%.2f")

# Buttons
btn_col1, btn_col2 = st.columns([1, 1])
with btn_col1:
    predict_btn = st.button("🔮 Predict", type="primary", use_container_width=True)
with btn_col2:
    if st.button("🔄 Reset", use_container_width=True):
        st.rerun()

if predict_btn:
    try:
        # Get baseline from last row
        last_row = df.iloc[-1]
        second_last_row = df.iloc[-2]
        
        # Calculate engineered features
        exchange_pct_change = ((exchange_rate - last_row['ExchangeRate']) / last_row['ExchangeRate']) * 100
        inflation_change = inflation - last_row['Inflation_Deflator']
        unemployment_change = unemployment - last_row['Unemployment']
        inflation_ma3 = np.mean([last_row['Inflation_Deflator'], second_last_row['Inflation_Deflator'], inflation])
        gdp_growth_lag1 = last_row['GDP_Growth']
        
        # Build feature dict
        feature_dict = {
            'Inflation_Deflator': inflation,
            'ExchangeRate': exchange_rate,
            'Unemployment': unemployment,
            'LendingRate': lending_rate,
            'ExchangeRate_pct_change': exchange_pct_change,
            'Inflation_change': inflation_change,
            'Unemployment_change': unemployment_change,
            'Inflation_ma3': inflation_ma3,
            'GDP_Growth_lag1': gdp_growth_lag1
        }
        
        # Create DataFrame with trained_features order
        input_df = pd.DataFrame([feature_dict])[trained_features]
        
        # Predict
        if model_choice == "Linear Regression":
            input_scaled = scaler.transform(input_df)
            prediction = lr_model.predict(input_scaled)[0]
        else:
            prediction = rf_model.predict(input_df)[0]
        
        # Display result
        st.markdown("---")
        st.success(f"### Predicted GDP Growth for Year {selected_year}: {prediction:.2f}%")
        
        # Plot historical + prediction
        st.subheader("📈 GDP Growth Trend")
        
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Historical data
        ax.plot(df['Year'], df['GDP_Growth'], marker='o', label='Historical GDP Growth', linewidth=2)
        
        # Single prediction point
        if not show_forecast:
            ax.scatter([selected_year], [prediction], color='red', s=100, zorder=5, label=f'Predicted ({selected_year})')
        else:
            # Forecast range
            if start_year <= end_year:
                forecast_years = list(range(start_year, end_year + 1))
                forecast_values = [prediction] * len(forecast_years)  # Same inputs for all years
                ax.plot(forecast_years, forecast_values, marker='s', linestyle='--', color='orange', 
                       linewidth=2, label='Scenario-based forecast (inputs fixed)')
        
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('GDP Growth (%)', fontsize=12)
        ax.set_title('Sri Lanka GDP Growth: Historical vs Predicted', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        
        # Show engineered features
        with st.expander("🔧 View Engineered Features"):
            st.dataframe(input_df, use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ Prediction Error: {str(e)}")

st.markdown("---")
st.markdown("<div style='text-align: center; color: gray;'><small>Sri Lanka Economy Predictor</small></div>", 
           unsafe_allow_html=True)
