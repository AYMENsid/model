import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import os
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(
    page_title="Multi-Commodity SARIMA Prediction Dashboard",
    layout="wide"
)

# Dashboard title
st.title("Multi-Commodity SARIMA Prediction Dashboard")
st.markdown("Visualize SARIMA model predictions for different fuel commodities")

# Sidebar for controls
st.sidebar.header("Dashboard Controls")

# Function to load data
def load_data(file_path=None):
    try:
        if file_path:
            # Use the provided file path
            data = pd.read_csv(file_path)
        else:
            # Use uploaded file
            data = pd.read_csv(uploaded_file)
       
        # Convert date to datetime
        data['date'] = pd.to_datetime(data['date'])
       
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Function to generate predictions using SARIMA
def generate_sarima_predictions(data, train_size_pct=0.8):
    # Extract close prices
    prices = data['close']
   
    # Split into train and test
    train_size = int(len(prices) * train_size_pct)
    train_data = prices[:train_size]
    test_data = prices[train_size:]
   
    if len(train_data) < 10 or len(test_data) < 5:
        st.warning("Not enough data points for reliable prediction. Consider using more data.")
        # Use default values to avoid errors
        return train_data, test_data, pd.DataFrame(), None, 0, 0, 0, 0
   
    # Fit SARIMA model
    try:
        # Note: In a real scenario, you would tune these parameters
        model = SARIMAX(
            train_data,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 7),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
       
        with st.spinner("Fitting SARIMA model... This may take a moment."):
            model_fit = model.fit(disp=False)
       
        # Make predictions
        forecast = model_fit.get_forecast(steps=len(test_data))
        forecast_mean = forecast.predicted_mean
        forecast_ci = forecast.conf_int()
       
        # Create predictions dataframe
        predictions = pd.DataFrame(forecast_mean)
        predictions.index = test_data.index
        predictions.columns = ['prediction']
       
        # Add confidence intervals
        predictions['lower_ci'] = forecast_ci.iloc[:, 0].values
        predictions['upper_ci'] = forecast_ci.iloc[:, 1].values
       
        # Combine actual and predictions
        results = pd.DataFrame(test_data)
        results.columns = ['actual']
        results = pd.merge(
            results,
            predictions,
            left_index=True,
            right_index=True
        )
       
        # Calculate error metrics
        mae = mean_absolute_error(results['actual'], results['prediction'])
        mse = mean_squared_error(results['actual'], results['prediction'])
        rmse = np.sqrt(mse)
        r2 = r2_score(results['actual'], results['prediction'])
       
        return train_data, test_data, results, model_fit, mae, mse, rmse, r2
   
    except Exception as e:
        st.error(f"Error in SARIMA modeling: {e}")
        return train_data, test_data, pd.DataFrame(), None, 0, 0, 0, 0

# Data source selection
data_source = st.sidebar.radio(
    "Select data source",
    ["local file", "upload"],
    index=0
)

# File path input
file_path = None
if data_source == "local file":
    file_path = st.sidebar.text_input(
        "Enter the path to your CSV file",
        value="C:\\Users\\WINDOWS\\Downloads\\all_fuels_data.csv"
    )
   
    if not os.path.exists(file_path):
        st.warning(f"File not found: {file_path}")
        st.info("Please enter a valid file path or upload a file.")
       
        # Create a sample data option if file not found
        if st.sidebar.checkbox("Use sample data instead", value=True):
            sample_data = """ticker,commodity,date,open,high,low,close,volume
HO=F,Heating Oil,2006-02-14,1.6410000324249268,1.6442999839782715,1.6074999570846558,1.6100000143051147,21859
HO=F,Heating Oil,2006-02-15,1.6089999675750732,1.6339999437332153,1.6030000448226929,1.6074999570846558,24195
HO=F,Heating Oil,2006-02-16,1.6075999736785889,1.6399999856948853,1.597000002861023,1.6253000497817993,25697
NG=F,Natural Gas,2007-08-27,5.293000221252441,5.433000087738037,5.191999912261963,5.380000114440918,47271
NG=F,Natural Gas,2007-08-28,5.428999900817871,5.728000164031982,5.364999771118164,5.5929999351501465,76715
NG=F,Natural Gas,2007-08-29,5.670000076293945,5.75,5.36899995803833,5.429999828338623,42514
NG=F,Natural Gas,2007-08-30,5.5929999351501465,5.784999847412109,5.520999908447266,5.635000228881836,43778
RB=F,RBOB Gasoline,2003-12-18,0.917900025844574,0.9380000233650208,0.9070000052452087,0.9272000193595886,23442
RB=F,RBOB Gasoline,2003-12-19,0.9290000200271606,0.9290000200271606,0.9049999713897705,0.9049999713897705,21601
RB=F,RBOB Gasoline,2003-12-22,0.9070000052452087,0.9070000052452087,0.8690000176429749,0.8716999888420105,16932
BZ=F,Brent Crude Oil,2010-05-17,77.62999725341797,77.62999725341797,75.06999969482422,75.0999984741211,20
BZ=F,Brent Crude Oil,2010-05-18,76.48999786376953,76.48999786376953,74.05000305175781,74.43000030517578,32
CL=F,Crude Oil,2000-08-23,31.950000762939453,32.79999923706055,31.950000762939453,32.04999923706055,79385
CL=F,Crude Oil,2000-08-24,31.899999618530273,32.2400016784668,31.399999618530273,31.6299991607666,72978
CL=F,Crude Oil,2000-08-25,31.700000762939453,32.099998474121094,31.31999969482422,32.04999923706055,44601"""
            data = pd.read_csv(io.StringIO(sample_data))
            data['date'] = pd.to_datetime(data['date'])
        else:
            st.stop()
    else:
        # Load the data
        data = load_data(file_path)
else:
    uploaded_file = st.sidebar.file_uploader("Upload your fuel data (CSV)", type="csv")
   
    if uploaded_file is None:
        st.warning("Please upload your data file or select 'local file' option")
        st.stop()
    else:
        # Load the data
        data = load_data()

if data is None:
    st.error("Failed to load data. Please check your file and try again.")
    st.stop()

# Get unique commodities
commodities = data['commodity'].unique()

# Commodity selection
selected_commodity = st.sidebar.selectbox(
    "Select commodity to analyze",
    options=commodities
)

# Filter data for selected commodity
filtered_data = data[data['commodity'] == selected_commodity].copy()
filtered_data.set_index('date', inplace=True)
filtered_data.sort_index(inplace=True)

# Display data info
st.sidebar.subheader("Data Information")
st.sidebar.write(f"Selected Commodity: {selected_commodity}")
st.sidebar.write(f"Date Range: {filtered_data.index.min().date()} to {filtered_data.index.max().date()}")
st.sidebar.write(f"Number of records: {len(filtered_data)}")

# SARIMA parameters
st.sidebar.subheader("SARIMA Parameters")
train_size_pct = st.sidebar.slider("Training data percentage", 0.5, 0.9, 0.8, 0.05)

# Generate predictions
if len(filtered_data) > 10:  # Ensure we have enough data
    train_data, test_data, results, model_fit, mae, mse, rmse, r2 = generate_sarima_predictions(filtered_data, train_size_pct)
else:
    st.error(f"Not enough data points for {selected_commodity}. Please select a different commodity.")
    st.stop()

# Main dashboard area
tab1, tab2, tab3 = st.tabs(["Price Predictions", "Model Performance", "Technical Analysis"])

with tab1:
    st.header(f"{selected_commodity} Price Predictions")
   
    # Create subplot with 2 rows
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("Historical Prices & Predictions", "Trading Volume"),
        row_heights=[0.7, 0.3]
    )
   
    # Add candlestick chart for the historical data
    fig.add_trace(
        go.Candlestick(
            x=filtered_data.index,
            open=filtered_data['open'],
            high=filtered_data['high'],
            low=filtered_data['low'],
            close=filtered_data['close'],
            name="Historical Prices",
            increasing_line_color='green',
            decreasing_line_color='red'
        ),
        row=1, col=1
    )
   
    # Add line for training data close prices
    fig.add_trace(
        go.Scatter(
            x=train_data.index,
            y=train_data.values,
            mode='lines',
            name='Training Data (Close)',
            line=dict(color='blue', width=1)
        ),
        row=1, col=1
    )
   
    # Add actual test values and predictions if available
    if not results.empty:
        # Add actual test values
        fig.add_trace(
            go.Scatter(
                x=results.index,
                y=results['actual'],
                mode='lines',
                name='Actual Test Values',
                line=dict(color='black', width=2)
            ),
            row=1, col=1
        )
       
        # Add predictions
        fig.add_trace(
            go.Scatter(
                x=results.index,
                y=results['prediction'],
                mode='lines',
                name='SARIMA Predictions',
                line=dict(color='orange', width=2)
            ),
            row=1, col=1
        )
       
        # Add confidence intervals
        fig.add_trace(
            go.Scatter(
                x=results.index.tolist() + results.index.tolist()[::-1],
                y=results['upper_ci'].tolist() + results['lower_ci'].tolist()[::-1],
                fill='toself',
                fillcolor='rgba(255,165,0,0.2)',
                line=dict(color='rgba(255,165,0,0)'),
                name='95% Confidence Interval'
            ),
            row=1, col=1
        )
   
    # Add volume bar chart
    fig.add_trace(
        go.Bar(
            x=filtered_data.index,
            y=filtered_data['volume'],
            name='Volume',
            marker_color='rgba(0, 0, 255, 0.5)'
        ),
        row=2, col=1
    )
   
    # Update layout
    fig.update_layout(
        height=800,
        title_text=f"{selected_commodity} Price Prediction Analysis",
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
   
    # Update y-axis labels
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
   
    st.plotly_chart(fig, use_container_width=True)
   
    # Display the prediction results
    if not results.empty:
        st.subheader("Prediction Results")
        st.dataframe(results)
    else:
        st.warning("No prediction results available. The model may have failed to fit properly.")

with tab2:
    st.header("Model Performance Metrics")
   
    if not results.empty and model_fit is not None:
        # Create metrics display
        col1, col2, col3, col4 = st.columns(4)
       
        with col1:
            st.metric("MAE", f"{mae:.4f}")
       
        with col2:
            st.metric("MSE", f"{mse:.4f}")
       
        with col3:
            st.metric("RMSE", f"{rmse:.4f}")
       
        with col4:
            st.metric("R² Score", f"{r2:.4f}")
       
        # Error analysis
        st.subheader("Error Analysis")
       
        # Calculate errors
        results['error'] = results['actual'] - results['prediction']
        results['pct_error'] = (results['error'] / results['actual']) * 100
       
        # Create error plots
        error_fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=("Prediction Errors Over Time", "Percentage Error Distribution"),
            row_heights=[0.5, 0.5]
        )
       
        # Add error over time
        error_fig.add_trace(
            go.Scatter(
                x=results.index,
                y=results['error'],
                mode='lines+markers',
                name='Prediction Error',
                line=dict(color='red')
            ),
            row=1, col=1
        )
       
        # Add zero line
        error_fig.add_trace(
            go.Scatter(
                x=[results.index.min(), results.index.max()],
                y=[0, 0],
                mode='lines',
                name='Zero Error',
                line=dict(color='black', dash='dash')
            ),
            row=1, col=1
        )
       
        # Add error histogram
        error_fig.add_trace(
            go.Histogram(
                x=results['pct_error'],
                name='Percentage Error Distribution',
                marker_color='rgba(255, 0, 0, 0.5)',
                nbinsx=20
            ),
            row=2, col=1
        )
       
        # Update layout
        error_fig.update_layout(
            height=600,
            showlegend=False
        )
       
        # Update y-axis labels
        error_fig.update_yaxes(title_text="Error (USD)", row=1, col=1)
        error_fig.update_yaxes(title_text="Frequency", row=2, col=1)
        error_fig.update_xaxes(title_text="Percentage Error (%)", row=2, col=1)
       
        st.plotly_chart(error_fig, use_container_width=True)
       
        # Model summary
        st.subheader("SARIMA Model Summary")
       
        # Display model parameters
        st.write("**Model Parameters:**")
        st.write(f"- Order (p,d,q): (1,1,1)")
        st.write(f"- Seasonal Order (P,D,Q,s): (1,1,1,7)")
       
        # Display model summary in an expander
        with st.expander("View Detailed Model Summary"):
            # Capture the model summary
            buffer = io.StringIO()
            model_fit.summary().tables[1].to_csv(buffer)
            summary_df = pd.read_csv(io.StringIO(buffer.getvalue()))
           
            st.dataframe(summary_df)
    else:
        st.warning("No model performance metrics available. The model may have failed to fit properly.")

with tab3:
    st.header("Technical Analysis")
   
    # Calculate technical indicators
    data_ta = filtered_data.copy()
   
    # Calculate Moving Averages (only if we have enough data)
    if len(data_ta) >= 50:
        data_ta['SMA_20'] = data_ta['close'].rolling(window=20).mean()
        data_ta['SMA_50'] = data_ta['close'].rolling(window=50).mean()
       
        # Calculate Bollinger Bands
        data_ta['stddev'] = data_ta['close'].rolling(window=20).std()
        data_ta['Upper_Band'] = data_ta['SMA_20'] + (data_ta['stddev'] * 2)
        data_ta['Lower_Band'] = data_ta['SMA_20'] - (data_ta['stddev'] * 2)
       
        # Calculate RSI (Relative Strength Index)
        delta = data_ta['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        data_ta['RSI'] = 100 - (100 / (1 + rs))
       
        # Create technical analysis chart
        ta_fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=("Price with Technical Indicators", "RSI (14)"),
            row_heights=[0.7, 0.3]
        )
       
        # Add candlestick chart
        ta_fig.add_trace(
            go.Candlestick(
                x=filtered_data.index,
                open=filtered_data['open'],
                high=filtered_data['high'],
                low=filtered_data['low'],
                close=filtered_data['close'],
                name="Price",
                increasing_line_color='green',
                decreasing_line_color='red'
            ),
            row=1, col=1
        )
       
        # Add Moving Averages
        ta_fig.add_trace(
            go.Scatter(
                x=data_ta.index,
                y=data_ta['SMA_20'],
                mode='lines',
                name='SMA (20)',
                line=dict(color='blue', width=1)
            ),
            row=1, col=1
        )
       
        ta_fig.add_trace(
            go.Scatter(
                x=data_ta.index,
                y=data_ta['SMA_50'],
                mode='lines',
                name='SMA (50)',
                line=dict(color='orange', width=1)
            ),
            row=1, col=1
        )
       
        # Add Bollinger Bands
        ta_fig.add_trace(
            go.Scatter(
                x=data_ta.index,
                y=data_ta['Upper_Band'],
                mode='lines',
                name='Upper Bollinger Band',
                line=dict(color='rgba(173, 204, 255, 0.7)')
            ),
            row=1, col=1
        )
       
        ta_fig.add_trace(
            go.Scatter(
                x=data_ta.index,
                y=data_ta['Lower_Band'],
                mode='lines',
                name='Lower Bollinger Band',
                line=dict(color='rgba(173, 204, 255, 0.7)'),
                fill='tonexty',
                fillcolor='rgba(173, 204, 255, 0.2)'
            ),
            row=1, col=1
        )
       
        # Add RSI
        ta_fig.add_trace(
            go.Scatter(
                x=data_ta.index,
                y=data_ta['RSI'],
                mode='lines',
                name='RSI (14)',
                line=dict(color='purple', width=1)
            ),
            row=2, col=1
        )
       
        # Add RSI overbought/oversold lines
        ta_fig.add_trace(
            go.Scatter(
                x=[data_ta.index.min(), data_ta.index.max()],
                y=[70, 70],
                mode='lines',
                name='Overbought (70)',
                line=dict(color='red', dash='dash')
            ),
            row=2, col=1
        )
       
        ta_fig.add_trace(
            go.Scatter(
                x=[data_ta.index.min(), data_ta.index.max()],
                y=[30, 30],
                mode='lines',
                name='Oversold (30)',
                line=dict(color='green', dash='dash')
            ),
            row=2, col=1
        )
       
        # Update layout
        ta_fig.update_layout(
            height=800,
            title_text=f"{selected_commodity} Technical Analysis Indicators",
            xaxis_rangeslider_visible=False,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
       
        # Update y-axis labels
        ta_fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
        ta_fig.update_yaxes(title_text="RSI", row=2, col=1)
       
        st.plotly_chart(ta_fig, use_container_width=True)
    else:
        st.warning(f"Not enough data points for technical analysis of {selected_commodity}. Need at least 50 data points.")

# Add a commodity comparison tab
tab4 = st.tabs(["Commodity Comparison"])[0]

with tab4:
    st.header("Commodity Comparison")
   
    # Select commodities to compare
    compare_commodities = st.multiselect(
        "Select commodities to compare",
        options=commodities,
        default=[selected_commodity]
    )
   
    if len(compare_commodities) > 0:
        # Create comparison dataframe
        comparison_data = pd.DataFrame()
       
        # Get data for each selected commodity
        for commodity in compare_commodities:
            # Filter data for this commodity
            commodity_data = data[data['commodity'] == commodity].copy()
            commodity_data['date'] = pd.to_datetime(commodity_data['date'])
            commodity_data.set_index('date', inplace=True)
            commodity_data.sort_index(inplace=True)
           
            # Add to comparison dataframe
            if comparison_data.empty:
                comparison_data[commodity] = commodity_data['close']
            else:
                comparison_data[commodity] = commodity_data['close']
       
        # Create comparison plot
        fig = go.Figure()
       
        # Add prices for each commodity
        for commodity in compare_commodities:
            fig.add_trace(
                go.Scatter(
                    x=comparison_data.index,
                    y=comparison_data[commodity],
                    mode='lines',
                    name=commodity
                )
            )
       
        # Update layout
        fig.update_layout(
            height=600,
            title_text="Commodity Price Comparison",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
       
        st.plotly_chart(fig, use_container_width=True)
       
        # Create normalized comparison (percentage change from first date)
        st.subheader("Normalized Price Comparison (% Change)")
       
        # Create normalized dataframe
        normalized_data = pd.DataFrame(index=comparison_data.index)
       
        for commodity in compare_commodities:
            # Calculate percentage change from first value
            first_value = comparison_data[commodity].dropna().iloc[0]
            normalized_data[commodity] = (comparison_data[commodity] / first_value - 1) * 100
       
        # Create normalized comparison plot
        fig = go.Figure()
       
        # Add normalized prices for each commodity
        for commodity in compare_commodities:
            fig.add_trace(
                go.Scatter(
                    x=normalized_data.index,
                    y=normalized_data[commodity],
                    mode='lines',
                    name=commodity
                )
            )
       
        # Add zero line
        fig.add_trace(
            go.Scatter(
                x=[normalized_data.index.min(), normalized_data.index.max()],
                y=[0, 0],
                mode='lines',
                name='Baseline',
                line=dict(color='black', dash='dash')
            )
        )
       
        # Update layout
        fig.update_layout(
            height=600,
            title_text="Normalized Price Comparison (% Change from First Date)",
            xaxis_title="Date",
            yaxis_title="Percentage Change (%)",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
       
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Please select at least one commodity to compare")

# Download section
st.header("Download Results")

# Convert results to CSV
if 'results' in locals() and not results.empty:
    csv = results.to_csv()
   
    # Create download button
    st.download_button(
        label=f"Download {selected_commodity} Prediction Results as CSV",
        data=csv,
        file_name=f"{selected_commodity.replace(' ', '_').lower()}_sarima_predictions.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.markdown("Multi-Commodity SARIMA Prediction Dashboard | Created with Streamlit")

# Display instructions
with st.expander("How to use this dashboard"):
    st.markdown("""
    ### Instructions
   
    1. **Data Input**:
       - The dashboard will automatically load your data from: `C:\\Users\\WINDOWS\\Downloads\\all_fuels_data.csv`
       - You can change the file path in the sidebar if needed
       - Or upload your data file directly
   
    2. **Commodity Selection**:
       - Select which commodity you want to analyze from the dropdown menu
       - Each commodity will be analyzed separately
       - In the "Commodity Comparison" tab, you can compare multiple commodities
   
    3. **Model Parameters**:
       - Adjust the training/testing split ratio using the slider
   
    4. **Visualization**:
       - The "Price Predictions" tab shows historical prices, actual test values, and SARIMA predictions
       - The "Model Performance" tab displays error metrics and error analysis
       - The "Technical Analysis" tab provides additional technical indicators
       - The "Commodity Comparison" tab allows you to compare different commodities
   
    5. **Download**:
       - Download the prediction results for further analysis
   
    ### About SARIMA Models
   
    SARIMA (Seasonal AutoRegressive Integrated Moving Average) models are an extension of ARIMA models that support seasonality in time series data. The model is specified as SARIMA(p,d,q)(P,D,Q,s) where:
   
    - p: Order of the autoregressive part
    - d: Degree of differencing
    - q: Order of the moving average part
    - P, D, Q: Seasonal equivalents of p, d, q
    - s: Length of the seasonal cycle
   
    SARIMA models are particularly useful for financial time series like fuel prices that often exhibit both trend and seasonal patterns.
    """)

print("Dashboard is running. Please view the Streamlit interface.")