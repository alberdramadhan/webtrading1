from tvDatafeed import TvDatafeed, Interval
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from flask import Flask, render_template_string
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import threading
import time
from flask import Flask, request, redirect, url_for, session, render_template
import pickle
from datetime import datetime, timedelta

# Initialize TvDatafeed
tv = TvDatafeed()

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Ganti dengan kunci rahasia yang kuat

# Global variables to store the last 20 predictions and the last model's prediction
prediction_history = []
latest_prediction = {'datetime': None, 'actual_close': None, 'predicted_close': None}

def validate_credentials(input_username, input_password, filename='credentials.pkl'):
    try:
        with open(filename, 'rb') as file:
            credentials = pickle.load(file)
            if (credentials['username'] == input_username and
                credentials['password'] == input_password):
                expiry_timestamp = credentials['expiry']
                if datetime.now().timestamp() < expiry_timestamp:
                    return True, expiry_timestamp
                else:
                    return False, None
            else:
                return False, None
    except FileNotFoundError:
        return False, None

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        is_valid, expiry_timestamp = validate_credentials(username, password)
        if is_valid:
            session['logged_in'] = True
            expiry_time = datetime.fromtimestamp(expiry_timestamp)
            expiry_duration = f'valid until {expiry_time.strftime("%Y-%m-%d %H:%M:%S")}'
            return redirect('/dashboard')
        else:
            return render_template('index.html', message='Invalid credentials or expired', expiry_duration='Unknown')
    
    return render_template('index.html', expiry_duration='1 day')

def fetch_data():
    # Fetch historical data for XAUUSD
    data = tv.get_hist(symbol='XAUUSD', exchange='OANDA', interval=Interval.in_daily, n_bars=1000)
    df = pd.DataFrame(data)

    # Reset index to include datetime as a column
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'datetime'}, inplace=True)

    # Set datetime as the index and sort
    df.set_index('datetime', inplace=True)
    df.sort_index(inplace=True)

    # Feature Engineering
    df['price_change'] = df['close'].pct_change()
    df['volatility'] = df['close'].rolling(window=24).std()
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['SMA_50'] = df['close'].rolling(window=50).mean()

    # Calculate RSI manually
    def calculate_rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    df['RSI'] = calculate_rsi(df['close'])

    # Create a binary target variable
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)  # 1 if price goes up, 0 if it goes down

    # Drop rows with NaN values
    df.dropna(inplace=True)
    
    return df

# Model Training Function
def train_model(df):
    # Define features and target
    features = df[['price_change', 'volatility', 'SMA_20', 'SMA_50', 'RSI']]
    target = df['target']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Initialize XGBoost Classifier
    model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss')

    # Train the model
    model.fit(X_train, y_train)

    return model

# Function to make and store the latest prediction and last 20 predictions
def make_predictions(df, model):
    global prediction_history, latest_prediction

    # Get the latest data point for prediction
    last_data = df[['price_change', 'volatility', 'SMA_20', 'SMA_50', 'RSI']].iloc[-1:]
    predicted_direction = model.predict(last_data)[0]
    
    # Calculate the predicted price based on the direction (BUY or SELL)
    predicted_price = df.iloc[-1]['close'] * (1 + 0.01) if predicted_direction == 1 else df.iloc[-1]['close'] * (1 - 0.01)
    
    # Predict for tomorrow (next datetime after the last one)
    prediction_datetime = df.index[-1] + pd.Timedelta(days=1)
    
    # Update the latest prediction to reflect tomorrow's predicted price
    latest_prediction = {
        'datetime': prediction_datetime,
        'actual_close': df.iloc[-1]['close'],
        'predicted_close': predicted_price
    }

    # Update prediction history (last 20 rows)
    prediction_history = []
    for i in range(20):
        if len(df) - 21 + i >= 0:
            row = df.iloc[len(df) - 21 + i]
            row_data = df[['price_change', 'volatility', 'SMA_20', 'SMA_50', 'RSI']].iloc[len(df) - 21 + i:len(df) - 21 + i + 1]
            pred_dir = model.predict(row_data)[0]
            pred_price = row['close'] * (1 + 0.01) if pred_dir == 1 else row['close'] * (1 - 0.01)
            
            prediction_history.append({
                'datetime': row.name,
                'actual_close': row['close'],
                'predicted_close': pred_price,
                'Tommorow prediction': pred_dir
            })

# Flask Route to Render HTML
@app.route('/dashboard')
def index():
    # Create Plotly Figure
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['close'], mode='lines', name='Close Price'))

    # Plot the latest prediction as a red point on the graph for today
    if latest_prediction['datetime'] is not None:
        # Use the close price from yesterday for prediction
        predicted_close_from_yesterday = df.iloc[-2]['close'] * (1 + 0.01) if latest_prediction['predicted_close'] > df.iloc[-2]['close'] else df.iloc[-2]['close'] * (1 - 0.01)
        fig.add_trace(go.Scatter(
            x=[df.index[-1]],  # Today's date
            y=[predicted_close_from_yesterday],  # Predicted close price based on yesterday's close price
            mode='markers',
            marker=dict(color='red', size=10),
            name='Predicted Price'
        ))

    fig.update_layout(
        title="XAUUSD Price Prediction with Prediction History",
        xaxis_title="Datetime",
        yaxis_title="Close Price",
    )

    # Generate HTML for Plotly Graph
    graph_html = fig.to_html(full_html=False)

    # Prepare table data
    table_data = [[row['datetime'].strftime('%Y-%m-%d %H:%M:%S'), row['actual_close'], row['predicted_close'], row['Tommorow prediction']] for row in prediction_history]

    # HTML template to display the graph and table
    template = """
 <!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>XAUUSD Price Prediction</title> <!-- Title for the browser tab -->
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f4f4f4;
            color: #333;
        }
        header {
            color: white; /* Text color */
            padding: 20px; /* Space around the text */
            text-align: center; /* Center align text */
            font-size: 24px; /* Adjust font size if needed */
            font-weight: bold; /* Make the text bold for better visibility */
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); /* Optional: add shadow for better contrast */
        }
        .container {
            width: 90%;
            margin: auto;
            background: white;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1, h2 {
            color: #007bff;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background-color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #f4f4f4;
        }
        tr:hover {
            background-color: #f1f1f1;
        }
        .chart-container {
            text-align: center;
            margin: 20px 0;
        }
    </style>
</head>
<body>
    <header>
        <h1>XAUUSD Price Prediction Dashboard</h1>
    </header>
    
    <div class="container">
        <h2>Latest Prediction</h2>
        <p><strong>Datetime:</strong> {{ latest_prediction['datetime'] }}</p>
        <p><strong>Actual Close Price:</strong> {{ latest_prediction['actual_close'] }}</p>
        <p><strong>Predicted Close Price:</strong> {{ latest_prediction['predicted_close'] }}</p>

        <h2>Predictions History Table (Last 20 Predictions)</h2>
        <table>
            <thead>
                <tr>
                    <th>Datetime</th>
                    <th>Actual Close</th>
                    <th>Predicted Close</th>
                    <th>Tomorrow Prediction</th>
                </tr>
            </thead>
            <tbody>
                {% for row in table_data %}
                <tr>
                    <td>{{ row[0] }}</td>
                    <td>{{ row[1] }}</td>
                    <td>{{ row[2] }}</td>
                    <td>{{ 'BUY' if row[3] == 1 else 'SELL' }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>

        <div class="chart-container">
            <h2>Price Graph</h2>
            {{ graph_html|safe }}
        </div>
    </div>
</body>
</html>
    """
    
    # Render the HTML with Plotly graph and table
    return render_template_string(template, latest_prediction=latest_prediction, graph_html=graph_html, table_data=table_data)
# Function to Fetch Real-Time Data and Retrain Model Periodically
def update_data():
    global df, model
    while True:
        df = fetch_data()
        model = train_model(df)
        make_predictions(df, model)  # Update the last 20 predictions with the latest data
        time.sleep(60)  # Fetch new data every 60 seconds

# Start background thread for real-time data fetching and retraining
threading.Thread(target=update_data, daemon=True).start()

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)