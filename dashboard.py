import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Initialize Dash app
app = dash.Dash(__name__)

# Sample training data for the model
np.random.seed(42)

neem_data = {
    "Initial Moisture (%)": np.random.uniform(30, 60, 20),
    "Drying Time (mins)": np.random.uniform(10, 12, 20),
}
eucalyptus_data = {
    "Initial Moisture (%)": np.random.uniform(30, 110, 20),
    "Drying Time (mins)": np.random.uniform(15, 17, 20),
}

neem_df = pd.DataFrame(neem_data)
eucalyptus_df = pd.DataFrame(eucalyptus_data)

# Train models
neem_model = RandomForestRegressor(random_state=42)
neem_model.fit(neem_df[["Initial Moisture (%)"]], neem_df["Drying Time (mins)"])

eucalyptus_model = RandomForestRegressor(random_state=42)
eucalyptus_model.fit(eucalyptus_df[["Initial Moisture (%)"]], eucalyptus_df["Drying Time (mins)"])

# Function to predict drying time
def predict_drying(initial_moisture, veneer_type):
    if veneer_type == "Neem":
        model = neem_model
    elif veneer_type == "Eucalyptus":
        model = eucalyptus_model
    else:
        return None
    return round(model.predict([[initial_moisture]])[0], 2)

# Layout of the dashboard
app.layout = html.Div([
    html.H1("Veneer Drying Optimization Dashboard", style={'textAlign': 'center'}),
    
    # User inputs
    html.Div([
        html.Label("Select Veneer Type"),
        dcc.Dropdown(
            id="veneer_type",
            options=[{'label': 'Neem', 'value': 'Neem'}, {'label': 'Eucalyptus', 'value': 'Eucalyptus'}],
            value='Neem'
        ),
        html.Label("Enter Initial Moisture Content (%)"),
        dcc.Input(id="moisture_input", type="number", value=40, min=30, max=110, step=0.5),
        html.Button("Predict Drying Time", id="predict_btn", n_clicks=0, style={'marginTop': '10px'})
    ], style={'width': '50%', 'margin': 'auto'}),

    html.Br(),
    
    # Output results
    html.Div(id="output_prediction", style={'textAlign': 'center', 'fontSize': 20, 'color': 'blue'}),
    
    # Comparison Chart
    dcc.Graph(id="comparison_chart")
])

# Callback to update prediction and graph
@app.callback(
    [Output("output_prediction", "children"), Output("comparison_chart", "figure")],
    [Input("predict_btn", "n_clicks")],
    [dash.State("moisture_input", "value"), dash.State("veneer_type", "value")]
)
def update_output(n_clicks, initial_moisture, veneer_type):
    if n_clicks == 0:
        return "", go.Figure()
    
    optimized_time = predict_drying(initial_moisture, veneer_type)
    standard_time = 20  # Fixed standard drying time

    if optimized_time is None:
        return "Invalid Veneer Type!", go.Figure()

    # Create comparison chart
    fig = go.Figure()
    fig.add_trace(go.Bar(x=['Standard Drying Time', 'Optimized Drying Time'], y=[standard_time, optimized_time], 
                         marker_color=['red', 'green'], text=[f"{standard_time} min", f"{optimized_time} min"],
                         textposition="auto"))
    fig.update_layout(title="Optimized vs Standard Drying Time", yaxis_title="Time (mins)")

    return f"Predicted Drying Time: {optimized_time} mins", fig

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=10000, debug=True)

