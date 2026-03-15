#"Trust in God with all your Heart"


import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, r2_score

# ML Models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

st.set_page_config(page_title="ML Lorenz Dashboard", layout="wide")
st.title("ML Lorenz Attractor Dashboard")

# -------------------------
# Sidebar - Simulation Settings
# -------------------------
st.sidebar.header("Simulation Settings")
x0 = st.sidebar.slider("Initial x", -10.0, 10.0, 0.1)
y0 = st.sidebar.slider("Initial y", -10.0, 10.0, 0.0)
z0 = st.sidebar.slider("Initial z", -10.0, 10.0, 0.0)
num_steps = st.sidebar.slider("Simulation steps", 100, 2000, 500)
future_steps = st.sidebar.slider("Future prediction steps", 0, 500, 100)

# -------------------------
# Lorenz Parameters
# -------------------------
sigma, rho, beta = 10, 28, 8/3
dt = 0.01

# -------------------------
# Generate Lorenz Data
# -------------------------
x = np.zeros(num_steps)
y = np.zeros(num_steps)
z = np.zeros(num_steps)

x[0], y[0], z[0] = x0, y0, z0

for i in range(num_steps-1):
    x[i+1] = x[i] + sigma*(y[i]-x[i])*dt
    y[i+1] = y[i] + (x[i]*(rho - z[i]) - y[i])*dt
    z[i+1] = z[i] + (x[i]*y[i] - beta*z[i])*dt

actual = np.column_stack((x, y, z))

# -------------------------
# Prepare ML Training Data
# -------------------------
X = actual[:-1]
y_target = actual[1:]

# -------------------------
# Train ML Models
# -------------------------

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X, y_target)
lr_pred = lr_model.predict(X)

# Random Forest
rf_model = RandomForestRegressor(n_estimators=100)
rf_model.fit(X, y_target)
rf_pred = rf_model.predict(X)

# Decision Tree
dt_model = DecisionTreeRegressor()
dt_model.fit(X, y_target)
dt_pred = dt_model.predict(X)

# KNN
knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X, y_target)
knn_pred = knn_model.predict(X)

# -------------------------
# ML Predictions
# -------------------------
predicted_models = {
    "Linear Regression": lr_pred,
    "Random Forest": rf_pred,
    "Decision Tree Regressor": dt_pred,
    "KNN Regressor": knn_pred
}

# Select models to display
selected_models = st.multiselect(
    "Select ML Models to Compare",
    options=list(predicted_models.keys()),
    default=list(predicted_models.keys())
)

# -------------------------
# 3D Lorenz Plot
# -------------------------
st.subheader("3D Lorenz Attractor (Actual vs Predictions)")

fig = go.Figure()

fig.add_trace(go.Scatter3d(
    x=actual[:,0],
    y=actual[:,1],
    z=actual[:,2],
    mode='lines',
    name='Actual Trajectory',
    line=dict(color='blue', width=3)
))

for model_name in selected_models:
    pred = predicted_models[model_name]

    fig.add_trace(go.Scatter3d(
        x=pred[:,0],
        y=pred[:,1],
        z=pred[:,2],
        mode='lines',
        name=model_name,
        line=dict(width=3)
    ))

fig.update_layout(
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z'
    )
)

st.plotly_chart(
    fig,
    use_container_width=True,
    config={
        'scrollZoom': True,
        'displayModeBar': False
    }
)

# -------------------------
# 2D Projections
# -------------------------
st.subheader("2D Projections")

fig2 = go.Figure()

fig2.add_trace(go.Scatter(y=actual[:,0], mode='lines', name='Actual x'))
fig2.add_trace(go.Scatter(y=actual[:,1], mode='lines', name='Actual y'))
fig2.add_trace(go.Scatter(y=actual[:,2], mode='lines', name='Actual z'))

for model_name in selected_models:
    pred = predicted_models[model_name]

    fig2.add_trace(go.Scatter(y=pred[:,0], mode='lines', name=f'{model_name} x'))
    fig2.add_trace(go.Scatter(y=pred[:,1], mode='lines', name=f'{model_name} y'))
    fig2.add_trace(go.Scatter(y=pred[:,2], mode='lines', name=f'{model_name} z'))

st.plotly_chart(fig2, use_container_width=True)

# -------------------------
# Metrics
# -------------------------
st.subheader("Prediction Metrics")

actual_compare = actual[1:]

for model_name in selected_models:

    pred = predicted_models[model_name]

    mse = mean_squared_error(actual_compare, pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual_compare, pred)

    st.write(f"**{model_name}:** MSE={mse:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")

# -------------------------
# Future Predictions
# -------------------------
if st.button("Predict Future Points"):

    for model_name in selected_models:

        pred = predicted_models[model_name]
        current_point = pred[-1].reshape(1,-1)

        future_points = []

        for _ in range(future_steps):
            next_point = current_point * 1.01
            future_points.append(next_point.flatten())
            current_point = next_point

        future_points = np.array(future_points)

        fig3 = go.Figure()

        fig3.add_trace(go.Scatter3d(
            x=actual[:,0],
            y=actual[:,1],
            z=actual[:,2],
            mode='lines',
            name='Actual',
            line=dict(color='blue')
        ))

        fig3.add_trace(go.Scatter3d(
            x=pred[:,0],
            y=pred[:,1],
            z=pred[:,2],
            mode='lines',
            name=model_name,
            line=dict(color='red')
        ))

        if future_steps > 0:
            fig3.add_trace(go.Scatter3d(
                x=future_points[:,0],
                y=future_points[:,1],
                z=future_points[:,2],
                mode='lines',
                name=f'{model_name} Future',
                line=dict(color='green', dash='dash')
            ))

        fig3.update_layout(scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ))

        st.plotly_chart(
            fig3,
            use_container_width=True,
            config={
                'scrollZoom': True,
                'displayModeBar': False
            }
        )

        df_future = pd.DataFrame(future_points, columns=['x','y','z'])
        csv = df_future.to_csv(index=False).encode()

        st.download_button(
            f"Download Future CSV for {model_name}",
            data=csv,
            file_name=f"future_points_{model_name}.csv"
        )
