# ML Lorenz Attractor Dashboard

A machine learning project exploring chaotic systems using the famous Lorenz Attractor. This project implements an interactive Streamlit dashboard that simulates the Lorenz system and trains multiple ML models to predict chaotic behavior.

## 📋 Project Overview

The **Lorenz Attractor** is a classic three-dimensional chaotic system that exhibits sensitive dependence on initial conditions. Despite being deterministic, its trajectories are impossible to predict long-term due to the exponential divergence of nearby initial conditions.

This project:
- Generates Lorenz Attractor trajectories with configurable parameters
- Trains and compares multiple machine learning models to predict system behavior
- Provides interactive 3D and 2D visualizations
- Analyzes model performance with metrics like MSE and R² score

## 🎯 Features

- **Interactive Simulation**: Adjust initial conditions and simulation parameters via sidebar controls
- **Multiple ML Models**: 
  - Linear Regression
  - Random Forest Regressor
  - Decision Tree Regressor
  - KNN Regressor
- **3D Visualization**: Compare actual vs predicted trajectories in 3D space
- **2D Projections**: Analyze system behavior across different 2D planes (x-y, x-z, y-z)
- **Model Comparison**: View performance metrics for selected models
- **Data Analysis**: Comprehensive Jupyter notebook with exploratory analysis

## 📁 Project Structure

```
.
├── main.py                                        # Streamlit dashboard application
├── Chaotic_System_using_Lorenz_Attractor.ipynb   # Jupyter notebook with analysis
├── lorenz_data_with_missing_and_region.csv       # Dataset with spatial and temporal data
└── README.md                                      # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- pip or conda package manager

### Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd ML_project
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

Or install packages individually:
```bash
pip install streamlit numpy pandas scikit-learn plotly
```

## 💻 Usage

### Run the Dashboard

Start the interactive Streamlit dashboard:
```bash
streamlit run main.py
```

The dashboard will open in your browser at `http://localhost:8501`

### Explore the Notebook

Open and run the Jupyter notebook for detailed analysis:
```bash
jupyter notebook Chaotic_System_using_Lorenz_Attractor.ipynb
```

## ⚙️ Lorenz System Parameters

The classic Lorenz system equations are:
- $\frac{dx}{dt} = \sigma(y - x)$
- $\frac{dy}{dt} = x(\rho - z) - y$
- $\frac{dz}{dt} = xy - \beta z$

Default parameters:
- **σ (sigma)**: 10 (controls horizontal convection)
- **ρ (rho)**: 28 (Rayleigh number)
- **β (beta)**: 8/3 (geometric factor)
- **dt**: 0.01 (time step for numerical integration)

## 📊 Dashboard Controls

### Simulation Settings
- **Initial x, y, z**: Set starting position (-10 to 10)
- **Simulation steps**: Number of time steps to simulate (100-2000)
- **Future prediction steps**: Steps ahead for predictions (0-500)

### Model Selection
- Multi-select interface to compare any combination of the 4 ML models
- Real-time metric updates for selected models

## 📈 Model Performance Metrics

Each model displays:
- **Mean Squared Error (MSE)**: Average squared prediction error
- **R² Score**: Coefficient of determination (1.0 = perfect fit)

## 🔍 Data

The `lorenz_data_with_missing_and_region.csv` file contains:
- Lorenz attractor trajectory data
- Regional classification information
- Missing value handling for robust analysis

## 📚 References

- [Lorenz, E. N. (1963)](https://journals.ametsoc.org/view/journals/atsc/20/2/1520-0469_1963_020_0130_dnf_2_0_co_2.xml) - "Deterministic Nonperiodic Flow"
- Chaos Theory fundamentals
- Machine Learning for dynamical systems prediction

## ⚡ Key Insights

1. **Chaotic Sensitivity**: Small changes in initial conditions lead to dramatically different trajectories
2. **ML Limitations**: Even sophisticated ML models struggle with long-term predictions in chaotic systems
3. **Local vs Global**: Models perform better for short-term predictions within the attractor region
4. **Model Comparison**: Different architectures capture different aspects of the system behavior

## 🤝 Contributing

Feel free to extend this project by:
- Adding more sophisticated neural network models
- Implementing ensemble methods
- Analyzing bifurcation diagrams
- Exploring other chaotic systems (Rössler, Hénon, etc.)

## 📝 License

This project is open source and available for educational and research purposes.

## 💡 Tips

- Start with fewer simulation steps (100-300) to see clear attractor patterns
- Compare Linear Regression vs Random Forest to understand model complexity trade-offs
- Experiment with different initial conditions to observe the butterfly effect
- Use the 2D projections to identify the characteristic "wings" of the Lorenz attractor

---
## you can visit here
https://shalulorenzattractor.streamlit.app/



