# housing-prices-ml-regression

A PyTorch model for predicting California housing prices, based on geographic and demographic features. The model achieves a a Mean Absolute Percentage Error (MAPE) of 28.4% on test data.

[Download the model here](housing_price_regression_model.pth)

[Kaggle Dataset link](https://www.kaggle.com/datasets/camnugent/california-housing-prices/)

## Overview

This project uses PyTorch to build a neural network regression model for housing price prediction. The model analyzes several geographic and demographic variables to predict median house values in California districts, based on the 1990 California Census.

## Dataset

The dataset uses 9 predictor variables:

1. longitude (float)
2. latitude (float)
3. housing_median_age (float)
4. total_rooms (float)
5. total_bedrooms (float)
6. population (float)
7. households (float)
8. median_income (float)
9. ocean_proximity (int)

To predict:

1. median_house_value (float)

## Requirements

- Python 3.8+
- PyTorch
- pandas
- scikit-learn
- matplotlib

## Usage

1. Clone the repository
2. Create a virtual environment `py -m venv .venv` and activate it `.venv/Scripts/activate`
3. Install dependencies: `pip install -r requirements.txt`
4. Run the model: `py main.py`

## Model Architecture

- Input layer: 9 features
- Hidden layer 1: 64 neurons with ReLU activation
- Hidden layer 2: 32 neurons with ReLU activation
- Output layer: 1 neuron (Regression)
- Optimization: Adam optimizer with learning rate 0.01
- Loss function: Mean Squared Error Loss

## Results

The model is trained for 1000 epochs and the training progress can be visualized through a loss plot that is automatically generated and saved as [loss_plot.png](loss_plot.png).
The model achieved:

- RMSE: $66,800
- MAPE: 28.4%

## License

[MIT License](LICENSE)
