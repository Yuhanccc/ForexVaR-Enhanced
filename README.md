# Enhanced VaR Estimation using Deep Learning

This repository demonstrates an enhanced approach to Value at Risk (VaR) estimation by leveraging deep learning techniques, specifically Long Short-Term Memory (LSTM) networks. The project includes a comprehensive implementation of traditional VaR methods and integrates LSTM models to improve the accuracy and robustness of VaR predictions.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Value at Risk (VaR) is a widely used risk management tool that quantifies the potential loss in value of a portfolio over a defined period for a given confidence interval. Traditional methods of VaR estimation include Historical VaR, Parametric VaR, and Monte Carlo VaR. This project enhances these traditional methods by incorporating LSTM networks to predict future returns and volatilities, thereby providing a more accurate and robust VaR estimation.

## Features

- **Traditional VaR Methods**: Historical VaR, Parametric VaR, and Monte Carlo VaR.
- **Deep Learning Integration**: LSTM models for predicting future returns and volatilities.
- **Comprehensive Data Handling**: Automated feature engineering and data preprocessing.
- **Visualization**: Plotting actual losses against estimated VaR measures.
- **Metrics Calculation**: Breach statistics, gap analysis, and accuracy metrics.

## Installation

To get started, clone the repository and install the required dependencies:
```
bash
git clone https://github.com/Yuhanccc/ForexVaR-Enhanced.git
cd ForexVaR-Enhanced
pip install -r requirements.txt
```

## Usage

1. **Prepare Data**: Ensure your data is in the correct format (CSV with columns: open, high, low, close, volume).
2. **Initialize VaR Object**: Load your data and initialize the VaR object.
3. **Calculate VaR**: Use the provided methods to calculate different types of VaR.
4. **Train LSTM Models**: Prepare features and train LSTM models for return and volatility series.
5. **Predict and Plot**: Use the trained models to predict VaR and visualize the results.

# Examples

Check out the `SampleCode.ipynb` notebook for a detailed walkthrough of the entire process, from data loading to VaR calculation and visualization.

## Example Model
**Model Architecture**: The LSTM model used in this project is designed to predict future returns and volatilities, which are then used to estimate Value at Risk (VaR). The model architecture is as follows:
```python
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(shape=input_shape),
    tf.keras.layers.Normalization(axis=-1, invert=False),
    tf.keras.layers.LSTM(lstm_units, return_sequences=True),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.LSTM(lstm_units, return_sequences=True),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.LSTM(lstm_units, return_sequences=True),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.LSTM(lstm_units, return_sequences=True),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.LSTM(lstm_units, return_sequences=True),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.LSTM(lstm_units),
    tf.keras.layers.Dense(output_units)
])

# Compile model
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
metric_r2 = tf.keras.metrics.R2Score()
model.compile(optimizer=optimizer, loss='mse', metrics=[metric_r2])
```
1. **Input Layer**: The model starts with an input layer that takes in data with the shape specified by input_shape.
2. **Normalization Layer**: This layer normalizes the input data, which helps in stabilizing and speeding up the training process.
3. **LSTM Layers**: The model includes six LSTM (Long Short-Term Memory) layers, each with lstm_units units. LSTM layers are well-suited for time series data as they can capture temporal dependencies. The return_sequences=True parameter ensures that each LSTM layer outputs the full sequence of predictions, which is necessary for stacking multiple LSTM layers.
4. **Dropout Layers**: After each LSTM layer, a Dropout layer with a dropout rate of 0.1 is added. Dropout is a regularization technique that helps prevent overfitting by randomly setting a fraction of input units to zero during training.
4. **Dense Layer**: The final layer is a Dense layer with output_units units, which produces the final output of the model.
Compilation: The model is compiled using the Adam optimizer with a specified learning rate. The loss function used is Mean Squared Error (MSE), and the model's performance is evaluated using the R2 Score metric.


## Example Result

1. **Data**: Data: The example uses GBP/USD forex data on a daily basis. The source data can be found in .data/GBPUSD1440.csv.
     -  Data from 2009 to 2018 is used to train the LSTM model for return and volatility prediction.
     -  Data from 2019 to September 2024 is used to conduct out-of-sample tests on the LSTM VaR method.
       
2. **Traditional Method**: Historical VaR, Parametric VaR and Monte Carlo Simulated VaR are introduced for setting a benchmark.
     -  Parameters (i.e., the mean and volatility of forex returns) are estimated using the past 125 trading days (assuming 250 trading days per year).
     -  VaR is estimated as a 5-day forward VaR (i.e., estimating the maximum possible loss over the next 5 days).
     -  The estimated VaR is compared with the actual loss.
     -  Monte Carlo Simulated VaR performs 10,000 simulations.
     -  VaR at the 5% level (or 95% confidence level) is estimated for each method.
       
3. **LSTM Based VaR**: Two models are separately trained to make predictions on the average return and volatility over the next 5 trading days.
     -  The pre-trained model is then used to make predictions iteratively.
     -  The sequence length for training and prediction in the LSTM model is 60, meaning we use data from the previous 60 trading days to make predictions.
     -  Predicted return and volatility are then used to generate 10,000 Monte Carlo simulations of returns, and the lowest 5% of returns are taken as the estimated VaR.
     -  Due to the poor predictive power of the model on future returns (a common issue for all assets), we introduced LSTM_VaR_HistMean, where the return for Monte Carlo Simulation is the historical mean instead of the prediction by the LSTM model.

![output](https://github.com/user-attachments/assets/43609a44-776a-464e-aec8-f4e2aea5491a)

 1. **Number of Breaches** : The number of times the actual loss exceeds the estimated VaR at the 5% level.
     * LSTM_VaR outperforms all other methods, achieving a breach count of 71 (20 times fewer than Historical VaR) and a breach ratio of 4.05% out of 1750 trading days.
     * LSTM_VaR_HistMean also outperforms traditional methods but underperforms compared to pure LSTM_VaR, suggesting some predictive power in our trained LSTM model for returns.
  
 2. **Gap of Breaches**: The distance between two breach events. A longer gap is better.
     * LSTM-based methods achieved similar gaps of around 25 days, while traditional methods have gaps of around 22 days.
     * This indicates significant outperformance by LSTM-based methods.
     
 3. **No Breach Accuracy**: This measures the distance between the actual loss and the estimated VaR when there is no breach. A lower value indicates better accuracy, as it suggests the estimated VaR is closer to the actual loss.
     * Historical VaR achieved a minimal value of 0.0147 in the test window, indicating that Historical VaR (at the 5% level) is 1.47% higher than the actual loss on average.
     * he rest of the methods are around 0.0156, indicating they are 1.56% higher than the actual loss on average, about 0.09% higher than Historical VaR.
     
 4. **Breach Accuracy**: This measures the distance between the actual loss and the estimated VaR when there is a breach. A smaller value is better.
     * Monte Carlo Simulated VaR demonstrated the best performance with a value of 0.0102 (1.02%), while the rest (except for LSTM_VaR) are only slightly higher.
     * LSTM_VaR performed the worst with a value of 0.0123 (1.23%).

### **Comments**
In all but one metric (Breach Accuracy), LSTM-based methods outperform or are at least as good as traditional methods. Given their superior performance in terms of breach times (1% lower than traditional methods), we can conclude that utilizing deep learning significantly enhances the performance of VaR estimation.


## Contributing
w
Contributions are welcome! Please fork the repository and submit a pull request with your changes. Ensure that your code adheres to the existing style and includes appropriate tests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

