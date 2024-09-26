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
git clone https://github.com/Yuhanccc/ForexVaR-with-LSTM.git
cd ForexVaR-with-LSTM
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

## Example Result

1. **Data**: The Example used GBP/USD forex data on daily basis, source data can be found in `.data/GBPUSD1440.csv`.
     -  Data from 2009 to 2018 is used to train LSTM model on return prediction and volatility prediction.
     -  Data from 2019 to Sep 2024 is used to conduct out-of-sample test on LSTM model.
       
2. **Traditional Method**: Historical VaR, Parametric VaR and Monte Carlo Simulated VaR are introduced for setting a benchmark.
     -  Parametes (i.e. the mean and volatility of forex returns) are estimated on past 125 trading days (assuming 250 trading days/year).
     -  VaR estimated is 5 days forward VaR (i.e. estiamte the maximum possible loss on next 5 days),
     -  Estimated VaR is compared with actual loss.
     -  Monte Carlo Simulated VaR perform simulation for 10000 times.
     -  VaR at 5% (or 95% confidence level the other way around) is estimated for each method.
       
3. **LSTM Based VaR**: Two models are separately trained to make predicitons on average return & volatility in next 5 trading days.
     -  The pre-trained model is then used to make prediciton iteratively.
     -  The sequence length for training & prediciton in the LSTM model is 60, i.e, we use data of previous 60 trading days to make prediction.
     -  Predicted return and volatility is then used to generate 10000 monte carlo simulations of return, and the least 5% return is taken asw estimated VaR.
     -  Due to the poor prediction power of model on future returns (the case is the same for all assets), we introduced LSTM_VaR_HistMean, where the return for Monte Carlo Simulation is the historical mean instead of prediction by LSTM model.

![output](https://github.com/user-attachments/assets/43609a44-776a-464e-aec8-f4e2aea5491a)

 1. **Number of Breaches** : Number of times when actual loss exceeds estimated VaR at 5% level.
     * LSTM_VaR outperform all other methods as it achieved a breach time of 71 (20 times less than Historical VaR) and a breach ratio of 4.05% out of 1750 trading days.
     * LSTM_VaR_HistMean also outperforms tradingtion methods but underperformed pure LSTM_VaR, which suggested some prediciton power in our trained LSTM model for return.
  
 2. **Gap of Breaches**: The distance between two breach events. The longer the better.
     * LSTM type of methods achieved similar gaps of around 25 days, while traditional methods are around 22 days gap.
     * Significant outperformed.
     
 3. **No Breach Accuracy**: This method measures the distance between actual loss and estimated VaR. We should expect lower distance (smaller value of accuracy) here as it suggests how close the estimated VaR is to acutual loss.
     * Historical VaR achieved minimal value of 0.0147 on the test window. That suggests that Historical VaR (5% level) is 1.47% higher than actual loss on average.
     * Rest of methods are all around 0.0156, that is 1.56% higher than acutal loss on average, about 0.09% higher than Historical VaR.
     
 4. **Breach Accuracy**: Measures the distance between acutal loss and estimated VaR when there is a breach. The smaller the better.
     * Monte Carlo Simulated VaR demonstrated the best performance of 0.0102 (1.02%), but the rest (except for LSTM_VaR) are all just slightly higher these level.
     * LSTM_VaR is performed worst (0.0123 or 1.23%)

### **Comments**
In all but 1 metric (Breach Accuracy) outperforms or as least is as good as traiditonal methods.
Given its superior perofermance on breach times (1% lower than traditional methods), we should say that there is a superior power of utilizing deep learning to enhance the perforamnce of VaR estimation.


## Contributing
w
Contributions are welcome! Please fork the repository and submit a pull request with your changes. Ensure that your code adheres to the existing style and includes appropriate tests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

