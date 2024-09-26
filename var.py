import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import tensorflow as tf
from sklearn.model_selection import train_test_split

class VaR:
    def __init__(self, data: pd.DataFrame, price_col: str, alpha=0.05, window=125, period: int = None):
        # Initialize VaR calculation parameters
        self.alpha = alpha
        self.window = window
        self.period = period
        self.data = self._basic_features(data, price_col, window)

        self.ret_model_trained = False
        self.vol_model_trained = False
        self.init_date = None
        self.timeseries = None
        self.var_cols = []

    # basic features
    def _basic_features(self, data: pd.DataFrame, price_col: str, window: int):
        # Calculate log returns and rolling volatility
        data['return'] = np.log(data.loc[:, price_col] / data.loc[:, price_col].shift(1))
        data['vol'] = data['return'].rolling(self.window).std()
        data['loss'] = data['return'].rolling(self.period).sum().shift(-self.period)
        data['loss'] = data['loss'].apply(lambda x: abs(x) if x < 0 else 0)
        data['exp_mean'] = data['return'].rolling(self.period).mean().shift(-self.period)
        data['exp_vol'] = data['vol'].rolling(self.period).mean().shift(-self.period)
        data.dropna(inplace=True, axis=0, how='any')
        return data

    # calculate VaR
    def calculate_var(self, col_name: str, type: str):
        # Apply VaR calculation to rolling window of returns
        self.data[col_name] = self.data['return'].rolling(self.window).apply(lambda x: self._var(x,
                                                                                                 type=type))
        self.var_cols.append(col_name)
        print(f'{col_name} Successfully Calculated!')

    # calculate VaR
    def _var(self, returns, type: str):
        # Historical VaR Type
        if type == 'historical':
            if len(returns) < self.window:
                raise ValueError("Not enough data to calculate VaR")
            sorted_returns = returns[-self.window:].sort_values()
            VaR = sorted_returns.quantile(self.alpha)
            VaR = abs(VaR)  # Ensure VaR is negative

        # Parametric VaR Type
        elif type == 'parametric':
            mean_return = returns.mean()
            std_return = returns.std()
            VaR = mean_return - std_return * norm.ppf(self.alpha)
            VaR = abs(VaR)  # Ensure VaR is negative
        
        # Scale previously calculated VaR to different period if needed using exponential method
        if self.period:
            t = self.period  # Number of trading days in the period
            VaR = np.sqrt(t) * VaR
            VaR =  VaR  # Ensure VaR is negative
            
        return VaR
    
    # calculate VaR
    def calculate_monte_carlo_var(self, col_name: str, num_simulations: int):
        self.data[col_name] = self.data['return'].rolling(self.window).apply(lambda x: self._monte_carlo_var(x,
                                                                                                             num_simulations))
        self.var_cols.append(col_name)
        print(f'{col_name} Successfully Calculated!')

    # calculate VaR
    def _monte_carlo_var(self, returns, num_simulations: int):
        if len(returns) < self.window:
            raise ValueError("Not enough data to calculate VaR")
    
        # Fit a normal distribution to the historical returns
        mean_return = returns.mean()
        std_return = returns.std()
        
        # Generate random returns using Monte Carlo simulation
        if self.period:
            simulated_returns = np.random.normal(mean_return, std_return, (num_simulations, self.period))
            cumulative_returns = np.cumprod(1 + simulated_returns, axis=1)[:, -1] - 1
        
            # Calculate cumulative VaR
            cumulative_VaR = np.percentile(cumulative_returns, self.alpha * 100)  # Convert alpha to percentile
        
            # Rescale cumulative VaR to 1-day VaR considering compounding effect
            VaR = abs(cumulative_VaR)
        else:
            simulated_returns = np.random.normal(mean_return, std_return, (num_simulations, 1))
            VaR = np.percentile(simulated_returns, self.alpha * 100)
        
        return VaR
    
    # calculate moving average
    def calculate_moving_average(self, input_col: str, output_col: str, window: int):
        self.data[output_col] = self.data[input_col].rolling(window).mean()
        print(f'{output_col} Successfully Calculated!')

    # create train data
    def create_train_data(self, type: str, seq_length: int, feature_cols, target_cols, size = 3000):
        # clean data
        self.data.dropna(inplace=True, axis=0, how='any')
        # create timeseries
        self.timeseries = np.array(sorted(self.data.index))
        # Initialize lists for features and targets
        features, targets = [], []
        
        # Extract feature and target columns
        dataframe = self.data.iloc[:size]
        feature_est = dataframe.loc[:, feature_cols]
        target_est = dataframe.loc[:, target_cols]

        # Create sequences of features and corresponding targets
        for i in range(len(feature_est) - seq_length):
            features.append(feature_est.iloc[i:i+seq_length].values)
            targets.append(target_est.iloc[i+seq_length - 1])
        
        # Convert to numpy arrays
        features, targets = np.array(features), np.array(targets)
        
        # Check date consistency
        end_date = dataframe.index[-1]
        if self.init_date is None:
            self.init_date = end_date
        else:
            assert self.init_date == end_date, 'Date mismatch'
        
        # Store features and targets based on type
        if type == 'return':
            self.ret_features = feature_cols
            self.ret_targets = target_cols
            self.ret_train_data = (features, targets)
            self.ret_input_shape = (seq_length, len(feature_cols))
            # notice
            print(f'{type} Train Data Successfully Created!')
        elif type == 'vol':
            self.vol_features = feature_cols
            self.vol_targets = target_cols
            self.vol_train_data = (features, targets)
            self.vol_input_shape = (seq_length, len(feature_cols))
            # notice
            print(f'{type} Train Data Successfully Created!')
        else:
            raise ValueError("Invalid type")
    
        # create LSTM dataset
        self._create_LSTM_dict(type = type, feature_cols = feature_cols, target_cols = target_cols,
                                seq_length = seq_length)

    # Create LSTM training dataset
    def _create_LSTM_dict(self, type, feature_cols, target_cols, seq_length):
        # extract data
        LSTM_data = {}
        feature_est = self.data.loc[:, feature_cols]
        target_est = self.data.loc[:, target_cols] 
        # create LSTM dataset
        for i in range(len(self.timeseries) - seq_length):
            timestamp = self.timeseries[i + seq_length - 1]
            feature_array = feature_est.iloc[i:i+seq_length,].values
            target_array = target_est.iloc[i+seq_length - 1]
            LSTM_data[timestamp] = (feature_array, target_array)
        # store LSTM data
        if type == 'return':
            self.ret_LSTM_data = LSTM_data
            # notice
            print(f'{type} Backtest Data Successfully Created!')
        elif type == 'vol':
            self.vol_LSTM_data = LSTM_data
            # notice
            print(f'{type} Backtest Data Successfully Created!')
        else:
            raise ValueError("Invalid type")
        

    def create_LSTM_model(self, type:str, lstm_units, output_units, learning_rate):
        # Validate input type and set input shape
        if type == 'return':
            assert self.ret_features and self.ret_targets is not None, 'No return features found'
            input_shape = self.ret_input_shape
        elif type == 'vol':
            assert self.vol_features and self.vol_targets is not None, 'No vol features found'
            input_shape = self.vol_input_shape
        else:
            raise ValueError("Invalid type")
        
        # Define model architecture
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
        
        # Store model based on type
        if type == 'return':
            self.ret_model = model
            # notice
            print(f'{type} LSTM Model Successfully Created!') 
        elif type == 'vol':
            self.vol_model = model
            # notice
            print(f'{type} LSTM Model Successfully Created!') 
        else:
            raise ValueError("Invalid type")
    
    def train_LSTM_model(self, type:str, epochs:int, validation_ratio:float = None, **kwargs):
        # Validate input type and set input shape
        if type == 'return':
            train_feature, train_target = self.ret_train_data
            if validation_ratio:
                train_feature, val_feature, train_target, val_target = train_test_split(train_feature, train_target,
                                                                                        test_size=validation_ratio, random_state = 42)
                self.ret_model.fit(train_feature, train_target, validation_data = (val_feature, val_target),
                                    epochs = epochs, **kwargs)
                self.ret_model_trained = True
            else:
                self.ret_model.fit(train_feature, train_target, epochs = epochs, **kwargs)
                self.ret_model_trained = True

        elif type == 'vol':
            train_feature, train_target = self.vol_train_data
            if validation_ratio:
                train_feature, val_feature, train_target, val_target = train_test_split(train_feature, train_target,
                                                                                        test_size=validation_ratio, random_state = 42)
                self.vol_model.fit(train_feature, train_target, validation_data = (val_feature, val_target),
                                    epochs = epochs, **kwargs)
                self.vol_model_trained = True
            else:
                self.vol_model.fit(train_feature, train_target, epochs = epochs, **kwargs)
                self.vol_model_trained = True

    # Calculate VaR using LSTM models
    def LSTM_predict(self, output_col = 'LSTM_VaR', num_simulations = 10000, hist_ret_col = 'MA_return_60'):
        assert self.ret_model_trained and self.vol_model_trained, 'LSTM model not trained'

        # Find the position of the initial date and add 1
        init_loc = np.where(self.timeseries == self.init_date)[0][0] + 1

        # Create a new column in the new_dataframe and initialize with NaN
        self.data.loc[:,output_col] = np.nan
        self.data.loc[:,output_col+'_HistMean'] = np.nan


        # Iterate through the time series starting from the initial position
        for i in range(init_loc, len(self.timeseries) - self.period):
            timestamp = self.timeseries[i]

            # Prepare input features for LSTM models
            ret_feature = np.expand_dims(self.ret_LSTM_data[timestamp][0], axis=0)
            vol_feature = np.expand_dims(self.vol_LSTM_data[timestamp][0], axis=0)

            # Use models to predict return and volatility
            hist_mean = self.data.loc[timestamp, hist_ret_col]
            ret_pred = self.ret_model.predict(ret_feature)[0][0]
            vol_pred = self.vol_model.predict(vol_feature)[0][0]

            # Simulate future returns
            LSTM_simulated_returns = np.random.normal(ret_pred, vol_pred,
                                                    (num_simulations, self.period))
            Hist_simulated_returns = np.random.normal(hist_mean, vol_pred,
                                                    (num_simulations, self.period))

            # Simulate cumulative returns
            LSTM_cumulative_returns = np.cumprod(1 + LSTM_simulated_returns, axis=1)[:, -1] - 1
            Hist_cumulative_returns = np.cumprod(1 + Hist_simulated_returns, axis=1)[:, -1] - 1

            # Calculate VaR
            LSTM_VaR = abs(np.percentile(LSTM_cumulative_returns, self.alpha * 100))
            Hist_VaR = abs(np.percentile(Hist_cumulative_returns, self.alpha * 100))
                
            # Store the calculated VaR in the new_dataframe
            self.data.loc[timestamp, output_col] = LSTM_VaR
            self.data.loc[timestamp, output_col+'_HistMean'] = Hist_VaR

        # append var cols
        self.var_cols.append(output_col)
        self.var_cols.append(output_col+'_HistMean')

            # clean data
        self.data.dropna(inplace=True, axis=0, how='any')    
        print(f'LSTM VaR Successfully Calculated!')

    def plot_var(self, figsize: tuple = (24, 16)):
        self._calculate_metrics()
        # Define colors for each VaR measure
        colors = {
            'HistVaR': 'darkorange',  
            'ParamVaR': 'burlywood', 
            'MonteCarloVaR': 'firebrick',    
            'LSTM_VaR': 'deepskyblue', 
            'LSTM_VaR_HistMean': 'cornflowerblue',
        }

        # Calculate differences between VaR and actual loss
        for var_col in self.var_cols:
            self.data[f'diff_{var_col}'] = self.data[var_col] - self.data['loss']

        # Plotting
        plt.figure(figsize=figsize)

        # Plot actual realized loss
        plt.subplot(2, 1, 1)
        plt.plot(self.data.index, self.data['loss'], label='Actual Realized Loss', color='lightsteelblue', linewidth=2)

        # Plot different VaR measures with breach statistics in the legend
        for var_col in self.var_cols:
            if var_col not in colors:
                print(f"Warning: {var_col} not in colors dictionary. Skipping this column.")
                continue
            breach_info = self.breach_stats[var_col]
            label = f"{var_col} (Breaches: {breach_info['breach_count']}, " \
                    f"Ratio: {breach_info['breach_percentage']:.2f}%, " \
                    f"Gap: {self.gap_stats[var_col]:.2f} days)"
            plt.plot(self.data.index, self.data[var_col], label=label, linestyle='solid', color=colors[var_col])

        # Add title and labels
        plt.title('Comparison of Actual Realized Loss and Estimated VaR Measures', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Value', fontsize=14)
        plt.legend(loc='best', fontsize='small')  # Locate the legend inside the graph
        plt.grid(True, linestyle='--', linewidth=0.5)

        # Plot the difference between VaR and actual loss
        plt.subplot(2, 1, 2)
        for var_col in self.var_cols:
            if var_col not in colors:
                print(f"Warning: {var_col} not in colors dictionary. Skipping this column.")
                continue
            no_breach_acc = self.convergence_stats[var_col]
            breach_acc = self.exceedance[var_col]
            label = f"{var_col} (No Breach Acc: {no_breach_acc:.4f}, Breach Acc: {breach_acc:.4f})"
            plt.plot(self.data.index, self.data[f'diff_{var_col}'], label=label, linestyle='-', color=colors[var_col], linewidth=1)

        # Add title and labels
        plt.title('Difference Between VaR and Actual Loss', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Difference (VaR - Loss)', fontsize=14)
        plt.axhline(0, color='black', linewidth=1, linestyle='--')  # Add a horizontal line at y=0
        plt.legend(loc='best', fontsize='small')  # Locate the legend inside the graph
        plt.grid(True, linestyle='--', linewidth=0.5)

        # Set font to serif
        plt.rcParams['font.family'] = 'serif'

        # Show plot
        plt.show()

    
    def _calculate_metrics(self):
        # initalize breach stats
        self.breach_stats = {}
        # breaches
        total_data_points = len(self.data)
        for var_col in self.var_cols:
            breaches = self.data[self.data['loss'] > self.data[var_col]]
            breach_count = len(breaches)
            breach_percentage = (breach_count / total_data_points) * 100
            self.breach_stats[var_col] = {
                'breach_count': breach_count,
                'breach_percentage': breach_percentage
            }

        # initialize gap stats
        self.gap_stats = {}
        # gaps
        for var_col in self.var_cols:
            breaches = self.data[self.data['loss'] > self.data[var_col]].index
            if len(breaches) > 1:
                        gaps = breaches.to_series().diff().dropna().dt.days
                        average_gap = gaps.mean()
            else:
                average_gap = None  # Not enough breaches to calculate gap
            # store gap stats
            self.gap_stats[var_col] = average_gap

        # initialize accuracy stats
        self.convergence_stats = {}
        # accuracy
        for var_col in self.var_cols:
            no_breach = self.data[self.data['loss'] <= self.data[var_col]]
            if not no_breach.empty:
                convergence = (no_breach[var_col] - no_breach['loss']).mean()
            else:
                convergence = None  # No data points for no breach
            # store accuracy stats
            self.convergence_stats[var_col] = convergence

        # initialize accuracy stats
        self.exceedance = {}
        # accuracy
        for var_col in self.var_cols:
            breaches = self.data[self.data['loss'] > self.data[var_col]]
            if not breaches.empty:
                exceedance = (breaches['loss'] - breaches[var_col]).mean()
            else:
                exceedance = None  # No data points for no breach
            # store accuracy stats
            self.exceedance[var_col] = exceedance


        