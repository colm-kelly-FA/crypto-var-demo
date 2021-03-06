print("I am being executed!")

import numpy as np 
import pandas as pd
from scipy import interpolate
import copy
from statsmodels.tsa.statespace.sarimax import SARIMAX
from arch import arch_model
from scipy.stats import norm






def weighted_hist_VaR_2(coin, alpha, history, lambda_,exponential_weighted=True, mass_centered=True):

    coin = pd.DataFrame(coin)
    coin=coin.sort_values(by=["Date"], ascending = False)
    i=0
    VaR_column = np.repeat(float("NaN"), history)

    while coin.iloc[-(history+i+1):-(1+i),:].shape[0] == history:
        
        port = coin.iloc[-(history+i+1):-(1+i),:]
        
        # Number of observations for a crypto
        k = port.shape[0]

        # Compute the equally weighted and exponentially weighted probabilites
        port.loc[port.index,"equal_prob"] = 1/k
        # Compute Receny
        port["Recency"] = port[["Return"]].apply(lambda x: np.arange(len(x)), axis=0)
        # Weighted probabilites are done so with respect to how 'recent' and observation is, more recent is assigned more weight
        port.loc[port.index,"weighted_prob"] = (1-lambda_)*(pow(lambda_,port.loc[port.index,"Recency"]-1)/(1-pow(lambda_,k)))
       
        # Rank the returns from lowest to highest, the key step in Historical VaR estimation
        port = port.sort_values(by="Return")
       
        # Compute the equally weighted and exponentially weighted cumulative probabilites
        port["cumul_equal_prob"] = port['equal_prob'].cumsum()
        port['cumul_weighted_prob'] = port['weighted_prob'].cumsum()

        # Capture the cumulative probabilites and corresponding crypto returns
        returns = np.array(port["Return"])

        ## EXP-WEIGHTED VAR
        if exponential_weighted:

            cumul_weighted_prob = np.array(port["cumul_weighted_prob"])

            ## MASS CENTERED VAR 
            if mass_centered:
                # Construct and array to the corresponding midpoints
                cumul_weighted_prob_mid = (cumul_weighted_prob[:-1] + cumul_weighted_prob[1:])/2
                returns_mid = (returns[:-1] + returns[1:])/2

                # construct array to hold cumuulative probabilities, their returnsa and resepctive midpoints
                cumul_weighted_prob_mass_center = [None]*(len(cumul_weighted_prob)+len(cumul_weighted_prob_mid))
                returns_mass_center = [None]*(len(returns)+len(returns_mid))

                # Concatenate the midpoint to their respective arrays also adding the first value of 0% and -Inf
                cumul_weighted_prob_mass_center[::2] = cumul_weighted_prob
                cumul_weighted_prob_mass_center[1::2] = cumul_weighted_prob_mid
                cumul_weighted_prob_mass_center = [0] + cumul_weighted_prob_mass_center
                returns_mass_center[::2] = returns
                returns_mass_center[1::2] = returns_mid
                returns_mass_center = [float("Inf")] + returns_mass_center

                # Interpolation to find the alpha% correpsoning Return
                x = cumul_weighted_prob_mass_center
                y = returns_mass_center
                mass_center_exponential_VaR_interpolator = interpolate.interp1d(x, y)

                # Interpolationg for alpha
                VaR = mass_center_exponential_VaR_interpolator(alpha)
                
                VaR_column = np.append(VaR, VaR_column)
                i = i+1

            # STANDARD Weighted VAR
            else:
                # Interpolation to find the alpha% correpsoning Return
                x = cumul_weighted_prob
                y = returns
                exponential_VaR_interpolator = interpolate.interp1d(x, y)

                # Interpolationg for alpha
                VaR = exponential_VaR_interpolator(alpha)
                
                VaR_column = np.append(VaR, VaR_column)
                i = i+1
   

        ## EQUALLY WEIGHTED VAR 
        else:
            cumul_equal_prob = np.array(port["cumul_equal_prob"])

            # Interpolation to find the alpha% correpsoning Return
            x = cumul_equal_prob
            y = returns
            VaR_interpolator = interpolate.interp1d(x, y)

            # Interpolationg for alpha
            VaR = VaR_interpolator(alpha)
        
            VaR_column = np.append(VaR, VaR_column)
            i = i+1
    coin_with_var=copy.copy(coin)
    ## EXP-WEIGHTED VAR
    if exponential_weighted:

        ## MASS CENTERED VAR 
        if mass_centered:


            coin_with_var["ExpW_mc VaR"] = VaR_column

        # STANDARD Weighted VAR
        else:

            coin_with_var["ExpW VaR"] = VaR_column

    ## EQUALLY WEIGHTED VAR 
    else:
        
        coin_with_var["EqW VaR"] = VaR_column
    
    return coin_with_var


def garch_VaR(coin, alpha, history, ar_order, garch_order=(1,1), horizon=1, dist="NORM"):
    
    if coin.index.names[0] == "Symbol":
        coin_data = coin.droplevel("Symbol").loc[:,["Return"]].dropna()
    else:
        coin_data = coin.loc[:,["Return"]].dropna()
        
    coin_data = coin_data.sort_values(by=["Date"], ascending = True)
    date_index = coin_data.index.get_level_values("Date")
    
    nforecasts = horizon
    
    # data and window size
    n_obs = len(coin_data)
    window = history
    window_break_init = date_index[window-1]
    window_break = window_break_init
    
    # ar_order = (2, 0, 1)
    # garch_p = 2
    # garch_q = 2
    
    # ARIMA training and in-sample residuals
    ar_mod = SARIMAX(coin_data[:window_break], order = ar_order)
    ar_mod_results = ar_mod.fit()
    coin_data["yhat"] = ar_mod_results.fittedvalues
    coin_data["ar_resid"] = ar_mod_results.resid
    
    # first out of sample AR forecast and residual
    coin_data.loc[date_index[window],"yhat"] = ar_mod_results.forecast(steps=nforecasts).values
    coin_data.loc[date_index[window],"ar_resid"] = coin_data.loc[date_index[window],"Return"]-coin_data.loc[date_index[window],"yhat"]

    # Garch Model
    resid_model = arch_model(coin_data.loc[:window_break,"ar_resid"], p = garch_order[0], q = garch_order[1], vol = "GARCH")
    resid_model_results = resid_model.fit(last_obs=window_break, disp="off")

    coin_data.loc[:, "GARCH vol"] = resid_model_results.forecast(start=window_break, horizon=nforecasts).residual_variance
    
    t=0

    while window_break < date_index[len(date_index)-2]:
        
        # increment
        t = t+1
        window_break = date_index[window-1 + t]

        # re-train garch model on expanded window 
        resid_model = arch_model(coin_data.loc[:window_break,"ar_resid"], p = garch_order[0], q = garch_order[1], vol = "GARCH")
        resid_model_results = resid_model.fit(last_obs=window_break, disp="off")

        # store garch variance
        coin_data.loc[window_break, "GARCH vol"] = resid_model_results.forecast(start=window_break, horizon=nforecasts).residual_variance.values[-1]

        # re-train AR model on expanding window
        ar_mod = SARIMAX(coin_data.loc[:window_break, ["Return"]], order = ar_order)
        ar_mod_results = ar_mod.fit()

        coin_data.loc[date_index[window + t], "yhat"] = ar_mod_results.forecast(1).values
        coin_data.loc[date_index[window + t], "ar_resid"]  = coin_data.loc[date_index[window + t],"Return"]-coin_data.loc[date_index[window + t],"yhat"]

    coin_data["GARCH vol"] = coin_data["GARCH vol"].shift(1)
    
    distribution_quantile = norm.ppf(alpha)
    coin_data["GARCH VaR"] = -coin_data["yhat"]+distribution_quantile*np.sqrt(coin_data["GARCH vol"])

    return coin_data
