import arch 
from arch import arch_model

def Arch_data(return_matrix , Horizon, distribution = 'gaussian' ):
    am     = arch_model(return_matrix , vol='Garch', p=1 , o=1 , q=1,mean='AR', lags=1,dist = distribution )
    res    = am.fit(update_freq=1,disp = 'off')
    forecasts = res.forecast(horizon = Horizon)
    forecasted_mean,forecasted_var,conditional_volatility = forecasts.mean.dropna().iloc[-1,-1] , forecasts.variance.dropna().iloc[-1,-1] , res.conditional_volatility
    return forecasted_mean , forecasted_var , conditional_volatility