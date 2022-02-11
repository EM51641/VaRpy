To install, just use pip :

.. code:: python

   pip install pyvar

Required Dependencies are listed below , such :

============ ========
Dependency   Version
============ ========
arch         5.0.1
numpy        1.20.1
scipy        1.6.2
pandas       0.12.2
numba        0.52.1
joblib       1.0.1
scipy         0.4
tabulate     3.3.4
============ ========

There is no dependency verification , so please, make sure to have
installed every required one before using the package.

**Example**
===========

To begin, let’s extract default data:

.. code:: python

   import pyvar
   from pyvar import EVT_VaR,Student_VaR,Normal_VaR 
   from pyvar.Backtester.bktst import Backtest
   from pyvar.Backtester.time_Significance import Testing
   import matplotlib.pyplot as plt 

   data = d1()* 100
   data

**Let’s compute our weekly standard VaR and CVaR**

.. code:: python

   VaR,CVaR = Normal_VaR(data.values.reshape(-1,) ,0.05,7)
   print(VaR,CVaR)


**Let’s backtest our VaR, to evaluate its consistency throughout time**

In each iteration, we choose to use a window of 500 data to evaluate our tail statistic. Additionally, our VaR is evaluated on a weekly basis for an alpha of 5%.

.. code:: python

   VaR , CVaR = Backtest(data,500,7,0.05,model = 'Gaussian')
   ts = Testing(data,VaR,CVaR,500,0.05)
   print(ts.summary)

**Plot your VaR and CVaR**

.. code:: python


   import matplotlib.pyplot as plt 

   fig = plt.figure(figsize=(15,5))
   plt.plot(data[500:])
   plt.plot(VaR)
   plt.plot(CVaR)
   plt.show()

.. image:: https://raw.githubusercontent.com/EM51641/VaRpy/main/output/output.png
