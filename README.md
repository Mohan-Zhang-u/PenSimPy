![alt text](figures/logo_light.png "Logo Title Text 1")
# **PenSimPy**
Python version of [IndPenSim](http://www.industrialpenicillinsimulation.com/). Sequential batch control, Raman spectra simulation, PRBS (pseudo random binary signal),
Faults (Aeration rate fault, Vessel back pressure fault...) and Substrate prediction are enabled. 
A fresh web application based on PenSimPy and [Quartic.AI](https://quartic.ai/) can be found [here](http://quartic.ddns.net:8000/).

#### **How to execute**
```
pip install -r requirements.txt
python main.py
```
#### **Setup in main.py**
```
total_runs = 1
num_of_batches = 1
plot_res = True
save_res = True
using_Raman = True
* No PRBS and Faults are applied by default
```
#### **Results**
##### **Inputs**
<p align="center">
  <img src="https://github.com/Quarticai/PenSimPy/blob/master/figures/Figure_2.png" width="60%">
</p>

##### **Outputs**
<p align="center">
  <img src="https://github.com/Quarticai/PenSimPy/blob/master/figures/Figure_1.png" width="60%">
</p>

##### **Csv data generation**
```
/path/to/PenSimPy/data/0/batch_median_trend.csv
/path/to/PenSimPy/data/0/batch_statistics.csv
/path/to/PenSimPy/data/0/raman.csv
```
#### **Benchmark**
##### *Time cost with Raman Simulation (* MacBook Pro (15-inch, 2018); 2.2 GHz 6-Core Intel Core i7; 16 GB 2400 MHz DDR4)*
<p align="center">
  <img src="https://github.com/Quarticai/PenSimPy/blob/master/figures/benchmark.png" width="80%">
</p>

