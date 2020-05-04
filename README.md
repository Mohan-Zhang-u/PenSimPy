# PenSimPy
Python version of [IndPenSim](http://www.industrialpenicillinsimulation.com/)

#### How to execute
```
pip install -r requirements.txt
python main.py
```
#### Setup in main.py
```
total_runs = 1
num_of_batches = 1
plot_res = True
save_res = True
using_Raman = True
* No PRBS and Faults applied
```
#### Results
##### Inputs
<p align="center">
  <img src="https://github.com/Quarticai/PenSimPy/blob/master/figures/Figure_2.png" width="50%">
</p>

##### Outputs
<p align="center">
  <img src="https://github.com/Quarticai/PenSimPy/blob/master/figures/Figure_1.png" width="50%">
</p>

##### Csv data generation
```
/path/to/PenSimPy/data/0/batch_median_trend.csv
/path/to/PenSimPy/data/0/batch_statistics.csv
/path/to/PenSimPy/data/0/raman.csv
```
#### Benchmark
##### Time cost with Raman Simulation
<p align="center">
  <img src="https://github.com/Quarticai/PenSimPy/blob/master/figures/benchmark.png" width="50%">
</p>

