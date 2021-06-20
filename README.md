![alt text](figures/logo_light.png "Logo Title Text 1")

**Development Status:** As of 08/2020 PenSimPy is under active maintenance (expect bug fixes and updates). 
Please follow, star, and fork to get the latest functions.
# PenSimPy ![](https://img.shields.io/badge/python-3.8.0-orange) ![](https://img.shields.io/badge/c%2B%2B-11-blue)
PenSimPy is a Python version of [IndPenSim](http://www.industrialpenicillinsimulation.com/), which simulates the industrial-scale Penicillium chrysogenum fermentation. 
This simulator is based on `c++` solving the ODE functions so as to achieve a faster performance than the MATLAB version. 

PenSimPy aims for recording those key features such as `pH`, `temperature`, `vessel volume`, `vessel weight`, `penicillin concentration`, 
`dissolved oxygen concentration`, `sugar feed rate`, `soil bean feed rate`, `aeration rate`, `back pressure`, `water`, `acid`, 
`phenylacetic acid flow-rate`, `Oxygen in percent in off-gas`, and the Raman spectroscopy data throughout the whole process. A conventionally used 
Sequential Batch Control strategy with PID control is presented with example codes and all the recipe's feed can be manually updated. 

Furthermore, we incorporate Reinforcement Learning to improve the penicillin gain and present them as example codes.
A web application based on PenSimPy can be found [here](http://quartic.ddns.net:8000/).

Installation
============
PenSimPy relies on [fastodeint](https://github.com/Quarticai/fastodeint), a python bindings for an ODE solver implemented in C++. Unfortunely, it's not available as a PyPI package at this point, so you need to follow the installation instruction [here](https://github.com/Quarticai/fastodeint/blob/master/README.md) to build it yourself.
Once fastodeint is installed, you can install PenSimPy by the following command
```
pip install pensimpy
```
Examples
============
See the `examples` directory
- see [examples/batch_generation.py](pensimpy/examples/batch_generation.py) to generate
batch and Raman data with self-designed recipes
- see [examples/reinforcement_learning_boilerplate.py](pensimpy/examples/reinforcement_learning_boilerplate.py) to 
simulate penicillin yield process with reinforcement learning

