# RoMAn-Model-Predictive-Control
Robust model agnostic predictive control algorithm for randomly excited dynamical systems

This repository contains the python codes of the paper 
  > + Tripura, T., & Chakraborty, S. (2023). Robust model agnostic predictive control algorithm for randomly excited dynamical systems. Probabilistic Engineering Mechanics, 103517. [Paper](https://doi.org/10.1016/j.probengmech.2023.103517)

## Schematic architecture of the RoMAn-MPC
![RoMAn](images/RoMAn.png)

## The system discovery module
![Model_discovery](images/Identification_control.png)

## The Robust control module
![Control](images/SMPC.png)

# Files
  + `Deterministic_Lotka_Volterra.py` This code is for deterministic control of Lotka-Volterra system. Refer it as a headstart to MPC.
  + `Stochastic_control_Lotka_Volterra.py` This code performs the identification, and control of the Lotka-Volterrra system.
  + `Stochastic_control_Lorenz.py` This code performs the identification, and control of the Lorenz oscillator.
  + `Stochastic_control_SDOF_TMD.py` This code performs the identification, and control of an SDOF Tuned-mass-damper system.
  + `Stochastic_control_76dof.py` This code performs the identification, and control of an 76DOF slender system.
  + `fun_library.py` This code generates the library for sparse Bayesian inference.
  + `fun_optimize.py` This code contains the functions, which are required
    - to generates the discrete dynamics from the identified systems, 
    - to create the constraints, and,
    - to generate the objective function for the control.
  + `fun_spikeslab.py` This code performs the sparse Bayesian inference using the Spike and Slab prior and darws the samples from posterior using Gibbs sampler.
  + `fun_stochresp.py` This code generates the synthetic stochastic responses for the systems.
  + `fun_plots.py` This code contains the figure plot settings.

# BibTex
If you take help of our codes, please cite us at,
```
@article{tripura2023robust,
  title={Robust model agnostic predictive control algorithm for randomly excited dynamical systems},
  author={Tripura, Tapas and Chakraborty, Souvik},
  journal={Probabilistic Engineering Mechanics},
  pages={103517},
  year={2023},
  publisher={Elsevier}
}
```
