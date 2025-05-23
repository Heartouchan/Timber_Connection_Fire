# Timber_Connection_Fire
This project aims to evaluate the fire performance of the timber connection with multi-fidelity neural network (MFNN).

The code programs for MFNN and low-fidelity model (LFM) are provided. 

Once the timber connection configuration is determined, the evaluation procedure can be conducted as below:

**Step 1:** Develop the high-fidelity model for the designed connection.

**Step 2:** Develop the low-fidelity connection model for the designed connection with package LFM.

**Step 3:** Integrate the displacements or fire resistance times from both high-fidelity models and low-fidelity models and develop MFNN with package MFNN.
 
**Step 4:** Conduct probabilistic fire engineering, i.e., reliability, sensitivity, and fragility analysis, for designed timber connection using developed MFNN and comprehensively evaluate its fire performance.


## Requirements

To run this code package, please install the following dependencies:

```bash
Python 3.9 or higher
Pytorch 2.6.0 or higher
tkinter
NumPy
Pandas
matplotlib
scikit-learn

