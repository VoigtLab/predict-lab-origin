# predict-lab-origin

This is the software that accompanies the publication:
Alec AK Nielsen & Christopher A Voigt. Deep learning to predict the lab-of-origin of engineered DNA. Nature Communications. 2018.

Training was performed on an NVIDIA GRID K520 GPU using Amazon Web Services Elastic Cloud Compute (EC2) running Ubuntu 16.04.1 LTS. Due to the size of the encoded input data, it was necessary to house it on "ephemeral" storage ('/mnt' for our setup). 

The following packages are needed to run the code:
Python (version 3.5.2)
NumPy (version 1.13.0)
SciPy (version 0.19.0)
Tensorflow backend (version 1.1.0) 
Keras (version 2.0.4)
Pickle
json (version 2.0.9)
bayesian-optimization (https://github.com/fmfn/BayesianOptimization/)