Project 39 : Divide and Conquer : Local Gaussian Processes to design Covalent Organic Frameworks for Methane Deliverable Capacity
----------------

In this project, we explore the use of local 
Gaussian Process models to accelerate materials 
discovery when the search spaces are very large. 
We evaluate the performance of the framework on a 
covalent organic framework (COF) dataset that consists 
of 69,840 2D and 3D COFs [1]. This dataset replicates 
some real-world scenarios wherein the search space 
to explore is very large. In this test, we used an
initial training dataset comprising 5\% of 
the total search space. These COF structures are 
designed for methane storage and our optimization
target here is the deliverable capacity 
(v STP/v) of the COF structure. We employ
gaussian process surrogates with zero prior mean
function and Matern kernel as the covariance
function.

Gaussian Process (GP) has been a popular choice of 
surrogate model in Bayesian Optimization due to its
flexibility and uncertainty quantification. However,
training a Gaussian Process involves several matrix
inversions, which can dramatically scale up the computational
cost as more data is obtained via Bayesian Optimization. 
Gaussian Process has a runtime complexity of $O(n^3)$, 
where n is the number of training samples. Given its poor scalability,
the application of GPs to high-dimensional problems 
with several thousand observations remains challenging. 
In this project, we aim to reduce the computational cost 
of GP-based Bayesian Optimization by breaking a global GP 
model into several local GP models. These local GP models 
will run in parallel, accelerating the optimization problem.
We have also designed a new acquisition function to aggregate 
predictions from local GPs and select the next points to explore
with a tunable parameter to adjust exploration vs. exploitation.
We hypothesize that our method will significantly accelerate 
the runtime of Bayesian Optimization, and enable us to explore 
more points in the COFs dataset which cannot 
be done with a standard global GP model.

![Image not found](schematic.pdf "Workflow of training local GP models")

References

[1] Mercado, R.; Fu, R.-S.; Yakutovich, A. V.; Talirz, L.; Haranczyk, M.; Smit, B. In Silico Design of 2D and 3D Covalent Organic Frameworks for Methane Storage Applications. Chem. Mater. 2018, 30 (15), 5069–5086. https://doi.org/10.1021/acs.chemmater.8b01425.


<hr>

User guide
----------------

1. Install the conda environment bo-hackathon from the bo-hackathon.yml file.
Type the following command in the terminal : conda env create -f bo-hackathon.yml

2. The inputs to the code must be provided in code_inputs.py

3. The jupyter notebook for running the code is located under src/BO.ipynb

4. The dataset for training the model is located under data/properties.csv

5. The results from training the model are located under bo_output

Authors
----------------

Contributors can be found [here](https://github.com/AC-BO-Hackathon/project-localGPs_for_COF/graphs/contributors).

<hr>

