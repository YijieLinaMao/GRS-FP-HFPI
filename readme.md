# An Efficient Beamforming Optimization Framework for Generalized Rate-Splitting with Imperfect CSIT

This is a code package related to the following paper:

Y. Wang, T. Fang and Y. Mao, "An Efficient Beamforming Optimization Framework for Generalized Rate-Splitting with Imperfect CSIT," in *IEEE Transactions on Communications*, doi: 10.1109/TCOMM.2025.3552748.

# Content of Code Package

Here is a detailed description of the package:

*   The code in all packages are implemented in Python environment.
*   This code is designed to reproduce the data presented in **Fig. 6** for the case where the number of users ( K = 3 ). Reproducing the entire figure may take a considerable amount of time; if needed, you can adjust the parameter settings to regenerate it as desired.By running `demo.py`, you can obtain the required results.

    The file `algorithm.py` provides a unified interface to call all six baselines. The `alg` folder contains implementations of the FP-HFPI algorithm and the WMMSE algorithm (using CVX) for each of the six baselines.

# Abstract of the Article

Rate-splitting multiple access (RSMA) emerges as a compelling physical-layer transmission paradigm for effectively managing interference in 6G networks. Within the realm of RSMA transmission frameworks, generalized rate-splitting (GRS) stands out as a versatile strategy that embraces existing multiple access (MA) schemes, including space division multiple access (SDMA), non-orthogonal multiple access (NOMA), and orthogonal multiple access (OMA) as specific instances. Despite its versatility, GRS encounters significant design challenges, particularly in dealing with the resource optimization complexities resulting from the exponential growth in the number of common streams with the number of users. To tackle the issue, in this work, we propose a novel and highly efficient beamforming optimization algorithm for GRS to maximize the ergodic sum rate (ESR) with imperfect channel state information at the transmitter (CSIT). Specifically, the stochastic ESR maximization problem is first transformed into a deterministic one using sampled average approximation (SAA). This transformed problem is further decomposed into a series of convex subproblems by the fraction programming (FP) approach. Based on the Karush-Kuhn-Tucker (KKT) conditions of each subproblem, we derive the optimal beamforming structure (OBS) of GRS. To determine the Lagrange dual variables within the OBS, we then propose a fixed point iteration (FPI)-based method. Through extensive numerical results, we show that the proposed algorithm significantly reduces the computational complexity without sacrificing ESR performance compared to conventional optimization algorithms. Thanks to the efficiency of our algorithm, we illustrate, for the first time, the performance of GRS with more than three users. We draw the conclusion that our proposed algorithm shows promise in advancing the practical application of RSMA in 6G.

# License and Referencing

This code package is licensed under the GPLv2 license. If you in any way use this code for research that results in publications, please cite our original article listed above.

# Acknowledgements

This work has been supported in part by the National Nature Science Foundation of China under Grant 62201347; and in part by Shanghai Sailing Program under Grant 22YF1428400.
