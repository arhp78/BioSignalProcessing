# BioSignalProcessing
## This repository includes all the practical assignments and the slides of the BioSignal Processing graduate course.


- [HW1](https://github.com/arhp78/BioSignalProcessing/tree/main/HW1_Prac): Basic concept of Signal Processing, Window filtering, Fourier Transform, Correlation, P300 extraction
- [HW2](https://github.com/arhp78/BioSignalProcessing/tree/main/HW2_Prac): Power Spectral Density Estimation with Periodogram, BT, and welch, Estimation of the type and degree of time series model: AR, MA, and ARMA.
- [HW3](https://github.com/arhp78/BioSignalProcessing/tree/main/HW3_Prac): Adaptive Filter, Cepstrom Analysis, Segmentation of PCG signal with classification, Segmentation of PCG signal with Hidden Markov Models


### This course contains the following topics:
- **BSP_01_2_M1: An introduction to deterministic signals and signal processing**:
  - Definition of signal and its types
  - The Purpose of signal processing
  - Definition of energy, power, internal multiplication and correlation for deterministic signals
  - Fourier transform of continuous and discrete signals
  - Parsol's theorem for continuous and discrete signals
  - Sampling
  - Discrete Fourier Transform(DFT)
  - Checking the windowing effect
  - Short-term Fourier transform
 
- **BSP_01_2_M2: Random process**:
  - Random variable/mathematical expectation
  - Binomial random variable/random vector
  - Estimation of a random variable without observation/estimation of a random variable with observation of another random variable
  - Properties of covariance matrix and correlation matrix
  - Definition of continuous stochastic process and description of its first and second order
  - Static definition
  - Process passage through an invariant linear system with definite time
  - Definition of ergodicity
  - Power spectrum density
  - Whitening process
  - Linear process
    
- **BSP_01_2_M3: BioSignal**:
  - Types of Bio signals in terms of origin
  - Types of Bio signals from the point of view of the producing organ:
      - Biosignals related to the brain
      - Biosignals related to the heart
      - Biosignals related to muscles
      - Biosignals related to the stomach
      - Biosignals related to the eye
      - Biosignals related to the respiratory system
      - Biosignals related to joints
  - A reference to the processing of brain signals
  - Reference to cardiac signal processing
 
- **BSP_01_2_M4: Estimation of statistical parameters of the process**:
  - Estimation of statistical parameters of a continuous process
    - Average estimate
    - Variance estimation
    - Correlation function estimation
  - Estimation of statistical parameters of a discrete process
    - Average estimate
    - Variance estimation
    - Correlation function estimation
  - Synchronous averaging
 
- **BSP_01_2_M5: Time Series and Parametric Models**:
   - Linear process
   - AR(p) model
      - Calculation of model parameters/estimation of model parameters
   - MA(q) model
      - Calculation of model parameters/estimation of model parameters
   - ARMA(p,q) model
   - Calculation of model parameters/estimation of model parameters
   - Estimation of the order of the model
   - Other models (linear/non-linear)
   - Signal segmentation

- **BSP_01_2_M6: Spectural Estimation**:
   - General estimation methods
   - Non-parametric methods
       - based on correlation estimation and its Fourier transform (BT (Tukey-Blackman) method)
       - based on the direct Fourier transformation of the sample function (Periodogram method) and its improvement (Welch)
   - Parametric methods
   - Methods based on ARMA, MA, AR models
   - Some special methods (Capon, PHD, Prony)

- **BSP_01_2_M7: Estimation and Adaptive Filter**:
   - Estimating a random vector by observing another vector
     - The most probable estimation/least error estimation/maximum likelihood estimation/linear estimation/affine estimation
   - Linear estimation of one process in terms of observations of another process
     - Non-causal IIR Wiener filter (smoothing)
     - Causal IIR Wiener filter (filtering)
     - Causal FIR Wiener filter
   - Fixed FIR Wiener filter problems
   - Adaptive filter in the area of noise estimation and removal
   - LMS Algorithm

- **BSP_01_2_M8: Kalman Filter**

- **BSP_01_2_M9: Classification**:
  - Bayes statistical classification
  - Bayes statistical classification considering different risk for error
  - Bayes statistical classification assuming Gaussian distribution of features
  - Class K is the nearest neighbor
  - Dimension feature reduction
  - Assessment of classification performance

- **BSP_01_2_M10: Cepstrum Analysis**:
  - Definition of mixed capstrom and real capstrom
  - Mixed capstrom of signals with rational fraction z-transform
  - Mixed Capstrum properties
  - Calculation of mixed capstrom in the time domain
  - Calculation of mixed capstrom using DFT
  - Calculation of the complex cepstrum of the phase signal from the real cepstrum
  - Important feature of mixed capstrom: converting convolution to sum
  - The origin of the name Capstrum
  - Deconvolution
