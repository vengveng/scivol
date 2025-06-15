# Project Roadmap & TODO List

## Version 0.1 (Target: October 2024)

- [x] **Basic GARCH Model Implementation**

  - [x] Implement GARCH and GJR model with Normal distribution
    - [x] Implement volatility calculation method
    - [x] Implement univariate Normal distribution density
    - [x] Handle data pipeline for univariate models
    - [x] Improve ModelResult output
  - [ ] Test basic parameter estimation

- [ ] **Refactor and Documentation**
  - [x] Refactor `ModelResult` class to improve structure
    - [x] Parameter dictionary
    - [x] Add core parameter signatures
  - [ ] Write basic user guide for GARCH usage
  - [ ] Add comments and clean up codebase for clarity
- [ ] **Transition to 'Method' paragidm**
  - [x] Implement the MLE framework

## Version 0.2 (Target: 2024)

- [ ] **Support for GARCH with Student-t Distribution**
  - [x] Implement univariate Student-t distribution
  - [x] Implement Skewed-t
  - [ ] Test Student-t distribution with different datasets
  - [ ] Ensure that parameter significance tests work with Student-t
- [ ] **Multivariate Support (DCC Model)**
  - [x] Implement basic DCC model structure
  - [x] Add DCC model parameter estimation
  - [ ] Explicit user warnings for univariate/multivariate inputs
  - [ ] Write test cases for DCC model validation
- [ ] **Performance Optimization**
  - [ ] Optimize volatility calculation for large datasets
  - [ ] Implement parallelization in the optimization routine
  - [ ] Profile code and eliminate bottlenecks
  - [ ] Add logging for debugging
- [ ] **Statistical Testing**
  - [ ] Add support for residual variance testing
  - [ ] Implement ARCH-LM test for volatility clustering
  - [ ] Add hypothesis testing for parameter significance
  - [ ] Implement automatic testing for optimal algorithm and convergence
- [ ] **Advanced Statistical Features**

  - [ ] Standard errors estimation
  - [ ] Add confidence intervals for model parameters
  - [ ] Add rolling GARCH estimation for time-varying parameters
  - [ ] Implement goodness-of-fit tests for residuals
  - [ ] Autographing for ModelResult objects
  - [ ] QMLE and other frameworks

## Wishlist

- [ ] HAR models
