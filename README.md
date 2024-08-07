# Climate States Estimation from Time Series Data

**Original paper published by**: V. N. Livina et al., 2010  
**Published in**: Clim. Past, 6, 77–82, 2010  
**Correspondence**: V. N. Livina (v.livina@uea.ac.uk)

## Overview

This repository contains methods and results for estimating the number of climate states in geophysical systems using time series data. The methodology applies a novel approach to detect bifurcations and transitions between different climate states, particularly in paleoclimate records.

## Methodology

### 1. Conceptual Model

The climate system is modeled as a nonlinear dynamical system with multiple states. Transitions between these states are driven by stochastic noise, represented by the Langevin equation:

\[ dz = -U'(z) \, dt + \sigma \, dW \]

where:
- \( U(z) \) is the potential function
- \( \sigma \) is the noise level
- \( W \) denotes a Wiener process

### 2. Potential Function

The potential \( U(z) \) is approximated by a polynomial of even order \( L \):

\[ U(z) = \sum_{i=1}^L a_i z^i \]

Higher-order polynomials allow the accommodation of more states. For example, a fourth-order polynomial can describe a system with two states.

### 3. Probability Density and Potential Reconstruction

The potential is reconstructed from the time series data using the relationship:

\[ p(z) \sim \exp\left[-\frac{2U(z)}{\sigma^2}\right] \]

The potential is estimated as:

\[ U = -\frac{\sigma^2}{2} \log p_d \]

where \( p_d \) is the empirical probability density function estimated using a Gaussian kernel.

### 4. Determining Number of States

The number of states \( S \) is inferred from the number of inflection points in the polynomial potential using:

\[ S = 1 + \frac{I}{2} \]

where \( I \) is the number of inflection points.

### 5. Sliding Window Analysis

The method uses sliding windows of varying sizes to analyze time series data and capture changes in the number of states over time.

## Results

### 1. Artificial Data

The method accurately identifies the number of states in artificial data generated from known potential functions, including:
- **One-well potential**: Identified correctly as a single state.
- **Double-well potential**: Correctly identified as two states.
- **Triple-well potential**: Correctly identified as three states.
- **Four-well potential**: Correctly identified as four states.

### 2. Ice-Core Data

#### GRIP and NGRIP Records

- **Period**: Last 60 kyr
- **Findings**:
  - Transition from a two-state to a one-state system around 25 kyr BP, indicating a bifurcation.
  - Consistent detection of two states for the Dansgaard-Oeschger events.
  - Transition to a one-state system detected before the Last Glacial Maximum (LGM).

#### Calcium Data

- **Source**: GRIP calcium data with annual resolution
- **Period**: 60–11 kyr BP
- **Findings**: Supports the bifurcation results observed in GRIP and NGRIP δ18O data.

### Interpretation

- **Bifurcation**: Detected transition from a two-state to a one-state climate system prior to the LGM, indicating the loss of a stable warm interstadial state.
- **Event Detection**: The results align with known paleoclimatic events but offer a refined understanding of their timing and implications.

## Conclusion

The methodology provides a robust approach for estimating the number of states in geophysical systems from time series data. It effectively detects changes in climate states and bifurcations, contributing to a deeper understanding of past climate dynamics and transitions.

For more details, please refer to the original paper or contact the authors.
