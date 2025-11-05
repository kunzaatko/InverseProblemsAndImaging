# Introduction to Inverse Problems and Imaging

This repository contains materials for the master's course "Introduction to Inverse Problems and Imaging" at ČVUT. It
includes homework assignments, lecture notes, and MATLAB code focused on solving inverse problems using numerical
methods and regularization techniques.

## Homework Assignments

### Homework 1: Abel Integral Equation
This homework explores the inversion of the Abel integral equation, a classic ill-posed problem in inverse problems,
using regularization techniques like Truncated SVD and Tikhonov regularization to handle noise and instability.
- **Exam Slides**: [Exam_slides_1_Martin_Kunz.pdf](https://github.com/kunzaatko/InverseProblemsAndImaging/releases/download/v1.0.0/Exam_slides_1_Martin_Kunz.pdf)
- **Report**: [HW_1_Martin_Kunz.pdf](https://github.com/kunzaatko/InverseProblemsAndImaging/releases/download/v1.0.0/HW_1_Martin_Kunz.pdf)

#### Figures
- Singular system comparison for the heat equation: **(1)** Singular vectors for the discretized operator and the
  analytical singular functions, **(2)** singular values comparison, **(3)** heatmap of singular vectors
<p align="center">
  <img alt="Singular vectors comparison" src="./homeworks/HW_1/src/figs/fig_singular_vectors.svg" width="45%">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="Singular values comparison" src="./homeworks/HW_1/src/figs/fig_singular_values.svg" width="45%">
</p>
<p align="center">
  <img alt="Heatmap singular vectors" src="./homeworks/HW_1/src/figs/fig_heatmap_singular_vectors.svg" width="60%">
</p>

- Demonstration of the discretized heat operator: **(1)** Singular functions and their images under K and A, **(2)** sum
  of a random linear combination of singular functions
<p align="center">
  <img alt="Singular demo" src="./homeworks/HW_1/src/figs/fig_singular_demo.svg" width="45%">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="Sum singular demo" src="./homeworks/HW_1/src/figs/fig_sum_singular_demo.svg" width="45%">
</p>

- Ill-posedness demonstration: **(1)** Fixed function, its image under K, and added noise, **(2)** Picard plot of
  singular values and coefficients
<p align="center">
  <img alt="Fixed function noised" src="./homeworks/HW_1/src/figs/fig_fixed_function_noised.svg" width="45%">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="Picard plot" src="./homeworks/HW_1/src/figs/fig_picard_plot.svg" width="45%">
</p>

- Abel problem regularization: **(1)** Ground truth function, forward transformed data, and noisy data, **(2)** optimal
  TSVD and Tikhonov reconstructions, **(3)** TSVD reconstructions for various k, **(4)** relative error dependence on k,
  **(5)** Tikhonov reconstructions for various α, **(6)** relative error dependence on α
<p align="center">
  <img alt="Abel intro" src="./homeworks/HW_1/src/figs/fig_abel_intro.svg" width="30%">
&nbsp; &nbsp;
  <img alt="Abel compare" src="./homeworks/HW_1/src/figs/fig_abel_compare.svg" width="30%">
&nbsp; &nbsp;
  <img alt="Abel TSVD" src="./homeworks/HW_1/src/figs/fig_abel_TSVD.svg" width="30%">
</p>
<p align="center">
  <img alt="Abel TSVD rel err" src="./homeworks/HW_1/src/figs/fig_abel_TSVD_relative_err.svg" width="30%">
&nbsp; &nbsp;
  <img alt="Abel Tikhonov" src="./homeworks/HW_1/src/figs/fig_abel_Tikhonov.svg" width="30%">
&nbsp; &nbsp;
  <img alt="Abel Tikhonov rel err" src="./homeworks/HW_1/src/figs/fig_abel_Tikhonov_relative_err.svg" width="30%">
</p>


- Discontinuous Abel regularization: **(1)** Discontinuous function, its forward transform, and noisy data, **(2)**
  optimal TSVD and Tikhonov reconstructions, **(3)** TSVD reconstructions for various k, **(4)** relative error
  dependence on k, **(5)** Tikhonov reconstructions for various α, **(6)** relative error dependence on α
<p align="center">
  <img alt="Abel discont intro" src="./homeworks/HW_1/src/figs/fig_discont_abel_intro.svg" width="30%">
&nbsp; &nbsp;
  <img alt="Abel discont compare" src="./homeworks/HW_1/src/figs/fig_abel_discont_compare.svg" width="30%">
&nbsp; &nbsp;
  <img alt="Abel discont TSVD" src="./homeworks/HW_1/src/figs/fig_abel_discont_TSVD.svg" width="30%">
</p>
<p align="center">
  <img alt="Abel discont TSVD rel err" src="./homeworks/HW_1/src/figs/fig_abel_discont_TSVD_relative_err.svg" width="30%">
&nbsp; &nbsp;
  <img alt="Abel discont Tikhonov" src="./homeworks/HW_1/src/figs/fig_abel_discont_Tikhonov.svg" width="30%">
&nbsp; &nbsp;
  <img alt="Abel discont Tikhonov rel err" src="./homeworks/HW_1/src/figs/fig_abel_discont_Tikhonov_relative_err.svg" width="30%">
</p>

- Deblurring convergence demonstration: **(1)** Reconstructions with different regularization parameters, **(2)**
  relative error as function of noise level δ
<p align="center">
  <img alt="Limit demo" src="./homeworks/HW_1/src/figs/fig_limit_demo.svg" width="45%">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="Numerical limit" src="./homeworks/HW_1/src/figs/fig_numerical_limit.svg" width="45%">
</p>

### Homework 2: Backwards Heat Equation
This assignment addresses the backwards heat equation, an extremely ill-posed inverse problem where small errors in data
measurements lead to large errors in the reconstructed initial condition, demonstrating the need for advanced
regularization methods.
- **Exam Slides**: [Exam_slides_2_Martin_Kunz.pdf](https://github.com/kunzaatko/InverseProblemsAndImaging/releases/download/v1.0.0/Exam_slides_2_Martin_Kunz.pdf)
- **Report**: [HW_2_Martin_Kunz.pdf](https://github.com/kunzaatko/InverseProblemsAndImaging/releases/download/v1.0.0/HW_2_Martin_Kunz.pdf)

#### Figures
- Heat equation forward solution: **(1)** The function $x(s_1, s_2) = \sin(ks_1) \sin(ls_2)$, **(2)** forward pass of
  the heat equation on implemented using FFT
<p align="center">
  <img alt="Ground truth for the heat equation" src="./homeworks/HW_2/src/figs/fig_initial_condition.png" width="40%">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="Forward solution to the heat equation on sinusoidal function in 2D" src="./homeworks/HW_2/src/figs/fig_forward_solution.png" width="40%">
</p>

- Karin-Harbo painting heat equation inversion: **(1)** Ground truth image, **(2)** forward passed image with added 10%
  magnitude noise, **(3)** reconstructed image using Landweber iterations
<p align="center">
  <img alt="Karin-Harbo ground truth" src="./homeworks/HW_2/src/figs/fig_karin_harbo.png" width="30%">
&nbsp; &nbsp;
  <img alt="Karin-Harbo forward passed" src="./homeworks/HW_2/src/figs/fig_karin_harbo_forward_noisy.png" width="30%">
&nbsp; &nbsp;
  <img alt="Karin-Harbo Landweber iteration reconstruction" src="./homeworks/HW_2/src/figs/fig_landweber_reconstruction.png" width="30%">
</p>

- Abel problem inversion: **(1)** Forward passed data $Ay$, analytical forward problem solution $y$ and the noisy data
  $y^\delta$
<p align="center">
  <img alt="Abel problem forward and analytical" src="./homeworks/HW_2/src/figs/fig_abel.png" width="50%">
</p>

**(2)** Finding the ideal regularization coefficient for the Tikhonov regularization using the discrepancy principle,
**(3)** and the ideal regularization coefficient for the Sobolev regularization
<p align="center">
  <img alt="Tikhonov regularization coefficient by discrepancy principle" src="./homeworks/HW_2/src/figs/fig_discrepancy_tikhonov.png" width="40%">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="Sobolev regularization coefficient by discrepancy principle" src="./homeworks/HW_2/src/figs/fig_discrepancy_sobolev.png" width="40%">
</p>

**(4)** Tikhonov and Sobolev regularized reconstructions and the initial function, **(5)** reconstruction using only the
$L_1$ matrix
<p align="center">
  <img alt="Tikhonov and Sobolev regularized reconstructions" src="./homeworks/HW_2/src/figs/fig_comparison_tikhonov_sobolev.png" width="40%">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="L_1 matrix reconstruction" src="./homeworks/HW_2/src/figs/fig_abel_l1.png" width="40%">
</p>

### Homework 3: Autoconvolution Inverse Problem
This homework deals with the non-linear inverse problem of autoconvolution, where the forward operator is a convolution
of the unknown function with itself, solved using Newton's method with appropriate regularization for convergence.
- **Report**: [HW_3_Martin_Kunz.pdf](https://github.com/kunzaatko/InverseProblemsAndImaging/releases/download/v1.0.0/HW_3_Martin_Kunz.pdf)

#### Figures
- Forward autoconvolution operator comparison: **(1)** Numerical quadrature solution vs. analytical solution, **(2)** their difference
<p align="center">
  <img alt="Compare quadrature v analytical forward autoconvolution" src="./homeworks/HW_3/src/figs/fig_compare_quadrature_v_analytical_forward_autoconvolution.png" width="45%">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="Difference quadrature v analytical" src="./homeworks/HW_3/src/figs/fig_compare_quadrature_v_analytical_forward_autoconvolution_diff.png" width="45%">
</p>

- Newton iteration reconstruction: **(1)** First Newton step with Tikhonov regularization, **(2)** convergence of first
  4 iterations with increasing regularization
<p align="center">
  <img alt="First Newton step" src="./homeworks/HW_3/src/figs/fig_first_newton_step.png" width="45%">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="Newton iterations reg lines" src="./homeworks/HW_3/src/figs/fig_newton_iterations_reg_lines.png" width="45%">
</p>

- Newton iterations from distant initial function: **(1)** 15 iterations failing to converge to the true solution,
  **(2)** 4 iterations converging to the negative solution
<p align="center">
  <img alt="Newton iterations alt" src="./homeworks/HW_3/src/figs/fig_newton_iterations_reg_lines_alt.png" width="45%">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="Newton iterations minus" src="./homeworks/HW_3/src/figs/fig_newton_iterations_reg_lines_minus.png" width="45%">
</p>
