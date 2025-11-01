# Introduction to Inverse Problems and Imaging

This repository contains materials for the master's course "Introduction to Inverse Problems and Imaging" at ČVUT. It includes homework assignments, lecture notes, and MATLAB code focused on solving inverse problems using numerical methods and regularization techniques.

## Homework Assignments

### Homework 1: Abel Integral Equation
This homework explores the inversion of the Abel integral equation, a classic ill-posed problem in inverse problems, using regularization techniques like Truncated SVD and Tikhonov regularization to handle noise and instability.
- **Exam Slides**: [Exam_slides_1_Martin_Kunz.pdf](https://github.com/kunzaatko/InverseProblemsAndImaging/releases/download/v1.0.0/Exam_slides_1_Martin_Kunz.pdf)
- **Report**: [HW_1_Martin_Kunz.pdf](https://github.com/kunzaatko/InverseProblemsAndImaging/releases/download/v1.0.0/HW_1_Martin_Kunz.pdf)

Key figures:
- ![Singular values of the Abel operator](homeworks/HW_1/src/figs/fig_singular_values.svg)  
  Decay of singular values illustrating the ill-posedness of the problem.
- ![TSVD reconstruction comparison](homeworks/HW_1/src/figs/fig_abel_TSVD.svg)  
  Truncated SVD regularization applied to noisy Abel data.
- ![Tikhonov regularization](homeworks/HW_1/src/figs/fig_abel_Tikhonov.svg)  
  Tikhonov method balancing data fit and smoothness.

### Homework 2: Backwards Heat Equation
This assignment addresses the backwards heat equation, an extremely ill-posed inverse problem where small errors in data measurements lead to large errors in the reconstructed initial condition, demonstrating the need for advanced regularization methods.
- **Exam Slides**: [Exam_slides_2_Martin_Kunz.pdf](https://github.com/kunzaatko/InverseProblemsAndImaging/releases/download/v1.0.0/Exam_slides_2_Martin_Kunz.pdf)
- **Report**: [HW_2_Martin_Kunz.pdf](https://github.com/kunzaatko/InverseProblemsAndImaging/releases/download/v1.0.0/HW_2_Martin_Kunz.pdf)

Key figures:
- ![Karin-Harbo test case](homeworks/HW_2/src/figs/fig_karin_harbo.png)  
  Initial condition for the backwards heat equation problem.
- ![Landweber iteration reconstruction](homeworks/HW_2/src/figs/fig_landweber_reconstruction.png)  
  Iterative Landweber method recovering the initial temperature distribution.
- ![Comparison of regularization methods](homeworks/HW_2/src/figs/fig_comparison_tikhonov_sobolev.png)  
  Tikhonov vs. Sobolev regularization for stability.

### Homework 3: Autoconvolution Inverse Problem
This homework deals with the non-linear inverse problem of autoconvolution, where the forward operator is a convolution of the unknown function with itself, solved using Newton's method with appropriate regularization for convergence.
- **Report**: [HW_3_Martin_Kunz.pdf](https://github.com/kunzaatko/InverseProblemsAndImaging/releases/download/v1.0.0/HW_3_Martin_Kunz.pdf)

Key figures:
- ![Forward autoconvolution comparison](homeworks/HW_3/src/figs/fig_compare_quadrature_v_analytical_forward_autoconvolution.png)  
  Numerical vs. analytical forward operator for autoconvolution.
- ![Newton method iterations](homeworks/HW_3/src/figs/fig_newton_iterations_reg_lines.png)  
  Convergence of Newton's method for the non-linear inverse problem.
