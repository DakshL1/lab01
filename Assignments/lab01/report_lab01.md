# Lab Report — Lab 01: Efficient Separable Gaussian Convolution

## Overview

The goal of this lab was to implement an efficient separable Gaussian filter that produces the same blurred output as a full 2D Gaussian convolution while substantially reducing computation time.

## Implementation

I implemented `GaussianFilt.gauss_kernel()` to generate a normalized 1D Gaussian kernel with size `int(6 * sigma + 1)`, using the standard Gaussian PDF and expanding it to a column vector. I then implemented `GaussianFilt.my_conv_method()` by padding the image with edge values, applying a 1D convolution along rows with the kernel, and then applying the same 1D convolution along columns. The final result was clipped to the valid grayscale range and converted to `np.uint8`.

## Results

The implementation passed all lab checks: the pytest suite completed with `10 passed in 6.79s`, and the provided evaluation script produced a convolution in `0.1004` seconds versus a direct `convolve2d` baseline of `5.7327` seconds. This is roughly `57x` faster than the naive full-kernel baseline and comfortably satisfies the strict performance thresholds used by the tests.
