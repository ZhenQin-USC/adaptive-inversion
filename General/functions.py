from collections.abc import Sequence
from typing import (Union, Optional, Dict, List, Tuple)
from functools import reduce
from operator import mul


def compute_paddings_for_conv(strides, kernel_size):
    """
    Computes the required padding for a given stride and kernel size in regular convolutions (downsampling).

    Args:
        strides (int or tuple): The stride values (s_x, s_y, s_z).
        kernel_size (int or tuple): The kernel size values (k_x, k_y, k_z).

    Returns:
        tuple: (p_x, p_y, p_z), the computed padding values.
    """
    # Convert integers to 3D tuples
    strides = (strides,) * 3 if isinstance(strides, int) else strides
    kernel_size = (kernel_size,) * 3 if isinstance(kernel_size, int) else kernel_size

    # Check if stride > kernel_size
    if any(s > k for s, k in zip(strides, kernel_size)):
        raise ValueError("Stride values cannot be greater than kernel size values.")

    # Compute proper padding
    padding = tuple((k - s + 1) // 2 for s, k in zip(strides, kernel_size))

    return padding


def compute_paddings_for_deconv(strides, kernel_size):
    """
    Computes the required padding and output padding for a given stride and kernel size
    in transposed convolutions (deconvolutions).

    Args:
        strides (int or tuple): The stride values (s_x, s_y, s_z).
        kernel_size (int or tuple): The kernel size values (k_x, k_y, k_z).

    Returns:
        tuple: (padding, output_padding), where:
            - padding (tuple): The computed padding values (p_x, p_y, p_z).
            - output_padding (tuple): The computed output padding values (op_x, op_y, op_z).
    """
    # Convert to 3D tuple if input is an integer
    strides = (strides,) * 3 if isinstance(strides, int) else strides
    kernel_size = (kernel_size,) * 3 if isinstance(kernel_size, int) else kernel_size

    padding = []
    output_padding = []

    for s, k in zip(strides, kernel_size):
        p = (k - s) // 2  # Compute symmetric padding
        if (k - s) % 2 != 0:  # Ensure even padding
            p += 1  # Adjust to make padding even
        padding.append(p)

        # Correct output padding calculation
        op = (s - (k - 2 * p) % s) % s
        output_padding.append(op)

    return tuple(padding), tuple(output_padding)


def calculate_grain_factors(blk_idx, strides):
    # First, convert strides to a list if it's not already
    strides_list = list(strides)

    # Call existing methods to calculate factors
    h_factors, z_factor = calculate_factors_product(blk_idx, strides_list)
    decode_factors, _ = calculate_factors_collect(blk_idx, strides_list)

    # Set the first decode factor as required
    decode_factor = decode_factors[0][::-1]

    # Calculate grain factors
    base_factor = h_factors[0]
    all_factors = h_factors[1:] + [z_factor]
    grain_factors = []
    for i in range(len(all_factors)):
        rel_factors = [int(c/b) for c, b in zip(all_factors[i][0], base_factor[0])]
        grain_factors.append(tuple(rel_factors))

    # Split the grain factors into h_grain_factors and z_grain_factor
    h_grain_factors = grain_factors[:-1]
    z_grain_factor = grain_factors[-1]

    return decode_factor, h_grain_factors, z_grain_factor


def calculate_factors_product(blk_idx, all_strides):    
    # Calculate the cumulative product of the strides for each dimension
    h_factors = []
    for i in range(len(all_strides)):
        # Calculate the cumulative product up to the current stride
        cumulative_stride = [reduce(mul, stride_dim[:i+1], 1) for stride_dim in zip(*all_strides)]
        h_factors.append([tuple(cumulative_stride)])
    return [h_factors[idx] for idx in blk_idx], h_factors[-1]   


def calculate_factors_collect(blk_idx, all_strides):
    # Collect the cumulative strides 
    h_factors = []
    for i in range(len(all_strides)):
        # Store the cumulative collection up to the current stride
        h_factors.append(all_strides[:i+1])
    return [h_factors[idx] for idx in blk_idx], h_factors[-1]     

