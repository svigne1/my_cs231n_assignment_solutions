from builtins import range
import numpy as np

def my_im2col_indices(x_shape, w_shape, pad, stride=1):
    N, C, H, W = x_shape
    F, C, HF, WF = w_shape
    H = H + 2 * pad
    W = W + 2 * pad

    # H * W is the image. HF * WF is the filter.
    # HO * WO is the output image

    assert (H - HF) % stride == 0
    assert (W - WF) % stride == 0

    HO = int((H - HF) / stride + 1)
    WO = int((W - WF) / stride + 1)

    # i,j indices in x(original image) for 1st tile in output image
    tile_i = np.repeat(np.arange(HF), WF)
    tile_j = np.tile(np.arange(WF), HF)

    # i,j indices in x for one entire cube in output image
    cube_i = np.tile(tile_i, C)
    cube_j = np.tile(tile_j, C)

    # i indices for all cubes in output image aling with stride corrections
    x_cubes_i = np.tile(cube_i, WO)
    xy_cubes_i = np.tile(x_cubes_i, HO)
    stride_i = np.arange(HO) * stride
    stride_i = np.repeat(stride_i, x_cubes_i.shape[0])
    i = xy_cubes_i + stride_i

    # j indices for all cubes in output image aling with stride corrections
    x_cubes_j = np.tile(cube_j, WO)
    stride_j = np.arange(WO) * stride
    stride_j = np.repeat(stride_j, cube_j.shape[0])
    x_cubes_j = x_cubes_j + stride_j
    j = np.tile(x_cubes_j, HO)

    # k indices(depth) for all cubes in output image.
    k = np.repeat(np.arange(C), HF * WF)
    k = np.tile(k, HO * WO)

    return k, i, j

def my_im2col(x, w_shape, pad, stride=1):
    N, C, H, W = x.shape
    F, C, HF, WF = w_shape

    x_pad = np.pad(x, ((0,), (0,), (pad,), (pad,)), 'constant')

    k, i, j = my_im2col_indices(x.shape, w_shape, pad, stride)

    # all_cubes of a single image is flattened out in a single array.
    all_cubes = x_pad[:, k, i, j]

    # Each flattened array of cubes is reshaped back for each image.
    return all_cubes.reshape(N, -1, HF * WF * C)

def my_col2im(grads, x_shape, w_shape, pad, stride=1):

    dx = np.zeros(x_shape[1:])
    dx = np.pad(dx, ((0,), (pad,), (pad,)), 'constant')

    k, i, j = my_im2col_indices(x_shape, w_shape, pad, stride)

    np.add.at(dx, (k, i, j), grads.reshape(-1))
    return dx
