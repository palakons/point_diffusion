import os
from pathlib import Path

from torch.utils.cpp_extension import load

# print("backend.py import Function")

gcc_path = os.getenv('CC', default='/usr/bin/gcc')
# print(f"gcc_path: {gcc_path}")
if not Path(gcc_path).is_file():
    raise ValueError('Could not find your gcc, please replace it here.')

# print("backend.py import Function done")

_src_path = os.path.dirname(os.path.abspath(__file__))
# print(f"_src_path: {_src_path}")
# _backend = load(
#     name='_pvcnn_backend',
#     extra_cflags=['-O3', '-std=c++17'],
#     extra_cuda_cflags=[f'--compiler-bindir={gcc_path}'],
#     sources=[os.path.join(_src_path,'src', f) for f in [
#         'ball_query/ball_query.cpp',
#         'ball_query/ball_query.cu',
#         'grouping/grouping.cpp',
#         'grouping/grouping.cu',
#         'interpolate/neighbor_interpolate.cpp',
#         'interpolate/neighbor_interpolate.cu',
#         'interpolate/trilinear_devox.cpp',
#         'interpolate/trilinear_devox.cu',
#         'sampling/sampling.cpp',
#         'sampling/sampling.cu',
#         'voxelization/vox.cpp',
#         'voxelization/vox.cu',
#         'bindings.cpp',
#     ]]
# )


cuda_home = os.getenv('CUDA_HOME', default='/usr/local/cuda')#paalkons added here

_backend = load(
    name='_pvcnn_backend',
    extra_cflags=['-O3', '-std=c++17'],
    extra_cuda_cflags=[f'--compiler-bindir={gcc_path}'],
    extra_include_paths = [os.path.join(cuda_home, 'include')], #paalkons added here
    sources=[os.path.join(_src_path,'src', f) for f in [
        'ball_query/ball_query.cpp',
        'ball_query/ball_query.cu',
        'grouping/grouping.cpp',
        'grouping/grouping.cu',
        'interpolate/neighbor_interpolate.cpp',
        'interpolate/neighbor_interpolate.cu',
        'interpolate/trilinear_devox.cpp',
        'interpolate/trilinear_devox.cu',
        'sampling/sampling.cpp',
        'sampling/sampling.cu',
        'voxelization/vox.cpp',
        'voxelization/vox.cu',
        'bindings.cpp',
    ]]
)

# print(os.path.join(_src_path,'src', 'ball_query/ball_query.cpp'))
# #get CUDA_HOME
# print(f"cuda_home: {cuda_home}")
# load(
#     name='_pvcnn_backend',
#     extra_cflags=['-O3', '-std=c++17'],
#     extra_cuda_cflags=[f'--compiler-bindir={gcc_path}'],
#     extra_include_paths = [os.path.join(cuda_home, 'include')],
#     sources=[os.path.join(_src_path,'src', 'ball_query/ball_query.cpp')]
#     )
# print("backend.py  _backend load done")

__all__ = ['_backend']
