from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='cuda_beam_search',
    ext_modules=[
        CUDAExtension('cuda_beam_search', [
            'src/cuda_beam_search.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
) 