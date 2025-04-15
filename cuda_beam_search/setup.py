from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='cuda_beam_search',
    packages=find_packages(),
    ext_modules=[
        CUDAExtension('cuda_beam_search.cuda_beam_search', [
            'src/cuda_beam_search.cu',
        ]),
        CUDAExtension('cuda_beam_search.diverse.cuda_beam_search', [
            'src/diverse/cuda_beam_search.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
) 