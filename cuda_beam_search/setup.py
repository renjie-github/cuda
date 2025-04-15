from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='cuda_beam_search',
    version='0.1.0',
    packages=find_packages(),
    ext_modules=[
        CUDAExtension('cuda_beam_search.cuda_beam_search', [
            'src/cuda_beam_search.cu',
        ]),
        CUDAExtension('cuda_beam_search.diverse.cuda_beam_search', [
            'src/diverse/cuda_beam_search.cu',
        ])
    ],
    install_requires=[
        'torch>=1.7.0',
        'numpy>=1.19.0',
        'transformers>=4.0.0',
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    python_requires='>=3.6',
) 