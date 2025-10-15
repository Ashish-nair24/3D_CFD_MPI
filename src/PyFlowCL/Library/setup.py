from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(
    name='solver_cpp',
    ext_modules=[cpp_extension.CppExtension(name='solver_cpp', 
                                            sources=['Solver.cpp'])
    ],
    cmdclass={
        'build_ext': cpp_extension.BuildExtension
    }
)
