from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='peft_torch_backend',
    ext_modules=[
        CppExtension(
            name='peft_torch_backend',
            sources=['src/turbo_fusion_torch.cpp'],
            extra_compile_args=['-std=c++17']
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
