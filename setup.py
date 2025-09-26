import os
import sys
import tempfile

import numpy as np
import pybind11
import setuptools
from setuptools import Extension, setup, find_packages
from setuptools.command.build_ext import build_ext

__version__ = '0.1.4'

def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    with tempfile.NamedTemporaryFile('w', suffix='.cpp', delete=False) as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        f.flush()
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
            return True
        except setuptools.distutils.errors.CompileError:
            return False
        finally:
            try:
                os.unlink(f.name)
                os.unlink(f.name + '.o')
            except OSError:
                pass

def cpp_flag(compiler):
    """Return the -std=c++[14/11] compiler flag."""
    if has_flag(compiler, '-std=c++14'):
        return '-std=c++14'
    elif has_flag(compiler, '-std=c++11'):
        return '-std=c++11'
    else:
        raise RuntimeError('Unsupported compiler -- at least C++11 support is needed!')

class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""

    def build_extensions(self):
        ct = self.compiler.compiler_type

        opts = []
        link_opts = []

        if ct == 'unix':
            opts.append(cpp_flag(self.compiler))

            opts.extend(['-O3', '-DNDEBUG'])

            opts.extend([
                '-fno-builtin-malloc',
                '-fno-builtin-calloc',
                '-fno-builtin-realloc',
                '-fno-builtin-free',
                '-Wall',
                '-DINFO'
            ])

            if not os.environ.get("NSG_NO_NATIVE"):
                if has_flag(self.compiler, '-march=native'):
                    opts.append('-march=native')
                    print('Using -march=native for optimization')
                elif sys.platform == 'darwin' and has_flag(self.compiler, '-mcpu=apple-m1'):
                    opts.append('-mcpu=apple-m1')
                    print('Using -mcpu=apple-m1 for Apple Silicon')

            if sys.platform == 'darwin':
                if has_flag(self.compiler, '-fopenmp'):
                    opts.append('-fopenmp')
                    link_opts.extend(['-fopenmp'])
                    print('Using -fopenmp for OpenMP')
                else:
                    print('Warning: OpenMP not found. NSG may not work properly.')
                    print('Consider installing libomp: brew install libomp')
            else:
                opts.append('-fopenmp')
                link_opts.extend(['-fopenmp', '-pthread'])

            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())

            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')

        elif ct == 'msvc':
            opts.extend(['/EHsc', '/O2', '/openmp'])
            opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())

        for ext in self.extensions:
            ext.extra_compile_args.extend(opts)
            ext.extra_link_args.extend(link_opts)

        build_ext.build_extensions(self)

include_dirs = [
    pybind11.get_include(),
    np.get_include(),
    './include/',
]

source_files = [
    'pynsg/bindings.cpp',
    'src/index_nsg.cpp',
    'src/index.cpp',
]

missing_files = []
for src in source_files:
    if not os.path.exists(src):
        missing_files.append(src)

if missing_files:
    print("ERROR: Missing source files:")
    for f in missing_files:
        print(f"  {f}")
    print("\nPlease check your file structure and update source_files in setup.py")
    sys.exit(1)

libraries = []
if sys.platform != 'darwin':
    libraries.extend(['gomp'])

ext_modules = [
    Extension(
        'pynsg._bindings',
        source_files,
        include_dirs=include_dirs,
        libraries=libraries,
        language='c++',
        define_macros=[
            ('VERSION_INFO', '"{}"'.format(__version__)),
        ],
    ),
]

def get_long_description():
    readme = open('pynsg/README.md', 'r')
    readme_text = readme.read()
    readme.close()
    return readme_text

setup(
    name='pynsg',
    version=__version__,
    description='Python bindings for Navigating Spreading-Out Graph (NSG)',
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    author='Theodor Wuebker',
    url='https://github.com/twuebker/nsg',
    project_urls={
        'Original NSG': 'https://github.com/ZJULearning/nsg',
        'Bug Reports': 'https://github.com/twuebker/nsg/issues',
        'Source': 'https://github.com/twuebker/nsg',
    },
    ext_modules=ext_modules,
    packages=find_packages(),
    install_requires=[
        'numpy>=1.16.0',
    ],
    extras_require={
	'knn': ['faiss-cpu>=1.12.0'],	
    },
    entry_points={
        'console_scripts': [
            'nsg-build-knn=pynsg.graph_creator:main',
        ],
    },
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,
    python_requires=">=3.6",
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
		'Operating System :: POSIX :: Linux',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    keywords='nsg graph nearest neighbor search ann',
    license='MIT',
)
