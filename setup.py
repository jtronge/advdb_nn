from setuptools import find_packages, setup

setup(
    name='advdb_nn',
    version='0.1',
    # TODO: Add scripts to run here
    # scripts=[],
    # Install the benchmark script as ivf-benchmark
    entry_points={
        'console_scripts': ['ivf-benchmark=advdb_nn.benchmark:main'],
    },
    # Find all packages in the source directory
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=[
        'requests',
        'sklearn',
        'numpy',
        'pyyaml',
    ],
)
