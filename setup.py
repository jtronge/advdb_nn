from setuptools import find_packages, setup

setup(
    name='advdb_nn',
    version='0.1',
    # TODO: Add scripts to run here
    scripts=[],
    # Find all packages in the source directory
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=[
        'requests',
    ],
)
