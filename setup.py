from setuptools import setup, find_packages

setup(
    name='geochemistry_helpers',
    version='0.1.0',
    packages=find_packages(include=['geochemistry_helpers/*']),
    install_requires=[
        'PyYAML',
        'numpy>=1.14.5',
        'matplotlib>=2.2.0'
    ]
)
