from setuptools import setup, find_packages

setup(
    name='principal_portfolios',
    version='1.0.5',
    packages=find_packages(),
    install_requires=[],
    author='Amin Izadyar',
    author_email='a.izadyar23@imperial.ac.uk',
    description='A Python implementation of the Principal Portfolios methodology by Kelly, Malamud, and Pedersen (2023), enabling optimal asset allocation by exploiting cross-predictability among asset returns.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown', 
    url='https://github.com/aminizadyar/Principal-Portfolios',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
