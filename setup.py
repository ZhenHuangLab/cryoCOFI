from setuptools import setup, find_packages

setup(
    name='cryoCOFI',
    version='0.2.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'cupy>=13.3.0',
        'mrcfile>=1.5.3',
        'matplotlib>=3.9.2',
        'argparse',
        'numpy>=2.0.2',
        'tqdm>=4.66.5',
        'pandas>=2.2.3'
    ],
    package_data={
        'cryoCOFI': ['lib/*.so'],
    },
    include_package_data=True,
    license='GPLv3',
    description='CarbOn FIlm detector for cryo-EM images',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Zhen Huang',
    author_email='zhen.victor.huang@gmail.com',
    url='https://github.com/ZhenHuangLab/cryoCOFI',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
        'Development Status :: 3 - Alpha',
        'Environment :: GPU :: NVIDIA CUDA :: 12 :: 12.2',
    ],
    python_requires='>=3.9',
    entry_points={
        'console_scripts': [
            'cryoCOFI=cryoCOFI.main:main'
        ],
    },
    exclude_package_data={
        '': ['__pycache__'],
        '**': ['__pycache__', '*.py[co]'],
    },
    keywords='cryo-EM, cryo-ET, carbon film, edge detection, CUDA, CuPy',
    project_urls={
        'Bug Reports': 'https://github.com/ZhenHuangLab/cryoCOFI/issues',
        'Source': 'https://github.com/ZhenHuangLab/cryoCOFI',
    },
)
