from setuptools import find_packages, setup

# Load text for description
with open('README.md') as f:
    readme = f.read()

# Load version number
# with open('version.txt', 'r') as f:
#     version = f.read()

# Go!
setup(
    # Module name (lowercase)
    name='pcpostprocess',
    version='0.0.1',
    description='Export high-throughput patch-clamp data from the Nanion SyncroPatch',
    long_description=readme,
    long_description_content_type="text/markdown",
    author='Frankie Patten-Elliot, Joseph Shuttleworth, Chon Lok Lei',
    author_email='joseph.shuttleworth@nottingham.ac.uk',
    maintainer='Joseph Shuttleworth',
    maintainer_email='joseph.shuttleworth@nottingham.ac.uk',
    # url='https://github.com/CardiacModelling/markov-builder',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],

    # Packages to include
    packages=find_packages(
        include=('pcpostprocess', 'pcpostprocess.scripts', 'pcpostprocess.*')),

    # Include non-python files (via MANIFEST.in)
    include_package_data=True,

    # Required Python version
    python_requires='>=3.7',

    # List of dependencies
    install_requires=[
        'scipy>=1.7',
        'numpy>=1.21',
        'matplotlib>=3.4',
        'pandas>=1.3',
        'regex>=2023.12.25',
        'openpyxl>=3.1.2',
        'jinja2>=3.1.0',
        'seaborn>=0.12.2'
    ],
    extras_require={
        'test': [
            'pytest-cov>=2.10',     # For coverage checking
            'pytest>=4.6',          # For unit tests
            'flake8>=3',            # For code style checking
            'isort',
            'mock>=3.0.5',         # For mocking command line args etc.
            'codecov>=2.1.3',
            'syncropatch_export @ git+ssh://git@github.com/CardiacModelling/syncropatch_export@main'
        ],
    },
    entry_points={
        'console_scripts': [
            'pcpostprocess='
            'pcpostprocess.scripts.__main__:main',
        ],
    },
)
