from setuptools import setup, find_packages

setup(
    name="MC-DFM",  # Package name
    version="1.0.0",  # Package version
    author="Huat Chiang",
    author_email="huatc@uw.edu",
    description="Simulated the small angle scattering curves of large biomolecular assemblies using the MC-DFM",
    long_description=open("README.md").read(),  # Load README as description
    long_description_content_type="text/markdown",
    url="https://github.com/pozzo-research-group/MC-DFM/tree/main",  # Project URL
    packages=find_packages(),  # Automatically find sub-packages
    install_requires=[
        'matplotlib==3.10.8',
        'numpy==2.3.0',
        'pandas==2.2.3',
        'scikit-learn==1.6.1',
        'scipy==1.15.1',
        'plotly==6.0.0',
        'h5py==3.12.1',
        'nbformat==5.10.4',
        'torch==2.8.0',
        'ipykernel==6.29.5',
        'openpyxl==3.1.5',
        'openai==2.24.0',
        'openai-agents==0.10.2',
        'agapi==2025.11.15',
        'gsd==3.4.2',
        'gradio==5.49.1'
    ],  # Dependencies to install
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",  # Minimum Python version
)