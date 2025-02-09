from setuptools import setup, find_packages

setup(
    name="autoprocess",  # Package name on PyPI
    version="0.1.0",  # Update version for each new release
    author="Shubham",
    author_email="shubhamsinghalswm123@gmail.com",
    description="Automated data preprocessing library using Google's Gemini AI",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ShubhamS1101/CleanGPT01",
    packages=find_packages(where="generate/src"),  # Finds packages inside generate/src
    package_dir={"": "generate/src"},  # Tell setuptools where the package starts
    install_requires=[
        "numpy>=1.18.0",
        "pandas>=1.0.0",
        "scikit-learn>=0.24.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
