from setuptools import setup, find_packages

setup(
    name="autoprocess_iitg",  
    version="0.1.0",  
    author="Shubham",
    author_email="shubhamsinghalswm123@gmail.com",
    description="Automated data preprocessing library using Google's Gemini AI",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ShubhamS1101/CleanGPT01",
    packages=find_packages(where="generate/src"),  
    package_dir={"": "generate/src"},
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
    license="MIT",  
    license_files = []
)
