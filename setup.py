from setuptools import setup

setup(
    name="src",
    version="0.0.1",
    author="Kamalaksha",
    description="A small package for dvc ml pipeline demo",
    long_description_content_type="text/markdown",
    url="https://github.com/kamalakshancg/BigMart_Sales_Prediction",
    author_email="kamalaksha.ncg@gmail.com",
    packages=["src"],
    python_requires=">=3.7",
    install_requires=[
        'dvc',
        'pandas',
        'scikit-learn'
    ]
)