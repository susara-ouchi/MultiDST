from setuptools import setup, find_packages

setup(
    name='multidst_001',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "statmodels",
        "seaborn"
    ],
    author="Susara Ouchithya",
    description="Multiple Testing made easy!",
    long_description=open("README.md").read(),
    long_description_content_type = "text/markdown"
)
