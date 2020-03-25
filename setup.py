from setuptools import setup

# Where the magic happens:
setup(
    name='final-year-project',
    version='1.0',
    description='abstractive text summarization',
	long_description=open('README.md').read(),
    author='bhm699',
    py_modules=['main'],
    install_requires=['numpy', 'pandas', 'sklearn', 'keras', 'matplotlib', 'rouge', ' networkx'],
    include_package_data=True,
)
