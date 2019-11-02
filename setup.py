from setuptools import setup

# Where the magic happens:
setup(
    name='final-year-project',
    version='1.0',
    description='text summarization',
	long_description=open('README.md').read(),
    author='bmistry12',
    # python_requires='3.5.0',
    py_modules=['main'],
    install_requires=['numpy', 'pandas', 'sklearn', 'keras'],
    include_package_data=True,
)
