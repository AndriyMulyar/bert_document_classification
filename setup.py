from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand
from bert_document_classification import __version__, __authors__
import sys

packages = find_packages()

def readme():
    with open('README.md') as f:
        return f.read()



class PyTest(TestCommand):
    """
    Custom Test Configuration Class
    Read here for details: https://docs.pytest.org/en/latest/goodpractices.html
    """
    user_options = [("pytest-args=", "a", "Arguments to pass to pytest")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = ""

    def run_tests(self):
        import shlex
        # import here, cause outside the eggs aren't loaded
        import pytest

        errno = pytest.main(shlex.split(self.pytest_args))
        sys.exit(errno)

setup(
    name='bert_document_classification',
    version=__version__,
    license='MIT',
    description='long document classification with language models',
    long_description=readme(),
    packages=packages,
    long_description_content_type='text/markdown',
    url='https://github.com/AndriyMulyar/bert_document_classification',
    author=__authors__,
    author_email='contact@andriymulyar.com',
    keywords='BERT, document classification',
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.5',
        'Natural Language :: English',
        'Topic :: Text Processing :: Linguistic',
        'Intended Audience :: Science/Research'
    ],

    install_requires=[
        'pytorch-transformers',
        'torch',
        'configargparse',
        'scikit-learn'
    ],
    tests_require=["pytest"],
    cmdclass={"pytest": PyTest},
    include_package_data=True,
    zip_safe=False

)