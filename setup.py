from distutils.core import setup
from setuptools import find_packages

with open("README.md", "r") as f:
  long_description = f.read()

setup(name='leaven',
      version='0.0.0',
      license='MIT',
      description='A package bridging Python and Lean verification for proof search using language models',
      long_description=long_description,
      author='Huajian Xin',
      author_email='xinhuajian2000@gmail.com',
      url='https://github.com/xinhjBrant/leaven',
      download_url = 'https://github.com/user/reponame/archive/v_01.tar.gz',
      install_requires=[
        "networkx", "tarfile", "urllib", "zipfile", "pathlib", "platform", "datetime", "pytz", "shelve"
      ],
      packages=find_packages(),
      platforms=["all"],
      classifiers=[
          'Intended Audience :: Developers',
          'Operating System :: OS Independent',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3.9',
          'Topic :: Software Development :: Libraries'
      ],
      python_requires='>=3.9',
      )
