from distutils.core import setup
from setuptools import find_namespace_packages, Command
from leaven.src.lean_manager import get_lean, download_mathlib_datasets

class PostInstallCommand(Command):
    description = "Run get_lean after installation"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        get_lean()
        download_mathlib_datasets()

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
        "networkx", "tarfile", "urllib", "zipfile", "pathlib", "platform", "datetime", "pytz", "shelve", "tqdm"
      ],
      packages=find_namespace_packages(exclude=['leaven.__pycache__', 'leaven.src.__pycache__']),
      include_package_data=True,
      platforms=["all"],
      classifiers=[
          'Intended Audience :: Developers',
          'Operating System :: OS Independent',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3.9',
          'Topic :: Software Development :: Libraries'
      ],
      long_description_content_type="text/markdown",
      python_requires='>=3.9',
      cmdclass={
          'install': PostInstallCommand,
      },
      )
