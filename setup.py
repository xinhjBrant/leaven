from setuptools import find_namespace_packages, setup

with open("README.md", "r") as f:
  long_description = f.read()

setup(name='leaven',
      version='1.1.0b30',
      license='MIT',
      description='A package bridging Python and Lean verification for proof search using language models',
      long_description=long_description,
      author='Huajian Xin',
      author_email='xinhuajian2000@gmail.com',
      url='https://github.com/xinhjBrant/leaven',
      download_url = 'https://github.com/user/reponame/archive/v_01.tar.gz',
      install_requires=[
        "networkx", "pathlib", "datetime", "pytz", "tqdm"
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
      )
