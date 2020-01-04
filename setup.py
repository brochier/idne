from setuptools import setup, find_packages

import os 
current_folder = os.path.dirname(os.path.abspath(__file__))
version = '0.0.0.0.0.0' # year.month.day.hour.minute.second
with open(os.path.join(current_folder,'VERSION')) as version_file:
    version = version_file.read().strip()

setup(name='idne',
      version=version,
      description='IDNE: Python package for the paper "Inductive Document Network Embedding with Topic-Word Attention" presented at ECIR 2020.',
      url='https://github.com/brochier/idne',
      author='Robin Brochier',
      author_email='robin.brochier@univ-lyon2.fr',
      license='MIT',
      include_package_data=True,
      packages=find_packages(exclude=['tests/data/*']),
      package_data={'': ['idne/resources/*', 'idne/conf.yml']},
      install_requires=[
          'numpy',
          'scipy',
          'sklearn',
          'matplotlib',
          'unidecode',
          'tensorflow',
          'theano',
          'gensim',
          'panda',
          'scikit-multilearn'
      ],
      zip_safe=False)
