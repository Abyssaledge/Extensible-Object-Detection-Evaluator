from setuptools import setup, find_packages

setup(
    name='UniversalDetectionEvaluator',
      packages = find_packages(exclude=['config']),
    #   package_dir={'':'src'},
      version='0.1.0',
      author='Lue Fan',
      install_requires=[]
    )