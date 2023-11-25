import setuptools

setuptools.setup(name='var-next',
      version='0.1',
      description='The funniest joke in the world',
      url='https://github.com/sanyalsc/var-Next',
      author='Shantanu Sanyal',
      author_email='sanyalster@gmail.com',
      license='MIT',
      packages=setuptools.find_packages("src"),
      package_dir={'':"src"},
      install_requires=[
      ])
