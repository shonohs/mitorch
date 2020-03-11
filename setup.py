import setuptools


setuptools.setup(name='mitorch',
                 version='0.0.1',
                 author='shono',
                 description="MiTorch training framework",
                 url='https://github.com/shonohs/mitorch',
                 packages=setuptools.find_packages(),
                 install_requires=['mitorch-models', 'pytorch_lightning', 'torch', 'torchvision', 'sklearn'],
                 entry_points={
                     'console_scripts': [
                         'mitrain=mitorch.train:main'
                         ]
                 },
                 classifiers=[
                     'Development Status :: 4 - Beta',
                     'License :: OSI Approved :: MIT License',
                     'Programming Language :: Python :: 3 :: Only',
                     'Topic :: Scientific/Engineering :: Artificial Intelligence'
                 ],
                 license='MIT')
