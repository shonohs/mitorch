import setuptools


setuptools.setup(name='mitorch',
                 version='0.0.1',
                 author='shono',
                 description="MiTorch training framework",
                 url='https://github.com/shonohs/mitorch',
                 packages=setuptools.find_namespace_packages(include=['mitorch.*']),
                 install_requires=['mitorch-models', 'pymongo', 'pytorch_lightning', 'requests', 'torch', 'torchvision', 'sklearn', 'azureml-sdk'],
                 entry_points={
                     'console_scripts': [
                         'micontrol=mitorch.service.control:main',
                         'mitrain=mitorch.train:main',
                         'misubmit=mitorch.service.submit:main',
                         'miquery=mitorch.service.query:main',
                         'miamlmanager=mitorch.azureml.manager:main',
                         'miamlrun=mitorch.azureml.runner:main',
                         'midataset=mitorch.service.dataset:main'
                         ]
                 },
                 classifiers=[
                     'Development Status :: 4 - Beta',
                     'License :: OSI Approved :: MIT License',
                     'Programming Language :: Python :: 3 :: Only',
                     'Topic :: Scientific/Engineering :: Artificial Intelligence'
                 ],
                 license='MIT')
