import setuptools


setuptools.setup(name='mitorch',
                 version='0.0.2',
                 author='shono',
                 description="MiTorch training framework",
                 url='https://github.com/shonohs/mitorch',
                 packages=setuptools.find_namespace_packages(include=['mitorch', 'mitorch.*']),
                 install_requires=['mitorch-models',
                                   'pymongo',
                                   'pytorch_lightning~=1.3.0',
                                   'requests',
                                   'tenacity',
                                   'torch~=1.9.0',
                                   'torchvision>=0.5.0',
                                   'sklearn',
                                   'albumentations'],
                 entry_points={
                     'console_scripts': [
                         'miagent=mitorch.commands.agent:main',
                         'misubmit=mitorch.commands.submit:main',
                         'mitrain=mitorch.commands.train:main',
                         'miquery=mitorch.commands.query:main',
                         ]
                 },
                 classifiers=[
                     'Development Status :: 4 - Beta',
                     'License :: OSI Approved :: MIT License',
                     'Programming Language :: Python :: 3 :: Only',
                     'Topic :: Scientific/Engineering :: Artificial Intelligence'
                 ],
                 python_requires='>=3.8',
                 license='MIT')
