import setuptools


setuptools.setup(name='mitorch',
                 version='0.1.1',
                 author='shono',
                 description="MiTorch training framework",
                 url='https://github.com/shonohs/mitorch',
                 packages=setuptools.find_namespace_packages(include=['mitorch', 'mitorch.*']),
                 install_requires=['mitorch-models',
                                   'albumentations',
                                   'jsons',
                                   'pymongo',
                                   'pytorch_lightning~=1.4.5',
                                   'requests',
                                   'scikit-learn',
                                   'tenacity',
                                   'torch~=1.9.0',
                                   'torchvision>=0.5.0'],
                 entry_points={
                     'console_scripts': [
                         'miagent=mitorch.commands.agent:main',
                         'mipredict=mitorch.commands.predict:main',
                         'misubmit=mitorch.commands.submit:main',
                         'mitrain=mitorch.commands.train:main',
                         'miquery=mitorch.commands.query:main',
                         ]
                 },
                 classifiers=[
                     'Development Status :: 4 - Beta',
                     'License :: OSI Approved :: MIT License',
                     'Topic :: Scientific/Engineering :: Artificial Intelligence'
                 ],
                 python_requires='>=3.8',
                 license='MIT')
