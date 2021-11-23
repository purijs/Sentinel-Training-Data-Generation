from setuptools import setup

setup(
    name='training',
    version='0.0.1',
    description='Boundary segmentation package for generating training data',
    author='Jaskaran',
    author_email='jaskaran@training.co.in',
    license='Private',
    packages=['training','training.unsupervised','training.unsupervised.algorithms','training.unsupervised.ndim','training.unsupervised.metric'],
    zip_safe=False,
    install_requires=['matplotlib','scikit-image','rasterio','numpy','Pillow']
)