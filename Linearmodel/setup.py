from setuptools import setup, find_packages

setup(
    name='Linearmodel',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib'
    ],
    description='Un package pour effectuer des régressions linéaires et visualiser les résultats',
    author='MEBARKA Sheima',
    author_email='sheima.meb@gmail.com',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
)
