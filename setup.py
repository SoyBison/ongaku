from setuptools import setup
with open('docs/requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='ongaku',
    version='1.0',
    packages=[''],
    install_requires=requirements,
    url='https://ongaku.readthedocs.io',
    license='',
    author='Coen D. Needell',
    author_email='',
    description=''
)
