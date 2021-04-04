from setuptools import setup

requirements = [
    'Django>3',
    'django-polymorphic',
    'numpy',
    'pandas>=1.0.5',
    'matplotlib',
    'dcodex @ git+https://github.com/rbturnbull/dcodex.git#egg=dcodex',
]

setup(
    install_requires=requirements,
)


