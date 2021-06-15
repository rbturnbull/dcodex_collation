from setuptools import setup

requirements = [
    'Django>3',
    'django-polymorphic',
    'django-extensions',
    'numpy',
    'pandas>=1.0.5',
    'matplotlib',
    'plotly',
    'dcodex @ git+https://github.com/rbturnbull/dcodex.git#egg=dcodex',
]

setup(
    install_requires=requirements,
)


