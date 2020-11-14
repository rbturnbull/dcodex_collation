from setuptools import setup

requirements = [
    'Django>3',
    'django-polymorphic',
    'numpy',
    'pandas>=1.0.5',
    'matplotlib',
    "jsonfield>=3.1.0",
    "django-ndarray>=0.0.3",
]

setup(
    install_requires=requirements,
)


