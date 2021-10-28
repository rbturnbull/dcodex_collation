# dcodex_collation

![pipline](https://github.com/rbturnbull/dcodex_collation/actions/workflows/coverage.yml/badge.svg)
[<img src="https://img.shields.io/badge/code%20style-black-000000.svg">](<https://github.com/psf/black>)

An extension for dcodex to collate transcriptions.

# Installation

For a brand new dcodex site, it is easiest to install using [dcodex-cookiecutter](https://github.com/rbturnbull/dcodex-cookiecutter).

To install dcodex as a plugin in a Django site already set up. Install with pip:
```
pip install git+https://github.com/rbturnbull/dcodex_collation.git
```

Then add to your installed apps after `dcodex`:
```
INSTALLED_APPS += [
    # dcodex dependencies
    "adminsortable2",
    'easy_thumbnails',
    'filer',
    'mptt',
    'imagedeck',
    # dcodex apps
    "dcodex",
    "dcodex_collation",
]
```

Then add the urls to your main `urls.py` after the urls for `dcodex`:
```
urlpatterns += [
    path('dcodex/', include('dcodex.urls')),    
    path('dcodex_collation/', include('dcodex_collation.urls')),    
]
```

Documentation to come soon.

More information found at https://github.com/rbturnbull/dcodex
