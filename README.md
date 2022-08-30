# ECPICK
[![Powered by ](https://img.shields.io/badge/Powered%20by-DataX%20Lab-orange.svg?style=flat&colorA=555&colorB=b42b2c)](https://www.dataxlab.org)
[![Powered by ](https://img.shields.io/badge/Powered%20by-CPS%20Lab-orange.svg?style=flat&colorA=555&colorB=007580)](https://www.sunmoon.ac.kr)

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ecpick)](https://pypi.org/project/ecpick/)
[![PyPI](https://img.shields.io/pypi/v/ecpick?style=flat&colorB=0679BA)](https://pypi.org/project/ecpick/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/ecpick?label=pypi%20downloads)](https://pypi.org/project/ecpick/)
[![PyPI - License](https://img.shields.io/pypi/l/ecpick)](https://pypi.org/project/ecpick/)

Enzyme Commission Predict Interpretable CK?? can

- **Website**: https://ecpick.dataxlab.org
- **Documentation**: https://readthedocs.org/projects/ecpick
- **Source code**: https://github.com/datax-lab/ECPICK


## Installation
**ECPICK** support Python 3.6+, Additionally, you will need 
```biopython```, ```numpy```, ```scikit-learn```, ```torch```, ```tqdm```.
However, these packages should be installed automatically when installing this codebase.

### Dependencies+

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ecpick)](https://pypi.org/project/ecpick/)
![PyPI](https://img.shields.io/pypi/v/torch?label=torch)
![PyPI](https://img.shields.io/pypi/v/biopython?label=biopython)
![PyPI](https://img.shields.io/pypi/v/numpy?label=numpy)
![PyPI](https://img.shields.io/pypi/v/scikit-learn?label=scikit-learn)
![PyPI](https://img.shields.io/pypi/v/tqdm?label=tqdm)


```ECPICK``` is available through PyPi and can easily be installed with a pip install
```shell
$ pip install ecpick
```

## Documentation
Read the documentation on readthedocs (Getting ready)

## Quick Start
```python
from ecpick import ECPICK

ecpick = ECPICK()
ecpick.predict_fasta(fasta_path='sample.fasta', output_path='output')
```

## Usage


## Links:

- ECPICK Web server: https://ecpick.dataxlab.org

## References
Sora Han, Mingyu Park, ECPICK (2022). <a href="#">link</a>