# ECPICK

[![Powered by ](https://img.shields.io/badge/Powered%20by-DataX%20Lab-orange.svg?style=flat&colorA=555&colorB=b42b2c)](https://www.dataxlab.org)
[![Powered by ](https://img.shields.io/badge/Powered%20by-CPS%20Lab-orange.svg?style=flat&colorA=555&colorB=007580)](https://www.sunmoon.ac.kr)

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ecpick)](https://pypi.org/project/ecpick/)
[![PyPI](https://img.shields.io/pypi/v/ecpick?style=flat&colorB=0679BA)](https://pypi.org/project/ecpick/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/ecpick?label=pypi%20downloads)](https://pypi.org/project/ecpick/)
[![PyPI - License](https://img.shields.io/pypi/l/ecpick)](https://pypi.org/project/ecpick/)

## Biologically interpretable deep learning enhances trustworthy enzyme commission number prediction and discovers potential motif sites

The rapid growth of uncharacterized enzymes and their functional diversity urge
accurate and trustworthy computational functional annotation tools. However,
current approaches lack trustworthiness for the predictions and model interpretation,
limiting model reliability on the multi-label classification problem with thousands
of classes. Here, we demonstrate that our novel biologically interpretable deep
learning model (ECPICK) provides a robust solution for trustworthy predictions
of enzyme commission (EC) numbers with significantly enhanced predictive power
and the capability to discover potential motif sites. ECPICK learns complex
sequential patterns of amino acids and their hierarchical structures from twenty
million proteins to create the EC number predictions. Furthermore, ECPICK identifies
significant amino acids that contribute to the prediction in a given protein sequence
without multiple sequence alignment, which may match to known motif sites for trustworthy
prediction or potential motif sites. Our intensive assessment showed not only outstanding
enhancement of predictive performance on the largest databases of Uniprot, PDB, and KEGG,
but also a capability to discover new motif sites in microorganisms. ECPICK will be a
reliable EC number prediction tool to identify protein functions of an increasing number
of uncharacterized enzymes. We also provide <b>EnzymeX</b> — a web-based portal that offers 
comprehensive protein sequence data analysis using advanced deep learning tools and includes 
the newest pretrained ECPICK models.

</br>
<a href="http://enzymex.dataxlab.org">
  <img width="150" height="108" alt="Image" src="https://github.com/user-attachments/assets/d73bdae6-0319-4b43-a9a2-6960fdcaea8e" />
</a>

- **Website**: http://enzymex.dataxlab.org
- **ECPICK Models**: http://enzymex.egr.unlv.edu/models_datasets
- **Documentation**: https://readthedocs.org/projects/ecpick
- **Source code**: https://github.com/datax-lab/ECPICK
</br>

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
  
## References

Not available yet.
