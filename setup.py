import pathlib

from setuptools import setup, find_packages

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="ecpick",
    version="0.0.1",
    description="Enzyme Commission Number Prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/datax-lab/ECPICK",
    author="MINGYU PARK@Sunmoon University",
    author_email="duveen@duveen.me",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: GPU",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    license_files=('LICENSE.md',),
    keywords="enzyme, ec number, prediction",
    packages=find_packages(where="."),
    include_package_data=True,
    install_requires=[
        "biopython==1.78",
        "numpy==1.23.2",
        "torch==1.12.1",
        "torchvision==0.13.1",
        "torchaudio==0.12.1",
        "scikit-learn==1.1.2",
        "tqdm==4.64.0"
    ],
    extras_require={
        "dev": [
            "pytest"
        ]
    },
    project_urls={
        "Bug Reports": "https://github.com/datax-lab/ECPICK/issue",
        "Source": "https://github.com/datax-lab/ECPICK"
    }
)