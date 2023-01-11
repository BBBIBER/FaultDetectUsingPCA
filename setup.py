import setuptools
from setuptools import setup

setup(name="FaultDetectUsingPCA",
      version="0.1",
      description="Fault Detection, Identification and Reconstruction Using PCA(Principal Component Analysis)",
      url="https://github.com/Kyuhan1230/FaultDetect",
      author="Kyu Han, Seok",
      author_email="asdm159@gmail.com",
      zip_safe=False,
      packages=setuptools.find_packages(),
      classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent"
      ],
      python_requires=">=3.7",
)
