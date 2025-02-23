from setuptools import setup, find_packages

setup(
   name="s3pipe",
   version="1.3.0",
   description="This is a pipeline for superfast spherical surface processing",
   author="Fenqiang Zhao",
   author_email="zhaofenqiang0221@gmail.com",
   packages=find_packages(),
   install_requires=[
       "numpy",
       "scipy"
   ]
)