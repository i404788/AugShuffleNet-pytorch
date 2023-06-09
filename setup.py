from setuptools import setup, find_packages

setup(
  name = 'augshufflenet-pytorch',
  packages = find_packages(exclude=[]),
  version = '0.0.1',
  license='MIT',
  description = 'AugShuffleNet: Communicate More, Compute Less - Pytorch',
  author = 'Ferris Kwaijtaal',
  author_email = 'ferris+gh@devdroplets.com',
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/i404788/augshufflenet-pytorch',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'shufflenet',
    'convnet',
    'computer vision'
  ],
  install_requires=[
    'einops>=0.6',
    'torch>=1.13',
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.10',
  ],
)

