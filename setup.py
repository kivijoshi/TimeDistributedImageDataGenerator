from distutils.core import setup
setup(
  name = 'TimeDistributedImageDataGenerator',
  packages = ['TimeDistributedImageDataGenerator'],
  version = '0.4',
  license='MIT',
  description = 'Extension of Keras ImageDataGenerator class for TimeDistributed layer support.',
  author = 'Kaustubh Joshi',
  author_email = 'kaustubh.kivijoshi@gmail.com',
  url = 'https://github.com/kivijoshi/TimeDistributedImageDataGenerator',
  download_url = 'https://github.com/kivijoshi/TimeDistributedImageDataGenerator/archive/v_04.tar.gz',
  keywords = ['Keras', 'TimeDistributed', 'Extension'],
  install_requires=[            
          'keras',
          'tensorflow',
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Topic :: Software Development',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)