import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='pascal_voc_tools',
    version='0.1.1',
    author='ternence wang',
    author_email='ternencewang2015@outlook.com',
    description='some tools about pascal voc format dataset',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/wang-tf/pascal_voc_tools',
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    install_requires=['jinja2', 'opencv-python', 'tqdm'],
)
