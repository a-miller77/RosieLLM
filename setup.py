from setuptools import setup, find_packages

setup(
    name='rosiellm',
    version='0.1.0',
    author='Aiden Miller',
    author_email='milleraa@msoe.edu',
    description='A project for RosieLLM',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/a-miller77/RosieLLM',
    packages=find_packages(),
    install_requires=[
        'openai==0.10.2',
        'paramiko==2.7.2',
        'cryptography==3.4.7',
        'python-dotenv==0.19.2',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache License',
        'Operating System :: OS Independent',
    ],
    python_requires=">=3.9",
)