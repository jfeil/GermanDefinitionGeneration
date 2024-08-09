from setuptools import setup

setup(
    name='LLM-training',
    version='0.0.1',
    py_modules=['training', 'evaluate'],
    install_requires=[
        'Click',
    ],
    entry_points={
        'console_scripts': [
            'training = training:cli',
            'evaluate = evaluation:cli',
            'analysis = analysis:evaluate',
            'batch-runner = batch_runner:main',
        ],
    },
)
