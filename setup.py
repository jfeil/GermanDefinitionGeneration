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
            'training = training:train',
            'evaluate = evaluation:evaluate',
        ],
    },
)