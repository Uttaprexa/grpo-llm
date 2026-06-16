from setuptools import setup, find_packages

setup(
    name="grpo-llm",
    version="0.1.0",
    author="Uttapreksha Patel",
    author_email="patel.utt@northeastern.edu",
    description="GRPO for LLM math reasoning — from-scratch PyTorch implementation with async Trio rollout workers",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Uttaprexa/grpo-llm",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.2.0",
        "transformers>=4.40.0",
        "datasets>=2.18.0",
        "trio>=0.25.0",
        "pyyaml>=6.0",
        "wandb>=0.16.0",
    ],
    extras_require={
        "dev": [
            "pytest>=8.0.0",
            "pytest-trio>=0.8.0",
            "pytest-cov>=5.0.0",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)