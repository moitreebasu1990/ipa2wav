from setuptools import setup, find_namespace_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ipa2wav",
    version="0.1.0",
    author="Moitree Basu",
    author_email="sfurti.basu@gmail.com",
    description="A deep learning-based text-to-speech synthesis system using IPA symbols",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/moitreebasu1990/ipa2wav",
    package_dir={"": "src"},
    packages=find_namespace_packages(where="src", include=["*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "tensorflow==1.8.0",
        "librosa>=0.8.0",
        "tqdm>=4.50.0",
        "matplotlib>=3.3.0",
        "scipy>=1.5.0",
        "soundfile>=0.10.0",
        "tensorboard==1.8.0",  # Must match TensorFlow version
    ],
    extras_require={
        "dev": [
            "black",  # Code formatting
            "flake8",  # Code linting
            "isort",  # Import sorting
            "sphinx",  # Documentation
            "sphinx-rtd-theme",  # Documentation theme
        ],
        "test": [
            "pytest>=6.0.0",
            "pytest-cov",  # Test coverage
            "pesq>=0.0.3",  # Speech quality metric
            "pystoi>=0.3.3",  # Speech intelligibility metric
            "psutil>=5.8.0",  # System resource monitoring
        ]
    },
    entry_points={
        "console_scripts": [
            "ipa2wav-preprocess=data_processing.prepro:main",
            "ipa2wav-train=tts_model.train:main",
            "ipa2wav-synthesize=tts_model.synthesize:main",
        ],
    },
)
