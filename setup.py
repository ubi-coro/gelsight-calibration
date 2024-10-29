from setuptools import setup, find_packages

setup(
    name="gs_sdk",
    version="0.1.0",
    description="SDK for GelSight sensors usage, reconstruction, and calibration.",
    author="Hung-Jui Huang, Ruihan Gao",
    author_email="hungjuih@andrew.cmu.edu, ruihang@andrew.cmu.edu",
    packages=find_packages(),
    install_requires=[
        "pillow==10.0.0",
        "numpy==1.26.4",
        "opencv-python>=4.9.0",
        "scipy>=1.13.1",
        "torch>=2.1.0",
        "PyYaml>=6.0.1",
        "matplotlib>=3.9.0",
        "ffmpeg-python",
        "nanogui"
    ],
    python_requires=">=3.9",
    entry_points={
        'console_scripts': [
            'collect_data=calibration.collect_data:collect_data',
            'label_data=calibration.label_data:label_data',
            'prepare_data=calibration.prepare_data:prepare_data',
            'train_model=calibration.train_model:train_model',
            'test_model=calibration.test_model:test_model',
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
)
