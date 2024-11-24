from setuptools import setup, find_packages

setup(
    name='blueberry_detection_counting',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'opencv-python',  # Add other dependencies here if needed
    ],
    include_package_data=True,
    description='A package for blueberry detection and counting',
    author='Percy Cubas',
    author_email='pcubasm1@gmail.com',
    url='https://github.com/LABINM-Robotics-Automation/blueberry-detection-counting'
)

