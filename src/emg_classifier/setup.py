import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'emg_classifier'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name]
        ),
        (
            'share/' + package_name,
            ['package.xml']
        ),
        (
            os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')
        ),
        (
            os.path.join('share', package_name, 'models'),
            glob('models/*')
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='student',
    maintainer_email='student@example.com',
    description='EMG gesture classification and BiDexHand control node',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'emg_classifier_node = emg_classifier.emg_classifier_node:main',
            'emg_replay_publisher = emg_classifier.emg_replay_publisher:main',
            'emg_prediction_monitor = emg_classifier.emg_prediction_monitor:main',
        ],
    },
)
