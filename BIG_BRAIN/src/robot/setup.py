from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'robot'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'urdf'), glob('description/urdf/*.xacro')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
        (os.path.join('share', package_name, 'map'), glob('map/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nguyendang',
    maintainer_email='quangdangnguyen.2008@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'topic_bridge.py = robot.bridge.topic_bridge:main',
            'llm_bridge.py = robot.bridge.llm_bridge:main',
            'trajectory.py = robot.store.trajectory:main',
            'ring_bridge.py = robot.store.ring_bridge:main',
            'test_servo.py = robot.utilities.camera_servo:main',
            'person_lidar_tracker.py = robot.person_pose.person_lidar_tracker:main',
        ],
    },
)
