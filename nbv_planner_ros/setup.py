import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'nbv_planner_ros'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml')),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='aayushagrawal',
    maintainer_email='aayushagrawal@todo.todo',
    description='ROS 2 wrapper for NBV trajectory optimization planner',
    license='MIT',
    entry_points={
        'console_scripts': [
            'nbv_planner_node = nbv_planner_ros.nbv_planner_node:main',
        ],
    },
)
