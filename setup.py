from setuptools import find_packages, setup

package_name = 'pinky_llm'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name, ['pinky_llm/.env']),
        ('share/' + package_name, ['pinky_llm/object_detector.py', 'pinky_llm/nav2_bridge.py', 'pinky_llm/nav2_tools.py']),
        ('share/' + package_name + '/params',[
            'params/llm_params.yaml',
            'params/prompt_config.yaml',
            'params/house_points.yaml',
            ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='pinky',
    maintainer_email='pinky@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'object_detector = pinky_llm.object_detector:main',
            'agent_service = pinky_llm.agent_service:main',
            'agent_client = pinky_llm.agent_client:main',
        ],
    },
)
