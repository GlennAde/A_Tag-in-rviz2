import os
from glob import glob

from setuptools import find_packages, setup

package_name = "a_tag"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        # ament 资源索引，缺了它 ros2 run / get_package_share_directory 找不到本包
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        (os.path.join("share", package_name), ["package.xml"]),
        # config/ 与 launch/ 跟参考包一样在包内，安装到 share/<pkg>/ 下
        # config/*.yaml 是参数，*.rviz 是 RViz 配置（rviz2 -d 直接用）
        (os.path.join("share", package_name, "config"), glob("config/*.yaml") + glob("config/*.rviz")),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="GlennAde",
    maintainer_email="1519623572@qq.com",
    description="用共面四 AprilTag 标定板估计相机相对板的 6DoF 位姿",
    license="TODO: License declaration",
    entry_points={
        "console_scripts": [
            # 离线/直连采集的命令行
            "a-tag = a_tag.main:main",
            # ROS 2 节点
            "a_tag_node = a_tag.ros_node:main",
            # 链路自检（相机 -> 内参 -> 位姿 -> TF）
            "a-tag-check = a_tag.check_chain:main",
        ],
    },
)
