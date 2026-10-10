"""a_tag 节点的 launch 文件。

用法::

    ros2 launch a_tag a_tag.launch.py
    ros2 launch a_tag a_tag.launch.py image_topic:=/hik_camera/image_raw \
                                      board_frame_id:=tag_board

提醒：节点是用**系统 python3.10** 跑的（colcon 生成的入口点写死了 /usr/bin/python3），
所以 pupil-apriltags 必须装在那里面 —— 见 README「安装」。
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    params_file = os.path.join(
        get_package_share_directory("a_tag"), "config", "a_tag_params.yaml"
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(name="params_file", default_value=params_file),
            DeclareLaunchArgument(name="image_topic", default_value="/image_raw"),
            DeclareLaunchArgument(
                name="camera_info_topic", default_value="/camera_info"
            ),
            DeclareLaunchArgument(name="board_frame_id", default_value="tag_board"),
            DeclareLaunchArgument(name="use_sensor_data_qos", default_value="true"),
            DeclareLaunchArgument(name="image_queue_depth", default_value="1"),
            DeclareLaunchArgument(name="publish_debug_image", default_value="false"),
            DeclareLaunchArgument(name="debug_period", default_value="0.2"),
            DeclareLaunchArgument(name="debug_adaptive", default_value="true"),
            DeclareLaunchArgument(name="debug_scale", default_value="1.0"),
            DeclareLaunchArgument(name="publish_markers", default_value="false"),
            DeclareLaunchArgument(
                name="rviz",
                default_value="false",
                description="顺便起一个配好的 rviz2（Fixed Frame=相机，带 TF/板示意/叠加图显示）",
            ),
            DeclareLaunchArgument(name="publish_tf", default_value="true"),
            DeclareLaunchArgument(
                name="camera_frame_id", default_value="camera_optical_frame"
            ),
            Node(
                package="a_tag",
                executable="a_tag_node",
                name="a_tag",
                output="screen",
                emulate_tty=True,
                parameters=[
                    LaunchConfiguration("params_file"),
                    {
                        "image_topic": LaunchConfiguration("image_topic"),
                        "camera_info_topic": LaunchConfiguration("camera_info_topic"),
                        "board_frame_id": LaunchConfiguration("board_frame_id"),
                        "use_sensor_data_qos": LaunchConfiguration(
                            "use_sensor_data_qos"
                        ),
                        "image_queue_depth": LaunchConfiguration("image_queue_depth"),
                        "publish_debug_image": LaunchConfiguration(
                            "publish_debug_image"
                        ),
                        "debug_period": LaunchConfiguration("debug_period"),
                        "debug_adaptive": LaunchConfiguration("debug_adaptive"),
                        "debug_scale": LaunchConfiguration("debug_scale"),
                        "publish_markers": LaunchConfiguration("publish_markers"),
                        "publish_tf": LaunchConfiguration("publish_tf"),
                        "camera_frame_id": LaunchConfiguration("camera_frame_id"),
                    },
                ],
            ),
            # 可选：顺便起一个配好的 RViz（Fixed Frame=相机 + TF + 板示意 + 叠加图）。
            # 用 share 里的绝对路径，所以不用担心"忘了 source 工作区"。
            # 需要装了 rviz2（ros-humble-rviz2）。
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                output="screen",
                arguments=[
                    "-d",
                    os.path.join(
                        get_package_share_directory("a_tag"), "config", "a_tag.rviz"
                    ),
                ],
                condition=IfCondition(LaunchConfiguration("rviz")),
            ),
        ]
    )
