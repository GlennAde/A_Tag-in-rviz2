# 相机

ros2 launch hik_camera hik_camera.launch.py

# tag包

ros2 launch a_tag a_tag.launch.py

# 三维可视化

rviz2 -d "$(ros2 pkg prefix a_tag)/share/a_tag/config/a_tag.rviz"
