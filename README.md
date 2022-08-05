# R-VIO2

R-VIO2 is a novel square root information-based robocentric visual-inertial navigation algorithm using a monocular camera and a single IMU for consistent 3D motion tracking. It is developed based on our [robocentric VIO model](https://journals.sagepub.com/doi/pdf/10.1177/0278364919853361), while different with our previous work [R-VIO](https://github.com/rpng/R-VIO), we have derived and used i) our square-root robocentric formulation and ii) QR-based update combined with back substitution to improve the numerical stability and computational efficiency of the estimator. Moreover, the spatiotemporal calibration is performed online to robustify the performance of estimator in the presence of unknown parameter errors. Especially, this implementation can run in two modes: VIO or SLAM, where the former does not estimate any map points during the navigation (our *RA-L2022* paper), while the latter estimates a small set of map points in favor of localization (the frontend developed for our *ICRA2021* paper).

![](rvio2.gif)

If you find this work relevant to or use it for your research, please consider citing the following papers:
- Zheng Huai and Guoquan Huang, **Square-Root Robocentric Visual-Inertial Odometry with Online Spatiotemporal Calibration**, *IEEE Robotics and Automation Letters (RA-L)*, 2022: [download](https://ieeexplore.ieee.org/document/9830847).
```
@article{huai2022square,
  title={Square-root robocentric visual-inertial odometry with online spatiotemporal calibration},
  author={Huai, Zheng and Huang, Guoquan},
  journal={IEEE Robotics and Automation Letters},
  volume={7},
  number={4},
  pages={9961--9968},
  year={2022},
  publisher={IEEE}
}
```
- Zheng Huai and Guoquan Huang, **Markov Parallel Tracking and Mapping for Probabilistic SLAM**, *IEEE International Conference on Robotics and Automation (ICRA)*, 2021: [download](https://ieeexplore.ieee.org/document/9561238).
```
@inproceedings{huai2021markov,
  title     = {Markov parallel tracking and mapping for probabilistic SLAM},
  author    = {Huai, Zheng and Huang, Guoquan},
  booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
  pages     = {11661--11667},
  year      = {2021}
}
```

## 1. Prerequisites
### ROS
Download and install instructions can be found at: http://wiki.ros.org/kinetic/Installation/Ubuntu.
### Eigen
Download and install instructions can be found at: http://eigen.tuxfamily.org. **Tested with v3.1.0**.
### OpenCV
Download and install instructions can be found at: http://opencv.org. **Tested with v3.3.1**.


## 2. Build and Run
First `git clone` the repository and `catkin_make` it. Especially, `rvio2_mono` is used to run with rosbag in real time, while `rvio2_mono_eval` is used for evaluation purpose which preloads the rosbag and reads it as a txt file. A config file and a launch file are required for running R-VIO2 (for example, `rvio2_euroc.yaml` and `euroc.launch` are for [EuRoC](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets) dataset). The default mode is VIO, while you can switch to SLAM mode by setting the maximum number of SLAM features to nonzero from the config file (see `rvio2_euroc.yaml`). To visualize the outputs, please use `rviz`.
#### Start ROS:
  ```
  Terminal 1: roscore
  ```
  ```
  Terminal 2: rviz (AND OPEN rvio2_rviz.rviz IN THE CONFIG FOLDER)
  ```
#### Run `rvio2_mono`:
  ```
  Terminal 3: rosbag play --pause V1_01_easy.bag (AND SKIP SOME DATA IF NEEDED)
  ```
  ```
  Terminal 4: roslaunch rvio2 euroc.launch
  ```
#### Run `rvio2_mono_eval`:
  ```
  Terminal 3: roslaunch rvio2 euroc_eval.launch (PRESET PATH_TO_ROSBAG IN euroc_eval.launch)
  ```
Note that this implementation currently requires the sensor platform to start from stationary. Therefore, when testing the `Machine Hall` sequences you should skip the wiggling phase at the beginning. In particular, if you would like to run `rvio2_mono_eval`, the rosbag data to be skipped can be set in the config file (see `rvio2_euroc.yaml`).

## 3. License
This code is released under [GNU General Public License v3 (GPL-3.0)](https://www.gnu.org/licenses/gpl-3.0.en.html).
