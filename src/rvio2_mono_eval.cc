/**
* This file is part of R-VIO2.
*
* Copyright (C) 2022 Zheng Huai <zhuai@udel.edu> and Guoquan Huang <ghuang@udel.edu>
* For more information see <http://github.com/rpng/R-VIO2> 
*
* R-VIO2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* R-VIO2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with R-VIO2. If not, see <http://www.gnu.org/licenses/>.
*/

#include <string>

#include <opencv2/core/core.hpp>

#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>

#include <sensor_msgs/Imu.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>

#include "rvio2/System.h"


int main(int argc, char **argv)
{
    ros::init(argc, argv, "rvio2_mono_eval");

    ros::start();

    if(argc!=3)
    {
        ROS_ERROR("Usage: roslaunch rvio2 launch_file[path_to_settings path_to_rosbag]");
        ros::shutdown();
        return -1;
    }    

    RVIO2::System Sys(argv[1]);

    cv::FileStorage fsSettings(argv[1], cv::FileStorage::READ);

    std::string path_to_bag(argv[2]);

    rosbag::Bag bag;
    bag.open(path_to_bag, rosbag::bagmode::Read);

    // Load rosbag as a view
    rosbag::View view_full;
    rosbag::View view;

    // Play full bag if negative duration
    int bag_skip = fsSettings["INI.nTimeskip"];
    int bag_durr = -1;

    view_full.addQuery(bag);
    ros::Time time_init = view_full.getBeginTime()+ros::Duration(bag_skip);
    ros::Time time_finish = bag_durr<0 ? view_full.getEndTime() : time_init+ros::Duration(bag_durr);
    ROS_INFO("time start = %.6f", time_init.toSec());
    ROS_INFO("time end   = %.6f", time_finish.toSec());

    view.addQuery(bag, time_init, time_finish);
    if (view.size()==0)
    {
        ROS_ERROR("No messages were found to play!");
        ros::shutdown();
        return -1;
    }

    for (const rosbag::MessageInstance& m : view)
    {
        if (!ros::ok())
            break;

        // Handle IMU message
        sensor_msgs::ImuConstPtr imuMsg = m.instantiate<sensor_msgs::Imu>();
        if (imuMsg!=NULL && (m.getTopic()=="/imu0" || m.getTopic()=="/imu"))
        {
            static int lastseq = -1;
            if ((int)imuMsg->header.seq!=lastseq+1 && lastseq!=-1)
                ROS_ERROR("IMU message drop! curr seq: %d expected seq: %d.", imuMsg->header.seq, lastseq+1);
            lastseq = imuMsg->header.seq;

            Eigen::Vector3f angular_velocity(imuMsg->angular_velocity.x, imuMsg->angular_velocity.y, imuMsg->angular_velocity.z);
            Eigen::Vector3f linear_acceleration(imuMsg->linear_acceleration.x, imuMsg->linear_acceleration.y, imuMsg->linear_acceleration.z);

            double currtime = imuMsg->header.stamp.toSec();

            RVIO2::ImuData* pData = new RVIO2::ImuData();
            pData->AngularVel = angular_velocity;
            pData->LinearAccel = linear_acceleration;
            pData->Timestamp = currtime;

            static double lasttime = -1;
            if (lasttime!=-1)
                pData->TimeInterval = currtime-lasttime;
            else
                pData->TimeInterval = 0;
            lasttime = currtime;

            Sys.PushImuData(pData);
        }

        // Handle image message
        sensor_msgs::Image::ConstPtr imgMsg = m.instantiate<sensor_msgs::Image>();
        if (imgMsg!=NULL && (m.getTopic()=="/cam0/image_raw" || m.getTopic()=="/camera/image_raw"))
        {
            static int lastseq = -1;
            if ((int)imgMsg->header.seq!=lastseq+1 && lastseq!=-1)
                ROS_ERROR("Image message drop! curr seq: %d expected seq: %d.", imgMsg->header.seq, lastseq+1);
            lastseq = imgMsg->header.seq;

            cv_bridge::CvImageConstPtr cv_ptr;
            try
            {
                cv_ptr = cv_bridge::toCvShare(imgMsg, sensor_msgs::image_encodings::MONO8);
            }
            catch (cv_bridge::Exception &e)
            {
                ROS_ERROR("cv_bridge exception: %s", e.what());
                continue;
            }

            RVIO2::ImageData* pData = new RVIO2::ImageData();
            pData->Image = cv_ptr->image.clone();
            pData->Timestamp = cv_ptr->header.stamp.toSec();

            Sys.PushImageData(pData);

            Sys.run();
        }
    }

    ros::shutdown();

    return 0;
}
