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

#ifndef SYSTEM_H
#define SYSTEM_H

#include <deque>
#include <vector>
#include <string>
#include <utility>
#include <unordered_map>

#include <Eigen/Core>
#include <Eigen/Sparse>

#include <opencv2/core/core.hpp>

#include <ros/ros.h>
#include <tf/transform_broadcaster.h>

#include "Tracker.h"
#include "Updater.h"
#include "Propagator.h"
#include "Feature.h"
#include "InputBuffer.h"


namespace RVIO2
{

class System
{
public:

    System(const std::string& strSettingsFile);

    ~System();

    void run();

    void PushImuData(ImuData* pData) {mpInputBuffer->PushImuData(pData);}
    void PushImageData(ImageData* pData) {mpInputBuffer->PushImageData(pData);}

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:

    bool initialize(const ImageData& Image, const std::vector<ImuData>& vImuData);

protected:

    // State vector
    Eigen::VectorXf Localx;

    // Information factor
    Eigen::MatrixXf LocalFactor;

    std::deque<Eigen::Vector3f> mqLocalw;
    std::deque<Eigen::Vector3f> mqLocalv;

    std::unordered_map<int,Feature*> mmFeatures;
    std::vector<int> mvActiveFeatureIDs;

private:

    bool mbRecordOutputs;

    bool mbEnableAlignment;
    bool mbUseGroundTruthCalib;

    bool mbIsInitialized;

    int mnMaxSlamPoints;
    int mnLocalWindowSize;

    float mnGravity;
    float mnImuRate;

    float mnCamRate;
    float mnCamTimeOffset;

    float msigmaGyroNoise;
    float msigmaGyroBias;
    float msigmaAccelNoise;
    float msigmaAccelBias;

    float mnAngleThrd;
    float mnLengthThrd;

    Eigen::Matrix3f mRci;
    Eigen::Vector3f mtci;

    InputBuffer* mpInputBuffer;
    Propagator* mpPropagator;
    Updater* mpUpdater;
    Tracker* mpTracker;

    // Interact with rviz
    ros::NodeHandle mSystemNode;
    ros::Publisher mMapPub;
    ros::Publisher mPathPub;
    tf::TransformBroadcaster mTfPub;
};

} // namespace RVIO2

#endif
