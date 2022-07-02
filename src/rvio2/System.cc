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

#include <fstream>

#include <Eigen/Core>

#include <opencv2/core/eigen.hpp>

#include <ros/package.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TransformStamped.h>

#include "System.h"
#include "../util/numerics.h"


namespace RVIO2
{

int nImageId = 0;

nav_msgs::Path path;

std::ofstream fPoseResults;


System::System(const std::string& strSettingsFile)
{
    // Read settings file
    cv::FileStorage fsSettings(strSettingsFile, cv::FileStorage::READ);
    if (!fsSettings.isOpened())
    {
       ROS_ERROR("Failed to open settings file at: %s", strSettingsFile.c_str());
       exit(-1);
    }

    const int bRecordOutputs = fsSettings["INI.RecordOutputs"];
    mbRecordOutputs = bRecordOutputs;

    const int bEnableAlignment = fsSettings["INI.EnableAlignment"];
    mbEnableAlignment = bEnableAlignment;

    const int bUseGroundTruthCalib = fsSettings["INI.UseGroundTruthCalib"];
    mbUseGroundTruthCalib = bUseGroundTruthCalib;

    const int nMaxTrackingLength = fsSettings["Tracker.nMaxTrackingLength"];
    mnLocalWindowSize = nMaxTrackingLength-1;

    mnMaxSlamPoints = fsSettings["Tracker.nMaxSlamPoints"];

    mnAngleThrd = fsSettings["INI.nAngleThrd"];
    mnLengthThrd = fsSettings["INI.nLengthThrd"];

    mnGravity = fsSettings["IMU.nG"];
    mnImuRate = fsSettings["IMU.dps"];

    msigmaGyroNoise = fsSettings["IMU.sigma_g"];
    msigmaGyroBias = fsSettings["IMU.sigma_wg"];
    msigmaAccelNoise = fsSettings["IMU.sigma_a"];
    msigmaAccelBias = fsSettings["IMU.sigma_wa"];

    mnCamRate = fsSettings["Camera.fps"];

    if (mbUseGroundTruthCalib)
        mnCamTimeOffset = fsSettings["Camera.nTimeOffset_GT"];
    else
        mnCamTimeOffset = fsSettings["Camera.nTimeOffset"];

    cv::Mat T(4,4,CV_32F);
    if (mbUseGroundTruthCalib)
        fsSettings["Camera.T_BC0_GT"] >> T;
    else
        fsSettings["Camera.T_BC0"] >> T;

    Eigen::Matrix4f Tic;
    cv::cv2eigen(T,Tic);
    Eigen::Matrix3f Ric = Tic.topLeftCorner(3,3);
    Eigen::Vector3f tic = Tic.topRightCorner(3,1);
    mRci = Ric.transpose();
    mtci = -mRci*tic;

    mbIsInitialized = false;

    mpInputBuffer = new InputBuffer(fsSettings);
    mpPropagator = new Propagator(fsSettings);
    mpUpdater = new Updater(fsSettings);
    mpTracker = new Tracker(fsSettings);

    mPathPub = mSystemNode.advertise<nav_msgs::Path>("/rvio2/trajectory", 1);

    if (mbRecordOutputs)
    {
        std::string pkg_path = ros::package::getPath("rvio2");
        fPoseResults.open(pkg_path+"/stamped_traj_estimate.txt", std::ofstream::out);
    }
}


System::~System()
{
    delete mpInputBuffer;
    delete mpPropagator;
    delete mpUpdater;
    delete mpTracker;
}


bool System::initialize(const ImageData& Image, const std::vector<ImuData>& vImuData)
{
    static int nImuCount = 0;

    static Eigen::Vector3f wm = Eigen::Vector3f::Zero();
    static Eigen::Vector3f am = Eigen::Vector3f::Zero();
    static double Dt = 0;

    static Eigen::Vector3f wm_last;

    static cv::Mat im_last;
    static double im_last_timestamp;

    Eigen::Vector3f ang = Eigen::Vector3f::Zero();
    Eigen::Vector3f vel = Eigen::Vector3f::Zero();
    Eigen::Vector3f len = Eigen::Vector3f::Zero();

    for (const ImuData& data : vImuData)
    {
        Eigen::Vector3f tempw = data.AngularVel;
        Eigen::Vector3f tempa = data.LinearAccel;
        double dt = data.TimeInterval;

        tempa -= mnGravity*Eigen::Vector3f(tempa/tempa.norm());

        ang += dt*tempw;
        vel += dt*tempa;
        len += dt*vel+.5*pow(dt,2)*tempa;
    }

    // Not move yet
    if (ang.norm()*180./M_PI<mnAngleThrd && len.norm()<mnLengthThrd)
    {
        for (const ImuData& data : vImuData)
        {
            wm += data.AngularVel;
            am += data.LinearAccel;
            Dt += data.TimeInterval;

            nImuCount++;
        }

        wm_last = vImuData.front().AngularVel;

        im_last = Image.Image;
        im_last_timestamp = Image.Timestamp;

        return false;
    }

    if (nImuCount==0)
    {
        // Start in motion
        wm = vImuData.back().AngularVel;
        am = vImuData.back().LinearAccel;
        nImuCount = 1;

        im_last = Image.Image;
        im_last_timestamp = Image.Timestamp;

        return false;
    }

    Eigen::Vector3f g, bg, ba;
    bg.setZero();
    ba.setZero();

    if (nImuCount==1)
    {
        g = am;
        g.normalize();
    }
    else
    {
        wm /= nImuCount;
        am /= nImuCount;

        g = am;
        g.normalize();

        bg = wm;
        ba = am-mnGravity*g;
    }

    Localx.setZero(27);

    if (mbEnableAlignment)
    {
        Eigen::Vector3f zv = g;

        Eigen::Vector3f ex = Eigen::Vector3f(1,0,0);
        Eigen::Vector3f xv = ex-zv*zv.transpose()*ex;
        xv.normalize();

        Eigen::Vector3f yv = SkewSymm(zv)*xv;
        yv.normalize();

        Eigen::Matrix3f R;
        R << xv, yv, zv;

        Localx.head(4) = RotToQuat(R);
    }
    else
        Localx.head(4) << 0, 0, 0, 1;

    Localx.segment(7,3) = g;
    Localx.segment(10,4) = RotToQuat(mRci);
    Localx.segment(14,3) = mtci;
    Localx(17) = mnCamTimeOffset;
    Localx.tail(6) << bg, ba;

    LocalFactor.setZero(25,26);
    LocalFactor(0,0) = 1./1e-6; // qG
    LocalFactor(1,1) = 1./1e-6;
    LocalFactor(2,2) = 1./1e-6;
    LocalFactor(3,3) = 1./1e-6; // pG
    LocalFactor(4,4) = 1./1e-6;
    LocalFactor(5,5) = 1./1e-6;
    LocalFactor(6,6) = 1./sqrt(Dt)/msigmaAccelNoise; // g
    LocalFactor(7,7) = 1./sqrt(Dt)/msigmaAccelNoise;
    LocalFactor(8,8) = 1./sqrt(Dt)/msigmaAccelNoise;

    LocalFactor(9,9) = mbUseGroundTruthCalib ? 1./2e-2 : 1./2e-1; // qci
    LocalFactor(10,10) = mbUseGroundTruthCalib ? 1./2e-2 : 1./2e-1;
    LocalFactor(11,11) = mbUseGroundTruthCalib ? 1./2e-2 : 1./2e-1;
    LocalFactor(12,12) = mbUseGroundTruthCalib ? 1./1e-2 : 1./1e-1; // pci
    LocalFactor(13,13) = mbUseGroundTruthCalib ? 1./1e-2 : 1./1e-1;
    LocalFactor(14,14) = mbUseGroundTruthCalib ? 1./1e-2 : 1./1e-1;
    LocalFactor(15,15) = mbUseGroundTruthCalib ? 10*mnImuRate : mnCamRate; // td

    LocalFactor(16,16) = 1./1e-3; // v
    LocalFactor(17,17) = 1./1e-3;
    LocalFactor(18,18) = 1./1e-3;
    LocalFactor(19,19) = 1./sqrt(Dt)/msigmaGyroBias; // bg
    LocalFactor(20,20) = 1./sqrt(Dt)/msigmaGyroBias;
    LocalFactor(21,21) = 1./sqrt(Dt)/msigmaGyroBias;
    LocalFactor(22,22) = 1./sqrt(Dt)/msigmaAccelBias; // ba
    LocalFactor(23,23) = 1./sqrt(Dt)/msigmaAccelBias;
    LocalFactor(24,24) = 1./sqrt(Dt)/msigmaAccelBias;

    Eigen::Vector3f v = Localx.segment(18,3);
    Eigen::Vector3f w = wm_last-Localx.segment(21,3);

    mqLocalv.push_back(v);
    mqLocalw.push_back(w);

    mbIsInitialized = true;

    if (mbRecordOutputs)
    {
        Eigen::Vector4f qkG = Localx.segment(0,4);
        Eigen::Vector3f pGk = -QuatToRot(QuatInv(qkG))*Localx.segment(4,3);

        fPoseResults << std::setprecision(19) << im_last_timestamp << " "
                     << pGk(0) << " " << pGk(1) << " " << pGk(2) << " "
                     << qkG(0) << " " << qkG(1) << " " << qkG(2) << " " << qkG(3) << std::endl;
    }

    // Start tracker
    Eigen::Matrix3f RcG = mRci*QuatToRot(Localx.head(4));
    Eigen::Vector3f tcG = mRci*Localx.segment(4,3)+mtci;
    mpTracker->track(0, im_last, RcG, tcG, 0, mmFeatures);

    return true;
}


void System::run()
{
    std::pair<ImageData, std::vector<ImuData> > pMeasurements;
    if (!mpInputBuffer->GetMeasurements(mnCamTimeOffset, pMeasurements))
        return;

    if (!mbIsInitialized)
        if (!initialize(pMeasurements.first, pMeasurements.second))
            return;

    nImageId++;

    mpPropagator->propagate(nImageId, pMeasurements.second, Localx, LocalFactor);

    // Predict camera pose
    Eigen::VectorXf xk = Localx.tail(16);
    Eigen::Matrix3f Rk = QuatToRot(xk.head(4));
    Eigen::Vector3f tk = xk.segment(4,3);
    Eigen::Matrix3f RkG = Rk*QuatToRot(Localx.head(4));
    Eigen::Vector3f tkG = Rk*(Localx.segment(4,3)-tk);

    Eigen::Matrix3f RcG = mRci*RkG;
    Eigen::Vector3f tcG = mRci*tkG+mtci;

    int nMapPtsNeeded = mnMaxSlamPoints-mvActiveFeatureIDs.size();

    mpTracker->track(nImageId, pMeasurements.first.Image, RcG, tcG, nMapPtsNeeded, mmFeatures);

    // Save local velocities
    Eigen::Vector3f w = pMeasurements.second.back().AngularVel-Localx.tail(6).head(3);
    Eigen::Vector3f v = Localx.tail(9).head(3);

    mqLocalw.push_back(w);
    mqLocalv.push_back(v);
    if (nImageId>mnLocalWindowSize)
    {
        mqLocalw.pop_front();
        mqLocalv.pop_front();
    }

    mpUpdater->update(nImageId, mmFeatures, 
                      mpTracker->mvFeatMeasForExploration, 
                      mpTracker->mvFeatInfoForInitSlam, mpTracker->mvvFeatMeasForInitSlam, 
                      mpTracker->mvFeatInfoForPoseOnly, mpTracker->mvvFeatMeasForPoseOnly, 
                      mvActiveFeatureIDs, mqLocalw, mqLocalv, Localx, LocalFactor);

    Eigen::Vector4f qkG = Localx.head(4);
    Eigen::Vector3f pGk = -QuatToRot(QuatInv(qkG))*Localx.segment(4,3);

    ROS_INFO("q_kG: %.6f %.6f %.6f %.6f", qkG(0), qkG(1), qkG(2), qkG(3));
    ROS_INFO("p_Gk: %.6f %.6f %.6f", pGk(0), pGk(1), pGk(2));

    if (mbRecordOutputs)
    {
        fPoseResults << std::setprecision(19) << pMeasurements.first.Timestamp << " "
                     << pGk(0) << " " << pGk(1) << " " << pGk(2) << " "
                     << qkG(0) << " " << qkG(1) << " " << qkG(2) << " " << qkG(3) << std::endl;
    }

    mRci = QuatToRot(Localx.segment(10,4));
    mtci = Localx.segment(14,3);
    mnCamTimeOffset = Localx(17);

    ROS_INFO("T_CI: %.6f %.6f %.6f %.6f %.6f %.6f %.6f", Localx(14), Localx(15), Localx(16), 
                                                         Localx(10), Localx(11), Localx(12), Localx(13));
    ROS_INFO("td: %.6f\n", mnCamTimeOffset);

    // Broadcast tf
    geometry_msgs::TransformStamped transformStamped;
    transformStamped.header.stamp = ros::Time::now();
    transformStamped.header.frame_id = "world";
    transformStamped.child_frame_id = "imu";
    transformStamped.transform.translation.x = pGk(0);
    transformStamped.transform.translation.y = pGk(1);
    transformStamped.transform.translation.z = pGk(2);
    transformStamped.transform.rotation.x = qkG(0);
    transformStamped.transform.rotation.y = qkG(1);
    transformStamped.transform.rotation.z = qkG(2);
    transformStamped.transform.rotation.w = qkG(3);
    mTfPub.sendTransform(transformStamped);

    // Visualize the trajectory
    geometry_msgs::PoseStamped pose;
    pose.header.frame_id = "world";
    pose.pose.position.x = pGk(0);
    pose.pose.position.y = pGk(1);
    pose.pose.position.z = pGk(2);
    pose.pose.orientation.x = qkG(0);
    pose.pose.orientation.y = qkG(1);
    pose.pose.orientation.z = qkG(2);
    pose.pose.orientation.w = qkG(3);

    path.header.frame_id = "world";
    path.poses.push_back(pose);
    mPathPub.publish(path);

    usleep(1000);
}

} //namespace RVIO2
