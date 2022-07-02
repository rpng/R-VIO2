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

#ifndef UPDATER_H
#define UPDATER_H

#include <deque>
#include <vector>
#include <utility>
#include <unordered_map>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <opencv2/core/core.hpp>

#include <ros/ros.h>
#include <visualization_msgs/Marker.h>

#include "Feature.h"
#include "Tracker.h"


namespace RVIO2
{

class Updater
{
public:

    Updater(const cv::FileStorage& fsSettings);

    void update(const int nImageId, 
                const std::unordered_map<int,Feature*>& mFeatures, 
                const std::vector<std::pair<int,cv::Point2f> >& vFeatMeasForExploration, 
                std::vector<std::pair<int,Type> >& vFeatInfoForInitSlam, 
                const std::vector<std::vector<cv::Point2f> >& vvFeatMeasForInitSlam, 
                std::vector<std::pair<int,Type> >& vFeatInfoForPoseOnly, 
                const std::vector<std::vector<cv::Point2f> >& vvFeatMeasForPoseOnly, 
                std::vector<int>& vActiveFeatureIDs, 
                const std::deque<Eigen::Vector3f>& qLocalw, 
                const std::deque<Eigen::Vector3f>& qLocalv, 
                Eigen::VectorXf& Localx, 
                Eigen::MatrixXf& LocalFactor);

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:

    int triangulate(const int nTrackLength, 
                    const std::vector<cv::Point2f>& vRevFeatMeas, 
                    const std::vector<Eigen::Matrix<float,7,1> >& vRevRelCamPoses,
                    float& phi, 
                    float& psi, 
                    float& rho);

    void composition(const int nImageId, 
                     const std::unordered_map<int,Feature*>& mFeatures, 
                     std::vector<int>& vActiveFeatureIDs, 
                     const int nDimOfWinx, 
                     const int nDimOfWinSR, 
                     Eigen::VectorXf& Localx, 
                     Eigen::MatrixXf& LocalFactor);

    void ComposeQR(const int nIdx, 
                   const int nDim, 
                   Eigen::MatrixXf& LocalFactor);

    void LocalQR(const Eigen::MatrixXf& H, 
                 const Eigen::VectorXf& r, 
                 Eigen::MatrixXf& LocalFactor);

    void ReorderQR(const std::vector<int>& vFeatureStatuses, 
                   std::vector<int>& vFeatureIDs, 
                   Eigen::MatrixXf& LocalFactor);

    void CreateNewFactor(const cv::Point2f& z, 
                         const Eigen::Vector3f& pfG, 
                         const Eigen::Vector3f& pfG_fej, 
                         const Eigen::Matrix<float,7,1>& xG, 
                         const Eigen::Matrix<float,7,1>& xk, 
                         const Eigen::Vector3f& wk, 
                         const Eigen::Vector3f& vk, 
                         Eigen::MatrixXf& Hf, 
                         Eigen::MatrixXf& HG, 
                         Eigen::MatrixXf& HP, 
                         Eigen::MatrixXf& Hk, 
                         Eigen::VectorXf& r);

    bool CreateNewFactor(std::pair<int,Type>& pFeatInfo, 
                         const std::vector<cv::Point2f>& vRevFeatMeas, 
                         const std::vector<Eigen::Matrix<float,7,1> >& vRevRelImuPoses, 
                         const std::vector<Eigen::Matrix<float,7,1> >& vRevRelCamPoses, 
                         const std::deque<Eigen::Vector3f>& qRevLocalw, 
                         const std::deque<Eigen::Vector3f>& qRevLocalv, 
                         Eigen::MatrixXf& Hf, 
                         Eigen::MatrixXf& HP, 
                         Eigen::MatrixXf& HW, 
                         Eigen::VectorXf& r, 
                         Eigen::Vector3f& xf);

    void GetRevRelPoses(const int type, 
                        const int nTrackLength, 
                        const Eigen::VectorXf& Winx, 
                        std::vector<Eigen::Matrix<float,7,1> >& vRevRelImuPoses, 
                        std::vector<Eigen::Matrix<float,7,1> >& vRevRelCamPoses);

    void InitLM(const float phi, 
                const float psi, 
                const float rho, 
                const int nTrackLength, 
                const std::vector<cv::Point2f>& vRevFeatMeas, 
                const std::vector<Eigen::Matrix<float,7,1> >& vRevRelCamPoses, 
                Eigen::Matrix3f& HTRinvH, 
                Eigen::Vector3f& HTRinve, 
                float& cost);

    inline float chi2(const Eigen::MatrixXf& H, 
                      const Eigen::VectorXf& r, 
                      const Eigen::Ref<Eigen::MatrixXf> U) const
    {
        Eigen::MatrixXf V = H*U;
        Eigen::MatrixXf S = V*(V.transpose());
        S.diagonal() += pow(mnImageNoiseSigma,2)*Eigen::VectorXf::Ones(H.rows());
        return r.dot(S.llt().solve(r));
    }

    inline void FlipToTail(Eigen::Ref<Eigen::MatrixXf> Mat, const int dim)
    {
        int cols = Mat.cols();
        Eigen::MatrixXf tempM1 = Mat.leftCols(dim);
        Eigen::MatrixXf tempM2 = Mat.rightCols(cols-dim);
        Mat.rightCols(dim).swap(tempM1);
        Mat.leftCols(cols-dim).swap(tempM2);
    }

    inline void FlipToHead(Eigen::Ref<Eigen::MatrixXf> Mat, const int dim)
    {
        int cols = Mat.cols();
        Eigen::MatrixXf tempM1 = Mat.rightCols(dim);
        Eigen::MatrixXf tempM2 = Mat.leftCols(cols-dim);
        Mat.leftCols(dim).swap(tempM1);
        Mat.rightCols(cols-dim).swap(tempM2);
    }

private:

    int mnLocalWindowSize;

    float mnImageNoiseSigma;
    float mnImageNoiseSigmaInv;

    Eigen::Matrix2f mSigmaInv;

    Eigen::Matrix3f mRci;
    Eigen::Vector3f mtci;
    Eigen::Matrix3f mRic;
    Eigen::Vector3f mtic;

    std::vector<int> mvNewActiveFeatureIDs;
    std::vector<int> mvLostActiveFeatureIDs;

    // Interact with rviz
    ros::NodeHandle mUpdaterNode;
    ros::Publisher mFeatPub;
    int mnPubRate;

    geometry_msgs::Vector3 scaleLandmark;
    std_msgs::ColorRGBA colorLandmark_po;
    std_msgs::ColorRGBA colorLandmark_is;
    std_msgs::ColorRGBA colorLandmark_ex;
};

} // namespace RVIO2

#endif
