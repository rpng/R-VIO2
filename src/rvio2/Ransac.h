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

#ifndef RANSAC_H
#define RANSAC_H

#include <vector>
#include <utility>

#include <Eigen/Core>

#include <opencv2/core/core.hpp>


namespace RVIO2
{

class Ransac
{
public:

    Ransac(const cv::FileStorage& fsSettings);

    void FindInliers(const Eigen::MatrixXf& Points1, 
                     const Eigen::MatrixXf& Points2, 
                     const Eigen::Matrix3f& R, 
                     int& nInliers, 
                     std::vector<unsigned char>& vInlierFlags);

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:

    void PairTwoPoints(const int nInlierCandidates, 
                       const std::vector<int>& vInlierCandidateIndexes, 
                       std::vector<std::pair<int,int> >& vTwoPointIndexes);

    void ComputeE(const int nIterNum, 
                  const Eigen::MatrixXf& Points1, 
                  const Eigen::MatrixXf& Points2, 
                  const std::vector<std::pair<int,int> >& vTwoPointIndexes, 
                  const Eigen::Matrix3f& R, 
                  Eigen::Matrix3f& E);

    int CountVotes(const Eigen::MatrixXf& Points1, 
                   const Eigen::MatrixXf& Points2, 
                   const std::vector<int>& vInlierCandidateIndexes, 
                   const Eigen::Matrix3f& E);

    inline float SampsonError(const Eigen::Vector3f& pt1, const Eigen::Vector3f& pt2, const Eigen::Matrix3f& E)
    {
        Eigen::Vector3f Fx1 = E*pt1;
        Eigen::Vector3f Fx2 = E.transpose()*pt2;
        return (pow(pt2.dot(E*pt1),2))/(pow(Fx1(0),2)+pow(Fx1(1),2)+pow(Fx2(0),2)+pow(Fx2(1),2));
    }

    inline float AlgebraicError(const Eigen::Vector3f& pt1, const Eigen::Vector3f& pt2, const Eigen::Matrix3f& E)
    {
        return fabs(pt2.dot(E*pt1));
    }

private:

    int mnIterations;

    bool mbUseSampson;

    float mnSampsonThrd;
    float mnAlgebraicThrd;
};

} // namespace RVIO2

#endif
