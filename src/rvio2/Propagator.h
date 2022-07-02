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

#ifndef PROPAGATOR_H
#define PROPAGATOR_H

#include <vector>
#include <unordered_map>

#include <Eigen/Core>

#include <opencv2/core/core.hpp>

#include "InputBuffer.h"
#include "../util/numerics.h"


namespace RVIO2
{

class Propagator
{
public:

    Propagator(const cv::FileStorage& fsSettings);

    void propagate(const int nImageId, 
                   const std::vector<ImuData>& vImuData, 
                   Eigen::VectorXf& Localx, 
                   Eigen::MatrixXf& LocalFactor);

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:

    void CreateNewFactor(const std::vector<ImuData>& vImuData, 
                         const Eigen::VectorXf& Localx, 
                         Eigen::Matrix<float,16,1>& x, 
                         Eigen::Matrix<float,15,27>& H);

    void LocalQR(const int nImageId, 
                 const Eigen::Matrix<float,16,1>& x, 
                 const Eigen::Matrix<float,15,27>& H, 
                 Eigen::VectorXf& Localx, 
                 Eigen::MatrixXf& LocalFactor);

    inline void FlipToHead(Eigen::Ref<SqrMatrixType> Mat, const int dim)
    {
        int cols = Mat.cols();
        Eigen::MatrixXf tempM1 = Mat.rightCols(dim);
        Eigen::MatrixXf tempM2 = Mat.leftCols(cols-dim);
        Mat.leftCols(dim).swap(tempM1);
        Mat.rightCols(cols-dim).swap(tempM2);
    }

    inline Eigen::MatrixXf pseudoInverse(const Eigen::MatrixXf& Mat, float epsilon=std::numeric_limits<float>::epsilon())
    {
	Eigen::JacobiSVD<Eigen::MatrixXf> svd(Mat, Eigen::ComputeFullU|Eigen::ComputeFullV);
	float tolerance = epsilon*std::max(Mat.cols(), Mat.rows())*svd.singularValues().array().abs()(0);
	return svd.matrixV()*(svd.singularValues().array().abs()>tolerance).select(svd.singularValues().array().inverse(),0).matrix().asDiagonal()*svd.matrixU().adjoint();
    }

private:

    int mnLocalWindowSize;

    float mnImuRate;
    float mnGravity;
    float mnSmallAngle;

    float mnGyroNoiseSigma;
    float mnGyroRandomWalkSigma;
    float mnAccelNoiseSigma;
    float mnAccelRandomWalkSigma;

    Eigen::Matrix<float,12,12> mSigma;
};

} // namespace RVIO2

#endif
