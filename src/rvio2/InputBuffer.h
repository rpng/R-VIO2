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

#ifndef INPUTBUFFER_H
#define INPUTBUFFER_H

#include <list>
#include <mutex>

#include <Eigen/Core>

#include <opencv2/core/core.hpp>


namespace RVIO2
{

struct ImuData
{
    Eigen::Vector3f AngularVel;
    Eigen::Vector3f LinearAccel;
    double TimeInterval;
    double Timestamp;

    ImuData()
    {
        AngularVel.setZero();
        LinearAccel.setZero();
        TimeInterval = 0.;
        Timestamp = 0.;
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct ImageData
{
    cv::Mat Image;
    double Timestamp;

    ImageData()
    {
        Image = cv::Mat();
        Timestamp = 0.;
    }

    ~ImageData()
    {
        Image.release();
    }
};


class InputBuffer
{
public:

    InputBuffer(const cv::FileStorage& fsSettings);

    void PushImuData(ImuData* pData);
    void PushImageData(ImageData* pData);

    bool GetMeasurements(double nTimeOffset, std::pair<ImageData, std::vector<ImuData> >& pMeasurements);

private:

    float mnImuRate;
    float mnCamRate;

    std::list<ImuData*> mlImuFIFO;
    std::list<ImageData*> mlImageFIFO;

    std::mutex mMutex;
};

} // namespace RVIO2

#endif
