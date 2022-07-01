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

#include "InputBuffer.h"
#include "../util/numerics.h"


namespace RVIO2
{

InputBuffer::InputBuffer(const cv::FileStorage& fsSettings)
{
    mnImuRate = fsSettings["IMU.dps"];
    mnCamRate = fsSettings["Camera.fps"];
}


void InputBuffer::PushImuData(ImuData* pData)
{
    std::unique_lock<std::mutex> lock(mMutex);

    mlImuFIFO.push_back(pData);

    if (!mlImuFIFO.empty())
        mlImuFIFO.sort([](const ImuData* a, const ImuData* b){return a->Timestamp<b->Timestamp;});
}


void InputBuffer::PushImageData(ImageData* pData)
{
    std::unique_lock<std::mutex> lock(mMutex);

    mlImageFIFO.push_back(pData);

    if (!mlImageFIFO.empty())
        mlImageFIFO.sort([](const ImageData* a, const ImageData* b){return a->Timestamp<b->Timestamp;});
}


bool InputBuffer::GetMeasurements(double nTimeOffset, std::pair<ImageData, std::vector<ImuData> >& pMeasurements)
{
    std::unique_lock<std::mutex> lock(mMutex);

    if (mlImuFIFO.empty() || mlImageFIFO.empty())
        return false;

    double timestamp = mlImageFIFO.front()->Timestamp+nTimeOffset;

    // No IMU measurements to process
    if (mlImuFIFO.front()->Timestamp>=timestamp)
    {
        delete mlImageFIFO.front();
        mlImageFIFO.pop_front();
        return false;
    }

    // Not enough IMU measurements to process
    if (mlImuFIFO.back()->Timestamp<timestamp)
        return false;

    // Still not enough IMU measurements to process
    if (mlImuFIFO.size()<mnImuRate/mnCamRate)
    {
        delete mlImageFIFO.front();
        mlImageFIFO.pop_front();
        return false;
    }

    ImageData Image = *mlImageFIFO.front();
    delete mlImageFIFO.front();
    mlImageFIFO.pop_front();

    std::vector<ImuData> vImus;
    while (!mlImuFIFO.empty() && mlImuFIFO.front()->Timestamp<=timestamp)
    {
        vImus.push_back(*mlImuFIFO.front());
        delete mlImuFIFO.front();
        mlImuFIFO.pop_front();
    }

    if (vImus.back().Timestamp<timestamp && !mlImuFIFO.empty())
    {
        Eigen::Vector3f w1 = vImus.back().AngularVel;
        Eigen::Vector3f a1 = vImus.back().LinearAccel;
        double t1 = vImus.back().Timestamp;

        Eigen::Vector3f w2 = mlImuFIFO.front()->AngularVel;
        Eigen::Vector3f a2 = mlImuFIFO.front()->LinearAccel;
        double t2 = mlImuFIFO.front()->Timestamp;

        double c1 = (t2-timestamp)/(t2-t1);
        double c2 = 1.-c1;

        ImuData data;
        data.AngularVel = c1*w1+c2*w2;
        data.LinearAccel = c1*a1+c2*a2;
        data.TimeInterval = timestamp-t1;
        data.Timestamp = timestamp;
        vImus.push_back(data);

        mlImuFIFO.front()->TimeInterval = t2-timestamp;
    }

    pMeasurements = {Image, vImus};

    return true;
}

} // namespace RVIO2
