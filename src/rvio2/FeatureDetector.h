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

#ifndef FEATURE_DETECTOR_H
#define FEATURE_DETECTOR_H

#include <vector>

#include <opencv2/core/core.hpp>


namespace RVIO2
{

class FeatureDetector
{
public:

    FeatureDetector(const cv::FileStorage& fsSettings);

    int DetectWithSubPix(const cv::Mat& im, const int nCorners, const int s, std::vector<cv::Point2f>& vCorners);

    int FindNewer(const std::vector<cv::Point2f>& vCorners, const std::vector<cv::Point2f>& vRefCorners, std::vector<cv::Point2f>& vNewCorners);

private:

    void ChessGrid(const std::vector<cv::Point2f>& vCorners);

private:

    int mnImageCols;
    int mnImageRows;

    int mnBlocks;
    int mnGridCols;
    int mnGridRows;

    int mnOffsetX;
    int mnOffsetY;

    float mnBlockSizeX;
    float mnBlockSizeY;

    float mnMinDistance;
    float mnQualityLevel;

    int mnMaxFeatsPerBlock;

    // Chess grid of features
    std::vector<std::vector<cv::Point2f> > mvvGrid;
};

} // namespace RVIO2

#endif
