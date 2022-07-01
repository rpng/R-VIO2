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

#ifndef NODE_H
#define NODE_H

#include <vector>

#include <Eigen/Core>

#include <opencv2/core/core.hpp>


namespace RVIO2
{

class Feature
{
public:

    Feature(const int nFeatureId, const int nImageId);

    inline void Inited() {mbIsInited = true;}

    inline void Marginalized() {mbIsMarginalized = true;}

    inline void SetPosition(const Eigen::Vector3f& position) {mPosition = position;}

    inline void SetFejPosition(const Eigen::Vector3f& position) {mFejPosition = position;}

    inline const int FeatureId() const {return mnFeatureId;}

    inline const int RootImageId() const {return mnRootImageId;}

    inline const bool IsInited() const {return mbIsInited;}

    inline const bool IsMarginalized() const {return mbIsMarginalized;}

    inline Eigen::Vector3f& Position() {return mPosition;}

    inline Eigen::Vector3f& FejPosition() {return mFejPosition;}

    inline void reset(const int nImageId)
    {
        mnRootImageId = nImageId;
        mbIsInited = false;
        mbIsMarginalized = false;
    }

    inline void clear()
    {
        mnRootImageId = -1;
        mbIsInited = false;
        mbIsMarginalized = false;
    }

private:

    int mnFeatureId;   // start from 0
    int mnRootImageId; // start from 0

    bool mbIsInited;
    bool mbIsMarginalized;

    Eigen::Vector3f mPosition;
    Eigen::Vector3f mFejPosition;
};

} // namespace RVIO2

#endif
