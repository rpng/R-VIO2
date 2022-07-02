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

#include "Ransac.h"


namespace RVIO2
{

Ransac::Ransac(const cv::FileStorage& fsSettings)
{
    int nIterations = fsSettings["Tracker.nRansacIter"];
    mnIterations = nIterations<16 ? 16 : nIterations;

    const int bUseSampson = fsSettings["Tracker.UseSampson"];
    mbUseSampson = bUseSampson;

    mnSampsonThrd = fsSettings["Tracker.nSampsonErrThrd"];
    mnAlgebraicThrd = fsSettings["Tracker.nAlgebraicErrThrd"];
}


void Ransac::PairTwoPoints(const int nInlierCandidates, 
                           const std::vector<int>& vInlierCandidateIndexes, 
                           std::vector<std::pair<int,int> >& vTwoPointIndexes)
{
    for (int i=0; i<mnIterations; ++i)
    {
        int idx1 = i;
        int idx2;
        do
            idx2 = rand()%nInlierCandidates;
        while (idx1==idx2);

        int idxA = vInlierCandidateIndexes.at(idx1);
        int idxB = vInlierCandidateIndexes.at(idx2);
        vTwoPointIndexes.emplace_back(idxA,idxB);
    }
}


void Ransac::ComputeE(const int nIterNum, 
                      const Eigen::MatrixXf& Points1, 
                      const Eigen::MatrixXf& Points2, 
                      const std::vector<std::pair<int,int> >& vTwoPointIndexes, 
                      const Eigen::Matrix3f& R, 
                      Eigen::Matrix3f& E)
{
    // point(i)(j): i=(A,B) in reference frame j=(1,2)
    Eigen::Vector3f pointA1 = Points1.col(vTwoPointIndexes.at(nIterNum).first);
    Eigen::Vector3f pointA2 = Points2.col(vTwoPointIndexes.at(nIterNum).first);
    Eigen::Vector3f pointB1 = Points1.col(vTwoPointIndexes.at(nIterNum).second);
    Eigen::Vector3f pointB2 = Points2.col(vTwoPointIndexes.at(nIterNum).second);

    // p0=R*p1
    Eigen::Vector3f pointA0 = R*pointA1;
    Eigen::Vector3f pointB0 = R*pointB1;

    // The solution of (p2^T)[tx]p0=0, where t is in terms of sinusoidals of
    // two directional angles: cos(alpha), sin(alpha), cos(beta) and sin(beta).
    // We need two correspondences (4 constraints) to solve t: {A0,A2}, {B0,B2}.
    float c1 = pointA2(0)*pointA0(1)-pointA0(0)*pointA2(1);
    float c2 = pointA0(1)*pointA2(2)-pointA2(1)*pointA0(2);
    float c3 = pointA2(0)*pointA0(2)-pointA0(0)*pointA2(2);
    float c4 = pointB2(0)*pointB0(1)-pointB0(0)*pointB2(1);
    float c5 = pointB0(1)*pointB2(2)-pointB2(1)*pointB0(2);
    float c6 = pointB2(0)*pointB0(2)-pointB0(0)*pointB2(2);

    float alpha = atan2(c3*c5-c2*c6, c1*c6-c3*c4);
    float beta = atan2(-c3, c1*sin(alpha)+c2*cos(alpha));

    float t0 = sin(beta)*cos(alpha);
    float t1 = cos(beta);
    float t2 = -sin(beta)*sin(alpha);

    Eigen::Matrix3f tx;
    tx <<  0, -t2,  t1,
          t2,   0, -t0,
         -t1,  t0,   0;

    E = tx*R;
}


int Ransac::CountVotes(const Eigen::MatrixXf& Points1, 
                       const Eigen::MatrixXf& Points2, 
                       const std::vector<int>& vInlierCandidateIndexes, 
                       const Eigen::Matrix3f& E)
{
    int nVotes = 0;
    for (const int &idx : vInlierCandidateIndexes)
    {
        if (mbUseSampson)
        {
            if (SampsonError(Points1.col(idx), Points2.col(idx), E)<mnSampsonThrd)
                nVotes++;
        }
        else
        {
            if (AlgebraicError(Points1.col(idx), Points2.col(idx), E)<mnAlgebraicThrd)
                nVotes++;
        }
    }

    return nVotes;
}


void Ransac::FindInliers(const Eigen::MatrixXf& Points1, 
                         const Eigen::MatrixXf& Points2, 
                         const Eigen::Matrix3f& R, 
                         int& nInliers, 
                         std::vector<unsigned char>& vInlierFlags)
{
    int nInlierCandidates = 0;
    std::vector<int> vInlierCandidateIndexes;
    vInlierCandidateIndexes.reserve((int)vInlierFlags.size());

    for (int i=0; i<(int)vInlierFlags.size(); ++i)
    {
        if (vInlierFlags.at(i))
        {
            vInlierCandidateIndexes.push_back(i);
            nInlierCandidates++;
        }
    }

    std::vector<std::pair<int,int> > vTwoPointIndexes;
    if (nInlierCandidates>mnIterations)
    {
        vTwoPointIndexes.reserve(nInlierCandidates);
        PairTwoPoints(nInlierCandidates, vInlierCandidateIndexes, vTwoPointIndexes);
    }
    else
    {
        // Not enough points to process
        std::fill(vInlierFlags.begin(), vInlierFlags.end(), 0);
        nInliers = 0;
        return;
    }

    int nWinnerVotes = 0;
    Eigen::Matrix3f WinnerE;
    for (int iter=0; iter<mnIterations; ++iter)
    {
        Eigen::Matrix3f E;
        ComputeE(iter, Points1, Points2, vTwoPointIndexes, R, E);
        int nVotes = CountVotes(Points1, Points2, vInlierCandidateIndexes, E);
        if (nVotes>nWinnerVotes)
        {
            WinnerE.swap(E);
            nWinnerVotes = nVotes;
        }
    }

    int nOutliers = 0;
    for (int i=0; i<nInlierCandidates; ++i)
    {
        int idx = vInlierCandidateIndexes.at(i);

        if (mbUseSampson)
        {
            float nError = SampsonError(Points1.col(idx), Points2.col(idx), WinnerE);
            if (nError>mnSampsonThrd || std::isinf(nError) || std::isnan(nError))
            {
                // Mark as outlier
                vInlierFlags.at(idx) = 0;
                nOutliers++;
            }
        }
        else
        {
            float nError = AlgebraicError(Points1.col(idx), Points2.col(idx), WinnerE);
            if (nError>mnAlgebraicThrd || std::isinf(nError) || std::isnan(nError))
            {
                // Mark as outlier
                vInlierFlags.at(idx) = 0;
                nOutliers++;
            }
        }
    }

    nInliers = nInlierCandidates-nOutliers;
}

} // namespace RVIO2
