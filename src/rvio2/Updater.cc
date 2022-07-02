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

#include "Updater.h"
#include "../util/numerics.h"


namespace RVIO2
{

int cloud_id = 0;


Updater::Updater(const cv::FileStorage& fsSettings)
{
    const int nMaxTrackingLength = fsSettings["Tracker.nMaxTrackingLength"];
    mnLocalWindowSize = nMaxTrackingLength-1;

    const float nImageNoiseSigmaX = fsSettings["Camera.sigma_px"];
    const float nImageNoiseSigmaY = fsSettings["Camera.sigma_py"];
    mnImageNoiseSigma = std::max(nImageNoiseSigmaX, nImageNoiseSigmaY);
    mnImageNoiseSigmaInv = 1./mnImageNoiseSigma;

    mSigmaInv << pow(mnImageNoiseSigmaInv,2), 0,
                 0, pow(mnImageNoiseSigmaInv,2);

    mFeatPub = mUpdaterNode.advertise<visualization_msgs::Marker>("/rvio2/landmarks", 1);
    mnPubRate = fsSettings["Displayer.nLandmarkPubRate"];

    scaleLandmark.x = fsSettings["Displayer.nLandmarkScale"];
    scaleLandmark.y = fsSettings["Displayer.nLandmarkScale"];
    scaleLandmark.z = fsSettings["Displayer.nLandmarkScale"];

    colorLandmark_po.a = 1;
    colorLandmark_po.r = 1;
    colorLandmark_po.b = 0;
    colorLandmark_po.g = 0;

    colorLandmark_is.a = 1;
    colorLandmark_is.r = 0;
    colorLandmark_is.b = 0;
    colorLandmark_is.g = 1;

    colorLandmark_ex.a = 1;
    colorLandmark_ex.r = 0;
    colorLandmark_ex.b = 1;
    colorLandmark_ex.g = 0;
}


int Updater::triangulate(const int nTrackLength, 
                         const std::vector<cv::Point2f>& vRevFeatMeas, 
                         const std::vector<Eigen::Matrix<float,7,1> >& vRevRelCamPoses, 
                         float& phi, 
                         float& psi, 
                         float& rho)
{
    phi = atan2(vRevFeatMeas.front().y, sqrt(pow(vRevFeatMeas.front().x,2)+1));
    psi = atan2(vRevFeatMeas.front().x, 1);
    rho = 0.;
    if (fabs(phi)>.5*M_PI || fabs(psi)>.5*M_PI)
        return -1;

    int nIter = 0;
    int maxIter = 20;
    float lambda = 1e-2;

    Eigen::Matrix3f HTRinvH, new_HTRinvH;
    Eigen::Vector3f HTRinve, new_HTRinve;
    float cost, new_cost;

    InitLM(phi, psi, rho, nTrackLength, vRevFeatMeas, vRevRelCamPoses, 
           HTRinvH, HTRinve, cost);

    while (nIter<maxIter)
    {
        HTRinvH.diagonal() *= 1.+lambda;
        Eigen::Vector3f dpfinv = HTRinvH.llt().solve(HTRinve);

        float new_phi = phi+dpfinv(0);
        float new_psi = psi+dpfinv(1);
        float new_rho = rho+dpfinv(2);

        if (dpfinv.norm()<1e-6)
            break;

        InitLM(new_phi, new_psi, new_rho, nTrackLength, vRevFeatMeas, vRevRelCamPoses, 
               new_HTRinvH, new_HTRinve, new_cost);

        if (new_cost<cost)
        {
            phi = new_phi;
            psi = new_psi;
            rho = new_rho;

            if (fabs(cost-new_cost)<1e-6 || fabs(dpfinv(2))<1e-6)
                break;

            HTRinvH.swap(new_HTRinvH);
            HTRinve.swap(new_HTRinve);
            cost = new_cost;

            lambda /= 2;
        }
        else
        {
            if (fabs(cost-new_cost)<1e-6 || fabs(dpfinv(2))<1e-6)
                break;

            lambda *= 1.5;
        }

        nIter++;
    }

    if (fabs(phi)>.5*M_PI || fabs(psi)>.5*M_PI || std::isnan(rho) || std::isinf(rho) || rho<0)
        return -1;

    if (nIter==maxIter)
        return 0;

    return 1;
}


void Updater::InitLM(const float phi, 
                     const float psi, 
                     const float rho, 
                     const int nTrackLength, 
                     const std::vector<cv::Point2f>& vRevFeatMeas, 
                     const std::vector<Eigen::Matrix<float,7,1> >& vRevRelCamPoses, 
                     Eigen::Matrix3f& HTRinvH, 
                     Eigen::Vector3f& HTRinve, 
                     float& cost)
{
    Eigen::Vector3f epfinv;
    epfinv << cos(phi)*sin(psi), sin(phi), cos(phi)*cos(psi);

    Eigen::Matrix<float,3,2> Jang;
    Jang << -sin(phi)*sin(psi), cos(phi)*cos(psi),
             cos(phi), 0,
            -sin(phi)*cos(psi), -cos(phi)*sin(psi);

    HTRinvH.setZero();
    HTRinve.setZero();
    cost = 0;

    for (int i=0; i<nTrackLength; ++i)
    {
        cv::Point2f z = vRevFeatMeas.at(i);

        Eigen::Matrix3f Rc = QuatToRot(vRevRelCamPoses.at(i).head(4));
        Eigen::Vector3f tc = vRevRelCamPoses.at(i).tail(3);
        Eigen::Vector3f h = Rc*epfinv+rho*tc;

        Eigen::Matrix<float,2,3> Hproj;
        Hproj << 1./h(2), 0, -h(0)/pow(h(2),2),
                 0, 1./h(2), -h(1)/pow(h(2),2);

        Eigen::Matrix<float,2,3> H;
        H << Hproj*Rc*Jang, Hproj*tc;

        Eigen::Vector2f e;
        e << z.x-h(0)/h(2), z.y-h(1)/h(2);

        HTRinvH += H.transpose()*mSigmaInv*H;
        HTRinve += H.transpose()*mSigmaInv*e;
        cost += e.dot(mSigmaInv*e);
    }
}


void Updater::GetRevRelPoses(const int type, 
                             const int nTrackLength, 
                             const Eigen::VectorXf& Winx, 
                             std::vector<Eigen::Matrix<float,7,1> >& vRevRelImuPoses, 
                             std::vector<Eigen::Matrix<float,7,1> >& vRevRelCamPoses)
{
    int k = 0;
    if (type==POSE_ONLY)
        k = Winx.rows()/7-1;
    else
        k = mnLocalWindowSize;

    vRevRelImuPoses.resize(nTrackLength);
    vRevRelCamPoses.resize(nTrackLength);
    vRevRelImuPoses.at(0) << 0, 0, 0, 1, 0, 0, 0;
    vRevRelCamPoses.at(0) << 0, 0, 0, 1, 0, 0, 0;

    for (int i=1; i<nTrackLength; ++i)
    {
        Eigen::Matrix<float,7,1> x1 = vRevRelImuPoses.at(i-1);
        Eigen::Matrix<float,7,1> x2 = Winx.segment(7*(k-i),7);

        Eigen::Matrix3f R = QuatToRot(QuatInv(x2.head(4)))*QuatToRot(x1.head(4));
        Eigen::Vector3f t = x2.tail(3)+QuatToRot(QuatInv(x2.head(4)))*x1.tail(3);

        vRevRelImuPoses.at(i) << RotToQuat(R), t;
        vRevRelCamPoses.at(i) << RotToQuat(mRci*R*mRic), mRci*R*mtic+mRci*t+mtci;
    }
}


bool Updater::CreateNewFactor(std::pair<int,Type>& pFeatInfo, 
                              const std::vector<cv::Point2f>& vRevFeatMeas, 
                              const std::vector<Eigen::Matrix<float,7,1> >& vRevRelImuPoses, 
                              const std::vector<Eigen::Matrix<float,7,1> >& vRevRelCamPoses, 
                              const std::deque<Eigen::Vector3f>& qRevLocalw, 
                              const std::deque<Eigen::Vector3f>& qRevLocalv, 
                              Eigen::MatrixXf& Hf, 
                              Eigen::MatrixXf& HP, 
                              Eigen::MatrixXf& HW, 
                              Eigen::VectorXf& r, 
                              Eigen::Vector3f& xf)
{
    int type = pFeatInfo.second;
    int nTrackLength = vRevFeatMeas.size();

    float phi, psi, rho;
    int rval = triangulate(nTrackLength, vRevFeatMeas, vRevRelCamPoses, phi, psi, rho);
    if (rval==-1)
    {
        if (type!=POSE_ONLY)
            pFeatInfo.second = BAD;

        return false;
    }
    else if (rval==0)
    {
        if (type!=POSE_ONLY)
            pFeatInfo.second = UNUSED;

        return false;
    }
    else
    {
        if (type==INIT_SLAM)
        {
            float bl = vRevRelCamPoses.back().tail(3).norm();
            float thrd = 40*bl*rho;
            if (thrd<1.)
            {
                type = POSE_ONLY_M;
                pFeatInfo.second = POSE_ONLY_M;
            }
        }
    }

    xf << phi, psi, rho;

    Eigen::Vector3f epfinv;
    epfinv << cos(phi)*sin(psi), sin(phi), cos(phi)*cos(psi);

    Eigen::Matrix<float,3,2> Jang;
    Jang << -sin(phi)*sin(psi),  cos(phi)*cos(psi),
             cos(phi), 0,
            -sin(phi)*cos(psi), -cos(phi)*sin(psi);

    SqrMatrixType tempHf, tempHP, tempHW;
    Eigen::VectorXf tempr;

    tempr.setZero(2*nTrackLength);
    tempHf.setZero(2*nTrackLength,3);
    tempHP.setZero(2*nTrackLength,7);

    if (type!=POSE_ONLY)
        tempHW.setZero(2*nTrackLength,6*(nTrackLength-1));
    else
        tempHW.setZero(2*nTrackLength,6*(nTrackLength-1)+15);

    Eigen::Matrix<float,2,3> subHqci, subHtci;
    Eigen::Vector2f subHt;

    Eigen::Matrix3f I;
    I.setIdentity();

    for (int i=0; i<nTrackLength; ++i)
    {
        cv::Point2f z = vRevFeatMeas.at(i);

        Eigen::Matrix3f R = QuatToRot(vRevRelImuPoses.at(i).head(4));
        Eigen::Vector3f t = vRevRelImuPoses.at(i).tail(3);

        Eigen::Matrix3f Rc = QuatToRot(vRevRelCamPoses.at(i).head(4));
        Eigen::Vector3f tc = vRevRelCamPoses.at(i).tail(3);
        Eigen::Vector3f h = Rc*epfinv+rho*tc;

        Eigen::Matrix<float,2,3> Hproj;
        Hproj << 1./h(2), 0, -h(0)/pow(h(2),2),
                 0, 1./h(2), -h(1)/pow(h(2),2);

        tempr.segment(2*i,2) << z.x-h(0)/h(2), z.y-h(1)/h(2);

        tempHf.block(2*i,0,2,3) << Hproj*Rc*Jang, Hproj*tc;

        subHt.setZero();
        subHqci.setZero();
        subHtci.setZero();

        if (i==0)
        {
            Eigen::Vector3f wc, vc;

            if (type!=POSE_ONLY)
            {
                wc = mRci*qRevLocalw.at(0);
                vc = mRci*qRevLocalv.at(0);
            }
            else
            {
                wc = mRci*qRevLocalw.at(1);
                vc = mRci*qRevLocalv.at(1);
            }

            subHt += Hproj*(Rc*(wc.cross(epfinv))+(vc.dot(epfinv))*tc);
        }

        if (i>0)
        {
            Eigen::Matrix<float,2,3> HRR = Hproj*mRci*R;

            for (int j=0; j<i; ++j)
            {
                Eigen::Matrix3f R1T = QuatToRot(vRevRelImuPoses.at(j).head(4)).transpose();
                Eigen::Vector3f t1 = -R1T*vRevRelImuPoses.at(j).tail(3);
                Eigen::Matrix3f R2T = QuatToRot(vRevRelImuPoses.at(j+1).head(4)).transpose();

                Eigen::Matrix3f Vx = SkewSymm(mRic*epfinv+rho*mtic-rho*t1);

                subHt += HRR*(-Vx*R1T*qRevLocalw.at(j)+rho*R1T*qRevLocalv.at(j));

                tempHW.block(2*i,6*(nTrackLength-2-j),2,6) << -HRR*Vx*R1T, rho*HRR*R2T;
            }

            subHqci = Hproj*((SkewSymm(Rc*epfinv)-rho*SkewSymm(Rc*mtci))*(I-Rc)+rho*SkewSymm(mRci*t));
            subHtci = Hproj*rho*(I-Rc);
        }

        tempHP.block(2*i,0,2,7) << subHqci, subHtci, subHt;
    }

    int M = 2*nTrackLength;
    int N = 3;

    if (tempHf.col(2).norm()/M<1e-6)
    {
        N = 2;

        if (type==INIT_SLAM)
        {
            type = POSE_ONLY_M;
            pFeatInfo.second = POSE_ONLY_M;
        }
    }

    Eigen::JacobiRotation<float> GR;

    for (int n=0; n<N; ++n)
    {
        for (int m=M-1; m>n; --m)
        {
            GR.makeGivens(tempHf(m-1,n), tempHf(m,n));
            tempHf.applyOnTheLeft(m-1,m,GR.adjoint());
            tempHP.applyOnTheLeft(m-1,m,GR.adjoint());
            tempHW.applyOnTheLeft(m-1,m,GR.adjoint());
            tempr.applyOnTheLeft(m-1,m,GR.adjoint());
            tempHf(m,n) = 0;
        }
    }

    if (type==INIT_SLAM)
    {
        Hf = tempHf;
        HP = tempHP;
        HW = tempHW;
        r = tempr;
    }
    else
    {
        HP = tempHP.bottomRows(M-N);
        HW = tempHW.bottomRows(M-N);
        r = tempr.tail(M-N);
    }

    return true;
}


void Updater::CreateNewFactor(const cv::Point2f& z, 
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
                              Eigen::VectorXf& r)
{
    r.resize(2);
    Hf.resize(2,3);
    HG.resize(2,9);
    HP.resize(2,7);
    Hk.resize(2,15);

    Eigen::Matrix3f RG = QuatToRot(xG.head(4));
    Eigen::Vector3f pG = xG.tail(3);

    Eigen::Matrix3f Rk = QuatToRot(xk.head(4));
    Eigen::Vector3f tk = xk.tail(3);

    Eigen::Matrix3f RkG = Rk*RG;
    Eigen::Vector3f pkG = Rk*(pG-tk);
    Eigen::Vector3f pfk = RkG*pfG+pkG;
    Eigen::Vector3f pfk_fej = RkG*pfG_fej+pkG;

    Eigen::Vector3f h = mRci*pfk+mtci;
    Eigen::Vector3f h_fej = mRci*pfk_fej+mtci;

    Eigen::Matrix<float,2,3> Hproj;
    Hproj << 1./h_fej(2), 0, -h_fej(0)/pow(h_fej(2),2),
             0, 1./h_fej(2), -h_fej(1)/pow(h_fej(2),2);

    Eigen::Matrix<float,2,3> HR = Hproj*mRci;

    r << z.x-h(0)/h(2), z.y-h(1)/h(2);

    Hf = HR*RkG;
    HG << HR*Rk*SkewSymm(RG*pfG_fej), HR*Rk, Eigen::MatrixXf::Zero(2,3);
    HP << Hproj*SkewSymm(mRci*pfk_fej), Hproj, HR*SkewSymm(pfk_fej)*wk-HR*vk;
    Hk << HR*SkewSymm(pfk_fej), -HR*Rk, Eigen::MatrixXf::Zero(2,9);
}


void Updater::LocalQR(const Eigen::MatrixXf& H, 
                      const Eigen::VectorXf& r, 
                      Eigen::MatrixXf& LocalFactor)
{
    int M = H.rows();
    int N = H.cols();

    SqrMatrixType tempLocalFactor;
    tempLocalFactor.resize(N+M,N+1);
    tempLocalFactor << LocalFactor.bottomRightCorner(N,N+1), H, r;

    Eigen::JacobiRotation<float> GR;

    for (int n=0; n<N; ++n)
    {
        for (int m=N+M-1; m>=N; --m)
        {
            if (tempLocalFactor(m,n)!=0)
            {
                GR.makeGivens(tempLocalFactor(n,n),tempLocalFactor(m,n));
                tempLocalFactor.applyOnTheLeft(n,m,GR.adjoint());
                tempLocalFactor(m,n) = 0;
            }
        }
    }

    LocalFactor.bottomRightCorner(N,N+1) = tempLocalFactor.topRows(N);
}


void Updater::ReorderQR(const std::vector<int>& vFeatureStatuses, 
                        std::vector<int>& vFeatureIDs, 
                        Eigen::MatrixXf& LocalFactor)
{
    int L = LocalFactor.rows();
    int l1 = 0;
    int l2 = 3*vFeatureIDs.size();

    std::vector<int> vActiveIDs;

    for (int n=0; n<(int)vFeatureIDs.size(); ++n)
    {
        int id = vFeatureIDs.at(n);

        if (vFeatureStatuses.at(n)==0)
        {
            if (n>0)
            {
                int N = 3*(n+1);

                FlipToHead(LocalFactor.block(0,l1,l1+l2,N), 3);
            }

            mvLostActiveFeatureIDs.push_back(id);
        }
        else
            vActiveIDs.push_back(id);
    }

    vFeatureIDs.swap(vActiveIDs);

    SqrMatrixType tempLocalFactor;
    tempLocalFactor = LocalFactor.block(l1,l1,l2,L+1-l1);

    Eigen::JacobiRotation<float> GR;

    for (int n=0; n<l2; ++n)
    {
        for (int m=l2-1; m>n; --m)
        {
            if (tempLocalFactor(m,n)!=0)
            {
                GR.makeGivens(tempLocalFactor(m-1,n),tempLocalFactor(m,n));
                tempLocalFactor.applyOnTheLeft(m-1,m,GR.adjoint());
                tempLocalFactor(m,n) = 0;
            }
        }
    }

    LocalFactor.block(l1,l1,l2,L+1-l1).swap(tempLocalFactor);
}


void Updater::update(const int nImageId, 
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
                     Eigen::MatrixXf& LocalFactor)
{
    int nWinSize = nImageId>mnLocalWindowSize ? mnLocalWindowSize : nImageId;
    int nDimOfWinx = 7*nWinSize;
    int nDimOfWinSR = 6*nWinSize+9;

    if (vFeatMeasForExploration.empty() && vFeatInfoForInitSlam.empty() && vFeatInfoForPoseOnly.empty())
    {
        mvLostActiveFeatureIDs.swap(vActiveFeatureIDs);

        composition(nImageId, mFeatures, vActiveFeatureIDs, nDimOfWinx, nDimOfWinSR, Localx, LocalFactor);

        return;
    }

    mRci = QuatToRot(Localx.segment(10,4));
    mtci = Localx.segment(14,3);
    mRic = mRci.transpose();
    mtic = -mRic*mtci;

    Eigen::VectorXf Winx = Localx.segment(10+8,nDimOfWinx);

    Eigen::MatrixXf NavSRinv;
    NavSRinv.setIdentity(9+7+nDimOfWinSR,9+7+nDimOfWinSR);
    LocalFactor.bottomRightCorner(9+7+nDimOfWinSR,9+7+nDimOfWinSR+1).leftCols(9+7+nDimOfWinSR)
               .triangularView<Eigen::Upper>().solveInPlace(NavSRinv);

    if (!vFeatMeasForExploration.empty())
    {
        // ROS rviz settings
        visualization_msgs::Marker cloud_ex;
        cloud_ex.header.frame_id = "world";
        cloud_ex.ns = "EXPLO";
        cloud_ex.id = ++cloud_id;
        cloud_ex.color = colorLandmark_ex;
        cloud_ex.scale = scaleLandmark;
        cloud_ex.pose.orientation.w = 1.0;
        cloud_ex.lifetime = ros::Duration(1./mnPubRate*mnLocalWindowSize);
        cloud_ex.action = visualization_msgs::Marker::ADD;
        cloud_ex.type = visualization_msgs::Marker::POINTS;

        Eigen::MatrixXf H;
        Eigen::VectorXf r;
        int nNewRows = 0;

        Eigen::Vector3f wk = qLocalw.back();
        Eigen::Vector3f vk = qLocalv.back();

        Eigen::Matrix<float,7,1> xG = Localx.head(7);
        Eigen::Matrix<float,7,1> xk = Localx.tail(16).head(7);

        int nActiveFeatures = vActiveFeatureIDs.size();
        std::vector<int> vFeatureStatuses(nActiveFeatures,0);

        for (std::vector<std::pair<int,cv::Point2f> >::const_iterator vitMeas=vFeatMeasForExploration.begin(); 
             vitMeas!=vFeatMeasForExploration.end(); ++vitMeas)
        {
            int id = vitMeas->first;
            Feature* pFeature = mFeatures.at(id);

            // Get feature index
            auto vit = std::find_if(vActiveFeatureIDs.begin(), vActiveFeatureIDs.end(), [id](const int& val){return val==id;});
            int idx = vit-vActiveFeatureIDs.begin();
            vFeatureStatuses.at(idx) = 1;

            Eigen::Vector3f pfG = pFeature->Position();
            Eigen::Vector3f pfG_fej = pFeature->FejPosition();

            Eigen::MatrixXf tempHf, tempHG, tempHP, tempHk;
            Eigen::VectorXf tempr;
            CreateNewFactor((*vitMeas).second, pfG, pfG_fej, xG, xk, wk, vk, 
                            tempHf, tempHG, tempHP, tempHk, tempr);

            Eigen::MatrixXf tempH;
            tempH.setZero(2,9+7+nDimOfWinSR);
            tempH.leftCols(16) << tempHG, tempHP;
            tempH.rightCols(15).swap(tempHk);

            float val = chi2(tempH, tempr, NavSRinv);
            if (val<CHI2_THRESHOLD[2-1])
            {
                geometry_msgs::Point point;
                point.x = pfG(0);
                point.y = pfG(1);
                point.z = pfG(2);
                cloud_ex.points.push_back(point);
            }
            else
                continue;

            nNewRows += 2;

            H.conservativeResizeLike(Eigen::MatrixXf::Zero(nNewRows,3*nActiveFeatures+9+7+nDimOfWinSR));
            H.block(nNewRows-2,3*idx,2,3).swap(tempHf);
            H.bottomRightCorner(2,9+7+nDimOfWinSR).swap(tempH);

            r.conservativeResize(nNewRows);
            r.tail(2).swap(tempr);
        }

        if (nNewRows>0)
        {
            H *= mnImageNoiseSigmaInv;
            r *= mnImageNoiseSigmaInv;
            LocalQR(H, r, LocalFactor);

            mFeatPub.publish(cloud_ex);
        }

        int n = std::accumulate(vFeatureStatuses.begin(), vFeatureStatuses.end(), 0);
        if (n<nActiveFeatures)
        {
            if (n>0)
                ReorderQR(vFeatureStatuses, vActiveFeatureIDs, LocalFactor);
            else
                mvLostActiveFeatureIDs.swap(vActiveFeatureIDs);
        }
    }
    else
        mvLostActiveFeatureIDs.swap(vActiveFeatureIDs);

    if (!vFeatInfoForInitSlam.empty())
    {
        // ROS rviz settings
        visualization_msgs::Marker cloud_is;
        cloud_is.header.frame_id = "imu";
        cloud_is.ns = "Init_SLAM";
        cloud_is.id = ++cloud_id;
        cloud_is.color = colorLandmark_is;
        cloud_is.scale = scaleLandmark;
        cloud_is.pose.orientation.w = 1.0;
        cloud_is.lifetime = ros::Duration(1./mnPubRate);
        cloud_is.action = visualization_msgs::Marker::ADD;
        cloud_is.type = visualization_msgs::Marker::POINTS;

        Eigen::MatrixXf Hf;
        Eigen::VectorXf rf;
        int nNewPoints = 0;

        Eigen::MatrixXf Hx;
        Eigen::VectorXf rx;
        int nNewRows = 0;

        std::deque<Eigen::Vector3f> qRevLocalw(qLocalw.rbegin(), qLocalw.rend());
        std::deque<Eigen::Vector3f> qRevLocalv(qLocalv.rbegin(), qLocalv.rend());

        std::vector<Eigen::Matrix<float,7,1> > vRevRelImuPoses, vRevRelCamPoses;
        GetRevRelPoses(INIT_SLAM, mnLocalWindowSize+1, Winx, vRevRelImuPoses, vRevRelCamPoses);

        std::vector<std::pair<int,Type> >::iterator vitInfo = vFeatInfoForInitSlam.begin();
        std::vector<std::vector<cv::Point2f> >::const_iterator vitMeas = vvFeatMeasForInitSlam.begin();

        for (; vitInfo!=vFeatInfoForInitSlam.end(); ++vitInfo, ++vitMeas)
        {
            int id = (*vitInfo).first;
            Feature* pFeature = mFeatures.at(id);

            std::vector<cv::Point2f> vRevFeatMeas((*vitMeas).rbegin(), (*vitMeas).rend());

            Eigen::MatrixXf tempHf, tempHP, tempHW;
            Eigen::VectorXf tempr;
            Eigen::Vector3f xf;
            if (!CreateNewFactor(*vitInfo, vRevFeatMeas, vRevRelImuPoses, vRevRelCamPoses, qRevLocalw, qRevLocalv, 
                                 tempHf, tempHP, tempHW, tempr, xf))
                continue;

            int M = tempHW.rows();
            int N = tempHW.cols();

            int type = (*vitInfo).second;

            if (type==INIT_SLAM)
            {
                Eigen::MatrixXf tempHx;
                Eigen::VectorXf temprx;
                tempHx.setZero(M-3,7+nDimOfWinSR);
                tempHx.leftCols(7+N) << tempHP.bottomRows(M-3), tempHW.bottomRows(M-3);
                temprx = tempr.tail(M-3);

                float val = chi2(tempHx, temprx, NavSRinv.bottomRightCorner(7+nDimOfWinSR,7+nDimOfWinSR));
                if (val<CHI2_THRESHOLD[M-3-1])
                {
                    float phi = xf(0);
                    float psi = xf(1);
                    float rho = xf(2);

                    Eigen::Vector3f epfinv;
                    epfinv << cos(phi)*sin(psi), sin(phi), cos(phi)*cos(psi);
                    Eigen::Vector3f pf = mRic*(1./rho*epfinv)+mtic;

                    geometry_msgs::Point feat;
                    feat.x = pf(0);
                    feat.y = pf(1);
                    feat.z = pf(2);
                    cloud_is.points.push_back(feat);
                }
                else
                {
                    (*vitInfo).second = BAD;
                    continue;
                }

                nNewRows += M-3;
                Hx.conservativeResize(nNewRows,7+nDimOfWinSR);
                Hx.bottomRows(M-3).swap(tempHx);

                rx.conservativeResize(nNewRows);
                rx.tail(M-3).swap(temprx);

                nNewPoints++;
                Hf.conservativeResizeLike(Eigen::MatrixXf::Zero(3*nNewPoints,7+nDimOfWinSR+3*nNewPoints));
                Hf.bottomLeftCorner(3,7+N) << tempHP.topRows(3), tempHW.topRows(3);
                Hf.bottomRightCorner(3,3) = tempHf.topRows(3);

                rf.conservativeResize(3*nNewPoints);
                rf.tail(3) = tempr.head(3);

                pFeature->SetPosition(xf);
                pFeature->SetFejPosition(xf);

                mvNewActiveFeatureIDs.push_back(id);
            }
            else
            {
                Eigen::MatrixXf tempH;
                tempH.setZero(M,7+nDimOfWinSR);
                tempH.leftCols(7+N) << tempHP, tempHW;

                float val = chi2(tempH, tempr, NavSRinv.bottomRightCorner(7+nDimOfWinSR,7+nDimOfWinSR));
                if (val<CHI2_THRESHOLD[M-1])
                {
                    float phi = xf(0);
                    float psi = xf(1);
                    float rho = xf(2);

                    Eigen::Vector3f epfinv;
                    epfinv << cos(phi)*sin(psi), sin(phi), cos(phi)*cos(psi);
                    Eigen::Vector3f pf = mRic*(1./rho*epfinv)+mtic;

                    geometry_msgs::Point feat;
                    feat.x = pf(0);
                    feat.y = pf(1);
                    feat.z = pf(2);
                    cloud_is.points.push_back(feat);
                }
                else
                {
                    (*vitInfo).second = BAD;
                    continue;
                }

                nNewRows += M;

                Hx.conservativeResize(nNewRows,7+nDimOfWinSR);
                Hx.bottomRows(M).swap(tempH);

                rx.conservativeResize(nNewRows);
                rx.tail(M).swap(tempr);
            }
        }

        if (nNewRows>0)
        {
            Hx *= mnImageNoiseSigmaInv;
            rx *= mnImageNoiseSigmaInv;
            LocalQR(Hx, rx, LocalFactor);

            mFeatPub.publish(cloud_is);
        }

        if (nNewPoints>0)
        {
            Hf *= mnImageNoiseSigmaInv;
            rf *= mnImageNoiseSigmaInv;

            int L = LocalFactor.rows();
            int l1 = L-9-7-nDimOfWinSR;
            int l2 = 3*nNewPoints;
            int l3 = 9+7+nDimOfWinSR;

            Eigen::MatrixXf tempM;
            tempM.setZero(L+l2,L+l2+1);
            tempM.topLeftCorner(l1,l1) = LocalFactor.topLeftCorner(l1,l1);
            tempM.topRightCorner(l1,l3+1) = LocalFactor.topRightCorner(l1,l3+1);
            tempM.bottomRightCorner(l3,l3+1) = LocalFactor.bottomRightCorner(l3,l3+1);
            tempM.block(l1,l1,l2,l2) = Hf.rightCols(l2);
            tempM.block(l1,l1+l2+9,l2,7+nDimOfWinSR+1) << Hf.leftCols(7+nDimOfWinSR), rf;
            LocalFactor.swap(tempM);
        }
    }

    if (!vFeatInfoForPoseOnly.empty())
    {
        // ROS rviz settings
        visualization_msgs::Marker cloud_po;
        cloud_po.header.frame_id = "imu";
        cloud_po.ns = "Pose_Only";
        cloud_po.id = ++cloud_id;
        cloud_po.color = colorLandmark_po;
        cloud_po.scale = scaleLandmark;
        cloud_po.pose.orientation.w = 1.0;
        cloud_po.lifetime = ros::Duration(1./mnPubRate);
        cloud_po.action = visualization_msgs::Marker::ADD;
        cloud_po.type = visualization_msgs::Marker::POINTS;

        Eigen::MatrixXf H;
        Eigen::VectorXf r;
        int nNewRows = 0;

        std::vector<std::pair<int,Type> >::iterator vitInfo = vFeatInfoForPoseOnly.begin();
        std::vector<std::vector<cv::Point2f> >::const_iterator vitMeas = vvFeatMeasForPoseOnly.begin();

        for (; vitInfo!=vFeatInfoForPoseOnly.end(); ++vitInfo, ++vitMeas)
        {
            int id = (*vitInfo).first;
            Feature* pFeature = mFeatures.at(id);

            int type = (*vitInfo).second;

            int nTrackLength = (int)(*vitMeas).size();

            std::deque<Eigen::Vector3f> qRevLocalw(qLocalw.rbegin(), qLocalw.rend());
            std::deque<Eigen::Vector3f> qRevLocalv(qLocalv.rbegin(), qLocalv.rend());
            if (type==POSE_ONLY)
            {
                qRevLocalw.pop_front();
                qRevLocalv.pop_front();
            }

            std::vector<Eigen::Matrix<float,7,1> > vRevRelImuPoses, vRevRelCamPoses;
            GetRevRelPoses(type, nTrackLength, Winx, vRevRelImuPoses, vRevRelCamPoses);

            std::vector<cv::Point2f> vRevFeatMeas((*vitMeas).rbegin(), (*vitMeas).rend());

            Eigen::MatrixXf tempHf, tempHP, tempHW;
            Eigen::VectorXf tempr;
            Eigen::Vector3f xf;
            if (!CreateNewFactor(*vitInfo, vRevFeatMeas, vRevRelImuPoses, vRevRelCamPoses, qRevLocalw, qRevLocalv, 
                                 tempHf, tempHP, tempHW, tempr, xf))
                continue;

            int M = tempHW.rows();
            int N = tempHW.cols();

            Eigen::MatrixXf tempH;
            tempH.setZero(M,7+nDimOfWinSR);
            if (type==POSE_ONLY)
            {
                tempH.leftCols(7).swap(tempHP);
                tempH.rightCols(N).swap(tempHW);
            }
            else
                tempH.leftCols(7+N) << tempHP, tempHW;

            float val = chi2(tempH, tempr, NavSRinv.bottomRightCorner(7+nDimOfWinSR,7+nDimOfWinSR));
            if (val<CHI2_THRESHOLD[M-1])
            {
                if (xf(2)>0)
                {
                    float phi = xf(0);
                    float psi = xf(1);
                    float rho = xf(2);

                    Eigen::Vector3f epfinv;
                    epfinv << cos(phi)*sin(psi), sin(phi), cos(phi)*cos(psi);
                    Eigen::Vector3f pf = mRic*(1./rho*epfinv)+mtic;

                    if (type==POSE_ONLY)
                    {
                        Eigen::Matrix3f R = QuatToRot(Winx.tail(7).head(4));
                        Eigen::Vector3f t = Winx.tail(3);
                        pf = R*(pf-t);
                    }

                    geometry_msgs::Point feat;
                    feat.x = pf(0);
                    feat.y = pf(1);
                    feat.z = pf(2);
                    cloud_po.points.push_back(feat);
                }
            }
            else
            {
                if (type==POSE_ONLY_M)
                    (*vitInfo).second = BAD;

                continue;
            }

            nNewRows += M;

            H.conservativeResize(nNewRows,7+nDimOfWinSR);
            H.bottomRows(M).swap(tempH);

            r.conservativeResize(nNewRows);
            r.tail(M).swap(tempr);
        }

        if (nNewRows>0)
        {
            H *= mnImageNoiseSigmaInv;
            r *= mnImageNoiseSigmaInv;
            LocalQR(H, r, LocalFactor);

            mFeatPub.publish(cloud_po);
        }
    }

    Eigen::VectorXf dLocalx = LocalFactor.rightCols(1);

    int L = LocalFactor.rows();
    LocalFactor.leftCols(L).triangularView<Eigen::Upper>().solveInPlace(dLocalx);
    LocalFactor.rightCols(1).setZero();

    int nStateOffset = 0;
    int nErrorOffset = 0;

    for (const int &id : mvLostActiveFeatureIDs)
    {
        mFeatures.at(id)->Position() += dLocalx.segment(nErrorOffset,3);
        nErrorOffset += 3;
    }

    for (const int &id : vActiveFeatureIDs)
    {
        mFeatures.at(id)->Position() += dLocalx.segment(nErrorOffset,3);
        nErrorOffset += 3;
    }

    for (const int &id : mvNewActiveFeatureIDs)
    {
        mFeatures.at(id)->Position() += dLocalx.segment(nErrorOffset,3);
        nErrorOffset += 3;
    }

    // xG
    Eigen::Vector4f dqG;
    dqG.head(3) = .5*dLocalx.segment(nErrorOffset,3);
    float dqGvn = dqG.head(3).norm();
    if (dqGvn>1)
    {
        dqG.head(3) *= 1./sqrt(1.+pow(dqGvn,2));
        dqG(3) = 1./sqrt(1.+pow(dqGvn,2));
    }
    else
        dqG(3) = sqrt(1.-pow(dqGvn,2));

    Localx.segment(nStateOffset,4) = QuatMul(dqG, Localx.segment(nStateOffset,4));
    Localx.segment(nStateOffset+4,6) += dLocalx.segment(nErrorOffset+3,6);
    Localx.segment(nStateOffset+7,3).normalize();

    nStateOffset += 10;
    nErrorOffset += 9;

    // xP
    Eigen::Vector4f dqP;
    dqP.head(3) = .5*dLocalx.segment(nErrorOffset,3);
    float dqPvn = dqP.head(3).norm();
    if (dqPvn>1)
    {
        dqP.head(3) *= 1./sqrt(1.+pow(dqPvn,2));
        dqP(3) = 1./sqrt(1.+pow(dqPvn,2));
    }
    else
        dqP(3) = sqrt(1.-pow(dqPvn,2));

    Localx.segment(nStateOffset,4) = QuatMul(dqP, Localx.segment(nStateOffset,4));
    Localx.segment(nStateOffset+4,3) += dLocalx.segment(nErrorOffset+3,3);
    Localx.segment(nStateOffset+7,1) += dLocalx.segment(nErrorOffset+6,1);

    nStateOffset += 8;
    nErrorOffset += 7;

    // xW
    for (int i=0; i<nWinSize; ++i)
    {
        Eigen::Vector4f dqw;
        dqw.head(3) = .5*dLocalx.segment(nErrorOffset,3);
        float dqwvn = dqw.head(3).norm();
        if (dqwvn>1)
        {
            dqw.head(3) *= 1./sqrt(1.+pow(dqwvn,2));
            dqw(3) = 1./sqrt(1.+pow(dqwvn,2));
        }
        else
            dqw(3) = sqrt(1.-pow(dqwvn,2));

        Localx.segment(nStateOffset,4) = QuatMul(dqw, Localx.segment(nStateOffset,4));
        Localx.segment(nStateOffset+4,3) += dLocalx.segment(nErrorOffset+3,3);

        nStateOffset += 7;
        nErrorOffset += 6;
    }

    Localx.segment(nStateOffset,9) += dLocalx.segment(nErrorOffset,9);

    mRci = QuatToRot(Localx.segment(10,4));
    mtci = Localx.segment(14,3);
    mRic = mRci.transpose();
    mtic = -mRic*mtci;

    composition(nImageId, mFeatures, vActiveFeatureIDs, nDimOfWinx, nDimOfWinSR, Localx, LocalFactor);
}


void Updater::ComposeQR(const int nIdx, 
                        const int nDim, 
                        Eigen::MatrixXf& LocalFactor)
{
    int L = LocalFactor.rows();

    SqrMatrixType tempLocalSR;
    tempLocalSR = LocalFactor.block(nIdx,nIdx,nDim,L-nIdx);

    Eigen::JacobiRotation<float> GR;

    for (int n=0; n<nDim; ++n)
    {
        for (int m=nDim-1; m>n; --m)
        {
            if (tempLocalSR(m,n)!=0)
            {
                GR.makeGivens(tempLocalSR(m-1,n),tempLocalSR(m,n));
                tempLocalSR.applyOnTheLeft(m-1,m,GR.adjoint());
                tempLocalSR(m,n) = 0;
            }
        }
    }

    LocalFactor.block(nIdx,nIdx,nDim,L-nIdx).swap(tempLocalSR);
}


void Updater::composition(const int nImageId, 
                          const std::unordered_map<int,Feature*>& mFeatures, 
                          std::vector<int>& vActiveFeatureIDs, 
                          const int nDimOfWinx, 
                          const int nDimOfWinSR, 
                          Eigen::VectorXf& Localx, 
                          Eigen::MatrixXf& LocalFactor)
{
    Eigen::Matrix<float,10,1> xG = Localx.head(10);
    Eigen::Matrix3f RG = QuatToRot(xG.head(4));
    Eigen::Vector3f pG = xG.segment(4,3);
    Eigen::Vector3f g = xG.tail(3);

    Eigen::Matrix<float,16,1> xk = Localx.tail(16);
    Eigen::Matrix3f Rk = QuatToRot(xk.head(4));
    Eigen::Vector3f tk = xk.segment(4,3);
    Eigen::Matrix3f RkT = Rk.transpose();

    Eigen::Matrix3f RkG = Rk*RG;
    Eigen::Vector3f pkG = Rk*(pG-tk);
    Eigen::Matrix3f RkGT = RkG.transpose();

    Eigen::Vector3f gk = Rk*g;
    gk.normalize();

    int L = LocalFactor.rows();
    int LG = L-9-7-nDimOfWinSR;
    int LC = LG+9;
    int Lk = L-15;

    Eigen::MatrixXf Jk, Hk, HG;
    Jk.setZero(9,6);
    Jk.block(0,0,3,3).setIdentity();
    Jk.block(3,0,3,3) = SkewSymm(pkG);
    Jk.block(3,3,3,3) = -Rk;
    Jk.block(6,0,3,3) = SkewSymm(gk);

    HG.setZero(9,9);
    HG.block(0,0,3,3) = RkT;
    HG.block(3,3,3,3) = RkT;
    HG.block(6,6,3,3) = RkT;
    Hk = -HG*Jk;

    Eigen::MatrixXf tempM;
    tempM = LocalFactor.block(0,LG,LG+9,9);
    LocalFactor.block(0,LG,LG+9,9) = tempM*HG;
    LocalFactor.block(0,Lk,LG+9,6) += tempM*Hk;

    ComposeQR(LG, 9, LocalFactor);

    Localx.head(10) << RotToQuat(RkG), pkG, gk;

    int nActiveFeatures = vActiveFeatureIDs.size();
    int nNewActiveFeatures = mvNewActiveFeatureIDs.size();
    int nLostActiveFeatures = mvLostActiveFeatureIDs.size();

    if (nNewActiveFeatures>0)
    {
        int l1 = 3*nLostActiveFeatures+3*nActiveFeatures;
        int l2 = 3*nNewActiveFeatures;

        Eigen::MatrixXf Jf, JG, JP, Hf, HG, HP;
        Jf.setZero(l2,l2);
        JG.setZero(l2,6);
        JP.setZero(l2,7);

        for (int i=0; i<nNewActiveFeatures; ++i)
        {
            int id = mvNewActiveFeatureIDs.at(i);
            Feature* pFeature = mFeatures.at(id);

            Eigen::Vector3f xf = pFeature->Position();
            float phi = xf(0);
            float psi = xf(1);
            float rho = xf(2);

            Eigen::Vector3f xf_fej = pFeature->FejPosition();
            float phi_fej = xf_fej(0);
            float psi_fej = xf_fej(1);
            float rho_fej = xf_fej(2);

            Eigen::Vector3f epfinv, epfinv_fej;
            epfinv << cos(phi)*sin(psi), sin(phi), cos(phi)*cos(psi);
            epfinv_fej << cos(phi_fej)*sin(psi_fej), sin(phi_fej), cos(phi_fej)*cos(psi_fej);

            Eigen::Vector3f pfG = RkGT*(1./rho*mRic*epfinv+mtic-pkG);
            Eigen::Vector3f pfG_fej = RkGT*(1./rho_fej*mRic*epfinv_fej+mtic-pkG);

            Eigen::Matrix<float,3,2> Jang;
            Jang << -sin(phi_fej)*sin(psi_fej), cos(phi_fej)*cos(psi_fej),
                     cos(phi_fej), 0,
                    -sin(phi_fej)*cos(psi_fej), -cos(phi_fej)*sin(psi_fej);

            Jf.block(3*i,3*i,3,3) << 1./rho_fej*RkGT*mRic*Jang, -1./pow(rho_fej,2)*RkGT*mRic*epfinv_fej;
            JG.block(3*i,0,3,6) << -SkewSymm(pfG_fej)*RkGT, -RkGT;
            JP.block(3*i,0,3,7) << -RkGT*mRic*SkewSymm(1./rho_fej*epfinv_fej-mtci), -RkGT*mRic, Eigen::Vector3f::Zero();

            pFeature->SetPosition(pfG);
            pFeature->SetFejPosition(pfG_fej);
            pFeature->Inited();

            vActiveFeatureIDs.push_back(id);
        }

        mvNewActiveFeatureIDs.clear();

        Eigen::ColPivHouseholderQR<Eigen::MatrixXf> qr(Jf);
        Hf = qr.inverse();
        HG = -Hf*JG;
        HP = -Hf*JP;

        tempM = LocalFactor.block(l1,l1,l2,l2);
        LocalFactor.block(l1,l1,l2,l2) = tempM*Hf;
        LocalFactor.block(l1,LG,l2,6) += tempM*HG;
        LocalFactor.block(l1,LC,l2,7) += tempM*HP;

        ComposeQR(l1, l2, LocalFactor);
    }

    if (nLostActiveFeatures>0)
    {
        int nDimOfSR = 3*(nActiveFeatures+nNewActiveFeatures)+9+7+nDimOfWinSR;
        Eigen::MatrixXf tempM = LocalFactor.bottomRightCorner(nDimOfSR, nDimOfSR+1);
        LocalFactor.swap(tempM);

        for (const int& id : mvLostActiveFeatureIDs)
            mFeatures.at(id)->Marginalized();

        mvLostActiveFeatureIDs.clear();
    }
}

} // namespace RVIO2
