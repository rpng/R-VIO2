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

#include <Eigen/Dense>

#include "Propagator.h"


namespace RVIO2
{

Eigen::Matrix3f I = Eigen::Matrix3f::Identity();
Eigen::MatrixXf I15 = Eigen::MatrixXf::Identity(15,15);
Eigen::MatrixXf I18 = Eigen::MatrixXf::Identity(18,18);


Propagator::Propagator(const cv::FileStorage& fsSettings)
{
    mnImuRate = fsSettings["IMU.dps"];

    mnGravity = fsSettings["IMU.nG"];
    mnSmallAngle = fsSettings["IMU.nSmallAngle"];

    mnGyroNoiseSigma = fsSettings["IMU.sigma_g"];
    mnGyroRandomWalkSigma = fsSettings["IMU.sigma_wg"];
    mnAccelNoiseSigma = fsSettings["IMU.sigma_a"];
    mnAccelRandomWalkSigma = fsSettings["IMU.sigma_wa"];

    mSigma.setIdentity();
    mSigma.block<3,3>(0,0) *= pow(mnGyroNoiseSigma,2);
    mSigma.block<3,3>(3,3) *= pow(mnGyroRandomWalkSigma,2);
    mSigma.block<3,3>(6,6) *= pow(mnAccelNoiseSigma,2);
    mSigma.block<3,3>(9,9) *= pow(mnAccelRandomWalkSigma,2);

    const int nMaxTrackingLength = fsSettings["Tracker.nMaxTrackingLength"];
    mnLocalWindowSize = nMaxTrackingLength-1;
}


void Propagator::CreateNewFactor(const std::vector<ImuData>& vImuData, 
                                 const Eigen::VectorXf& Localx, 
                                 Eigen::Matrix<float,16,1>& x, 
                                 Eigen::Matrix<float,15,27>& H)
{
    Eigen::VectorXf vb = Localx.tail(9);
    Eigen::Vector3f vk = vb.segment(0,3);
    Eigen::Vector3f bg = vb.segment(3,3);
    Eigen::Vector3f ba = vb.segment(6,3);

    Eigen::Vector3f gk = Localx.segment(7,3);

    // Local velocity/gravity in {Rk}
    Eigen::Vector3f vR = vk;
    Eigen::Vector3f gR = gk;

    Eigen::Matrix3f Rk, RkT;
    Eigen::Vector3f pk;
    Rk.setIdentity();
    RkT.setIdentity();
    pk.setZero();

    Eigen::Vector3f dp, dv;
    dp.setZero();
    dv.setZero();

    Eigen::Matrix<float,18,18> F, Phi, Psi;
    Eigen::Matrix<float,18,12> G;
    Eigen::Matrix<float,18,18> Q, P;
    F.setZero();
    Phi.setZero();
    Psi.setIdentity();
    G.setZero();
    Q.setZero();
    P.setZero();

    double Dt = 0;

    for (const ImuData& data : vImuData)
    {
        Eigen::Vector3f wm = data.AngularVel;
        Eigen::Vector3f am = data.LinearAccel;
        double dt = data.TimeInterval;

        Eigen::Vector3f w = wm-bg;
        Eigen::Vector3f a = am-ba;
        Dt += dt;

        float wn = w.norm();
        Eigen::Matrix3f wx = SkewSymm(w);
        Eigen::Matrix3f wx2 = wx*wx;
        Eigen::Matrix3f vx = SkewSymm(vk);
        Eigen::Matrix3f gx = SkewSymm(gk);

        // Covariance
        F.block<3,3>(3,3) = -wx;
        F.block<3,3>(3,12) = -I;
        F.block<3,3>(6,3) = -RkT*vx;
        F.block<3,3>(6,9) = RkT;
        F.block<3,3>(9,0) = -mnGravity*Rk;
        F.block<3,3>(9,3) = -mnGravity*gx;
        F.block<3,3>(9,9) = -wx;
        F.block<3,3>(9,12) = -vx;
        F.block<3,3>(9,15) = -I;

        G.block<3,3>(3,0) = -I;
        G.block<3,3>(9,0) = -vx;
        G.block<3,3>(9,6) = -I;
        G.block<3,3>(12,3) = I;
        G.block<3,3>(15,9) = I;

        Phi = I18+dt*F;
        Q = dt*G*mSigma*(G.transpose());

        P = Phi*P*(Phi.transpose())+Q;
        P = .5*(P+P.transpose()).eval();

        Psi.applyOnTheLeft(Phi);

        // State 
        Eigen::Matrix3f dR;
        float f1, f2, f3, f4;
        if (wn<mnSmallAngle)
        {
            dR = I-dt*wx+.5*pow(dt,2)*wx2;

            f1 = -pow(dt,3)/3;
            f2 = pow(dt,4)/8;
            f3 = -pow(dt,2)/2;
            f4 = pow(dt,3)/6;
        }
        else
        {
            dR = I-sin(wn*dt)/wn*wx+(1.-cos(wn*dt))/pow(wn,2)*wx2;

            f1 = (wn*dt*cos(wn*dt)-sin(wn*dt))/pow(wn,3);
            f2 = .5*(pow(wn*dt,2)-2*cos(wn*dt)-2*wn*dt*sin(wn*dt)+2)/pow(wn,4);
            f3 = (cos(wn*dt)-1.)/pow(wn,2);
            f4 = (wn*dt-sin(wn*dt))/pow(wn,3);
        }

        Rk.applyOnTheLeft(dR);
        RkT = Rk.transpose();

        dp += dv*dt;
        dp += RkT*(.5*pow(dt,2)*I+f1*wx+f2*wx2)*a;
        dv += RkT*(dt*I+f3*wx+f4*wx2)*a;

        pk = vR*Dt-.5*mnGravity*gR*pow(Dt,2)+dp;
        vk = Rk*(vR-mnGravity*gR*Dt+dv);

        gk = Rk*gR;
        gk.normalize();
    }

    Eigen::Matrix<float,15,15> Info;

    Eigen::ColPivHouseholderQR<Eigen::MatrixXf> qr(P.bottomRightCorner(15,15));
    if (qr.isInvertible())
        Info = qr.inverse();
    else
        Info = pseudoInverse(P.bottomRightCorner(15,15));

    Info = .5*(Info+Info.transpose()).eval();
    Eigen::Matrix<float,15,15> Low = Info.llt().matrixL();

    H << Psi.bottomLeftCorner(15,3), Psi.bottomRightCorner(15,9), -I15;
    H.applyOnTheLeft(Low.adjoint());

    x << RotToQuat(Rk), pk, vk, bg, ba;
}


void Propagator::LocalQR(const int nImageId, 
                         const Eigen::Matrix<float,16,1>& x, 
                         const Eigen::Matrix<float,15,27>& H, 
                         Eigen::VectorXf& Localx, 
                         Eigen::MatrixXf& LocalFactor)
{
    int L = LocalFactor.rows();
    int W = nImageId>mnLocalWindowSize ? 6*mnLocalWindowSize+9 : 6*(nImageId-1)+9;

    int Lg = L-W-7-3;
    int Lv = L-9;

    SqrMatrixType tempLocalSR;
    tempLocalSR.setZero(L+15,L+15);
    tempLocalSR.topLeftCorner(L,L) = LocalFactor.leftCols(L);
    tempLocalSR.block(L,Lg,15,3) = H.leftCols(3);
    tempLocalSR.block(L,Lv,15,9) = H.block(0,3,15,9);
    tempLocalSR.bottomRightCorner(15,15) = H.rightCols(15);

    FlipToHead(tempLocalSR.leftCols(L), 9);

    if (nImageId>mnLocalWindowSize)
        FlipToHead(tempLocalSR.leftCols(9+L-W+6), 6);

    Eigen::JacobiRotation<float> GR;

    for (int n=0; n<L+15; ++n)
    {
        for (int m=L+14; m>n; --m)
        {
            GR.makeGivens(tempLocalSR(m-1,n),tempLocalSR(m,n));
            tempLocalSR.applyOnTheLeft(m-1,m,GR.adjoint());
        }
    }

    if (nImageId>mnLocalWindowSize)
    {
        LocalFactor.leftCols(L) = tempLocalSR.bottomRightCorner(L,L).triangularView<Eigen::Upper>();
    }
    else
    {
        LocalFactor.setZero(L+6,L+7);
        LocalFactor.leftCols(L+6) = tempLocalSR.bottomRightCorner(L+6,L+6).triangularView<Eigen::Upper>();
    }

    if (nImageId>mnLocalWindowSize)
    {
        int w = 7*mnLocalWindowSize+9;
        Eigen::VectorXf tempx = Localx.segment(10+8+7,w-16);
        Localx.segment(10+8,w-16).swap(tempx);
        Localx.tail(16) = x;
    }
    else
    {
        int l = Localx.rows();
        Localx.conservativeResize(l+7);
        Localx.tail(16) = x;
    }
}


void Propagator::propagate(const int nImageId, 
                           const std::vector<ImuData>& vImuData, 
                           Eigen::VectorXf& Localx, 
                           Eigen::MatrixXf& LocalFactor)
{
    Eigen::Matrix<float,16,1> x;
    Eigen::Matrix<float,15,27> H;
    CreateNewFactor(vImuData, Localx, x, H);

    LocalQR(nImageId, x, H, Localx, LocalFactor);
}

} // namespace RVIO2
