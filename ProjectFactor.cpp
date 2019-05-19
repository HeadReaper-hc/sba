//
// Created by huchao on 19-5-17.
//

#include "ProjectFactor.h"
#include <Eigen/Dense>
#include <math.h>
using namespace Eigen;
using namespace std;


bool ProjectFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {

    /*
    Vector3d rot(parameters[0][0],parameters[0][1],parameters[0][2]);
    AngleAxisd rotation_vector( rot.norm(),rot.normalized() );
    Vector3d cam_point = rotation_vector * point + trans;
     */

    Quaterniond rot(parameters[0][0],parameters[0][1],parameters[0][2],parameters[0][3]);
    Vector3d trans(parameters[0][4],parameters[0][5],parameters[0][6]);
    Vector3d point(parameters[1][0],parameters[1][1],parameters[1][2]);
    Vector3d cam_point = rot * point + trans;
   // cout<<"rot : "<<rot<<endl;
   // cout<<"trans : " <<trans<<endl;
    double f = parameters[0][7];
    double k1 = parameters[0][8];
    double k2 = parameters[0][9];




    Vector2d persp_point( -cam_point(0)/cam_point(2), -cam_point(1)/cam_point(2));
    double r = 1.0 + k1*pow(persp_point.norm(),2) + k2*pow(persp_point.norm(),4);

    Vector2d result_point(f*r*persp_point);
    residuals[0] = result_point(0) - pt.x;
    residuals[1] = result_point(1) - pt.y;

    //cout << "residuals[0] : "<<residuals[0]<<endl;
    //cout << "residuals[1] : "<<residuals[1]<<endl;

    double x = persp_point(0);
    double y = persp_point(1);
    double x_2 = persp_point(0) * persp_point(0);
    double y_2 = persp_point(1) * persp_point(1);
    double x_4 = x_2 * x_2;
    double y_4 = y_2 * y_2;
    double dx_x = f * (1 + 3 * k1 * x_2 + k1 * y_2 + 5 * k2 * x_4 + 6 * k2 * x_2 * y_2 + k2 * y_4);
    double dx_y = f * (2 * k1 * x * y + 4 * k2 * x * y * x_2 + 4 * k2 * x * y * y_2);
    double dy_x = dx_y;
    double dy_y = f * (1 + 3 * k1 * y_2 + k1 * x_2 + 5 * k2 * y_4 + 6 * k2 * x_2 * y_2 + k2 * x_4);

    Matrix2d jac_r2p;
    jac_r2p << dx_x, dx_y, dy_x, dy_y;

    double inv_z = 1.0 / cam_point(2);
    double x_inv_z_2 = cam_point(0) * inv_z * inv_z;
    double y_inv_z_2 = cam_point(1) * inv_z * inv_z;
    Matrix<double, 2, 3> jac_p2P;
    jac_p2P << -inv_z, 0, x_inv_z_2, 0, -inv_z, y_inv_z_2;
    /*
    Matrix3d jac_P2theta;
    jac_P2theta << 0, cam_point(2), -cam_point(1),
            -cam_point(2), 0, cam_point(0),
            cam_point(1), -cam_point(1), 0;
    */

    if(jacobians)
    {
        double q_w = rot.coeffs()(3);
        double q_x = rot.coeffs()(0);
        double q_y = rot.coeffs()(1);
        double q_z = rot.coeffs()(2);
        double x = point(0);
        double y = point(1);
        double z = point(2);

        double m00 = 2*( q_w*x - q_z*y + q_y*z );
        double m01 = 2*( q_x*x + q_y*y + q_z*z );
        double m02 = 2*( -q_y*x + q_x*y + q_w*z );
        double m03 = 2*( -q_z*x - q_w*y + q_x*z );

        double m10 = 2*( q_z*x + q_w*y - q_x*z );
        double m11 = 2*( q_y*x - q_x*y - q_w*z );
        double m12 = 2*( q_x*x + q_y*y + q_z*z );
        double m13 = 2*( q_w*x - q_z*y + q_y*z );

        double m20 = 2*( -q_y*x + q_x*x + q_w*z );
        double m21 = 2*( q_z*x + q_w*y - q_x*z );
        double m22 = 2*( -q_w*x + q_z*y - q_y*z );
        double m23 = 2*( q_x*x + q_y*y + q_z*z );

        /*
        double m00 = 2*( -q_z*y + q_y*z );
        double m01 = 2*( q_y*y + q_z*z );
        double m02 = 2*(-2*q_y*x + q_x*y + q_w*z );
        double m03 = 2*(-2*q_z*x - q_w*y +q_x*z );

        double m10 = 2*( q_z*x - q_x*z );
        double m11 = 2*( q_y*x - 2*q_x*y - q_w*z );
        double m12 = 2*( q_x*x + q_z*z );
        double m13 = 2*( q_w*x -2*q_z*y +q_y*z );

        double m20 = 2*( -q_y*x + q_x*y );
        double m21 = 2*( q_z*x + q_w*y - 2*q_x*z );
        double m22 = 2*( -q_w*x + q_z*y - 2*q_y*z );
        double m23 = 2*( q_x*x + q_y*y );
        */
        Matrix<double, 3, 4> jac_P2q;
        jac_P2q << m00, m01, m02, m03,
                   m10, m11, m12, m13,
                   m20, m21, m22, m23;

        /*
        if(jacobians[0]){
            Map<Matrix<double, 2, 9> > jacobian0(jacobians[0]);

            jacobian0.block<2, 3>(0, 0) = jac_r2p * jac_p2P * jac_P2theta;

            jacobian0.block<2, 3>(0, 3) = jac_r2p * jac_p2P * Matrix3d::Identity(3, 3);

            jacobian0.block<2, 1>(0, 6) = r * persp_point;


            jacobian0.block<2, 1>(0, 7) = f * pow(persp_point.norm(), 2) * persp_point;

            jacobian0.block<2, 1>(0, 8) = f * pow(persp_point.norm(), 4) * persp_point;

           // cout <<"jacobians0"<<endl;
        }
        */

        if(jacobians[0]){
            Map<Matrix<double,2,10> > jacobian0(jacobians[0]);
            jacobian0.block<2,4>(0,0) = jac_r2p * jac_p2P * jac_P2q;

            jacobian0.block<2,3>(0,4) = jac_r2p * jac_p2P * Matrix3d::Identity();

            jacobian0.block<2,1>(0,7) = r * persp_point;

            jacobian0.block<2,1>(0,8) = f * pow(persp_point.norm(), 2) * persp_point;

            jacobian0.block<2,1>(0,9) = f * pow(persp_point.norm(), 4) * persp_point;


        }

        if(jacobians[1]){

            Map<Matrix<double,2,3> > jacobian1(jacobians[1]);
            jacobian1 = jac_r2p * jac_p2P * rot.toRotationMatrix();
          //  cout<<"jacobians1"<<endl;
           // cout <<"jacobian1 : " <<jacobian1 <<endl;
        }
    }

    return true;

}
