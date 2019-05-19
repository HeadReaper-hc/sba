//
// Created by huchao on 19-5-18.
//

#include "CameraParameterization.h"
#include <Eigen/Dense>
using namespace Eigen;
using namespace std;

bool CameraParameterization::Plus(const double *x, const double *delta, double *x_plus_delta) const {

/*    cout <<"delta : " <<delta[0]<<","<<delta[1]<<","<<delta[2]<<","<<delta[3]<<","<<delta[4]<<","<<delta[5]<<","<<delta[6]
         <<","<<delta[7]<<","<<delta[8]<<endl; */

    const double norm_delta =
            sqrt(delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]);
    if (norm_delta > 0.0) {
        const double sin_delta_by_delta = (sin(norm_delta) / norm_delta);
        double q_delta[4];
        q_delta[0] = cos(norm_delta);
        q_delta[1] = sin_delta_by_delta * delta[0];
        q_delta[2] = sin_delta_by_delta * delta[1];
        q_delta[3] = sin_delta_by_delta * delta[2];
        QuaternionProduct(q_delta, x, x_plus_delta);



    }else {
        for (int i = 0; i < 4; ++i)
            x_plus_delta[i] = x[i];
    }

    for (int i = 4; i < 10; ++i)
        x_plus_delta[i] = x[i] + delta[ i-1 ];

    return true;

}

bool CameraParameterization::ComputeJacobian(const double *x, double *jacobian) const {

    Map<MatrixXd> Jac(jacobian,9,10);
    Jac.setZero();
    jacobian[0] = -x[1]; jacobian[1]  = -x[2]; jacobian[2]  = -x[3];  // NOLINT
    jacobian[3] =  x[0]; jacobian[4]  =  x[3]; jacobian[5]  = -x[2];  // NOLINT
    jacobian[6] = -x[3]; jacobian[7]  =  x[0]; jacobian[8]  =  x[1];  // NOLINT
    jacobian[9] =  x[2]; jacobian[10] = -x[1]; jacobian[11] =  x[0];  // NOLINT

    Jac.block<6,6>(3,4).setIdentity();

    return true;
}

