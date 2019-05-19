//
// Created by huchao on 19-5-18.
//

#ifndef SBA_CAMERAPARAMETERIZATION_H
#define SBA_CAMERAPARAMETERIZATION_H
#include <ceres/ceres.h>

using namespace ceres;

class CameraParameterization : public LocalParameterization{

public:
    virtual ~CameraParameterization(){}

    virtual bool Plus(const double* x,
                      const double* delta,
                      double* x_plus_delta) const;

    virtual bool ComputeJacobian(const double* x,
                                 double* jacobian) const;

    virtual int GlobalSize() const { return 10; }

    virtual int LocalSize() const { return 9; }

    template<typename T> inline
    void QuaternionProduct(const T z[4],const T w[4], T zw[4]) const{
        zw[0] = z[0] * w[0] - z[1] * w[1] - z[2] * w[2] - z[3] * w[3];
        zw[1] = z[0] * w[1] + z[1] * w[0] + z[2] * w[3] - z[3] * w[2];
        zw[2] = z[0] * w[2] - z[1] * w[3] + z[2] * w[0] + z[3] * w[1];
        zw[3] = z[0] * w[3] + z[1] * w[2] - z[2] * w[1] + z[3] * w[0];
    }
};


#endif //SBA_CAMERAPARAMETERIZATION_H
