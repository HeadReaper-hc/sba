#include <iostream>
#include "readData.h"
#include <ceres/ceres.h>
#include "ProjectFactor.h"
#include "glog/logging.h"
#include <Eigen/Dense>
#include "CameraParameterization.h"
#include <ceres/rotation.h>
using namespace Eigen;
using namespace cv;
using namespace ceres;
void camVector2double( optimData& data, double** newData );
void pointsVector2double( optimData& data, double** newData);
//#define AUTODIFF

struct CameraCostFunctor{

    CameraCostFunctor(Point2f _pt){
        pt = _pt;
    }

    template <typename T>
    bool operator()(const T* const camera,const T* const point, T* residuals) const{
        /*
        double q_w = static_cast<double>( camera[0] );
        double q_x = static_cast<double>( camera[1] );
        double q_y = static_cast<double>( camera[2] );
        double q_z = static_cast<double>( camera[3] );
        Quaterniond rot(q_w, q_x, q_y, q_z);
        Vector3d trans(camera[4],camera[5],camera[6]);
        Vector3d pointVec(point[0],point[1],point[2]);
        Vector3d cam_point = rot * pointVec + trans;

        double f = camera[7];
        double k1 = camera[8];
        double k2 = camera[9];

        Vector2d persp_point( -cam_point(0)/cam_point(2), -cam_point(1)/cam_point(2));
        double r = 1.0 + k1*pow(persp_point.norm(),2) + k2*pow(persp_point.norm(),4);
        Vector2d result_point(f*r*persp_point);
        residual[0] = result_point(0) - pt.x;
        residual[1] = result_point(1) - pt.y;
        return true;
        */

        /*
        T p[3];
        ceres::AngleAxisRotatePoint(camera, point, p);
        // camera[3,4,5] are the translation.
        p[0] += camera[3]; p[1] += camera[4]; p[2] += camera[5];

        // Compute the center of distortion. The sign change comes from
        // the camera model that Noah Snavely's Bundler assumes, whereby
        // the camera coordinate system has a negative z axis.
        T xp = - p[0] / p[2];
        T yp = - p[1] / p[2];

        // Apply second and fourth order radial distortion.
        const T& l1 = camera[7];
        const T& l2 = camera[8];
        T r2 = xp*xp + yp*yp;
        T distortion = T(1.0) + r2  * (l1 + l2  * r2);

        // Compute final projected point position.
        const T& focal = camera[6];
        T predicted_x = focal * distortion * xp;
        T predicted_y = focal * distortion * yp;

        // The error is the difference between the predicted and observed position.
        residuals[0] = predicted_x - T(pt.x);
        residuals[1] = predicted_y - T(pt.y);
        return true;
         */

        T p[3];
        ceres::QuaternionRotatePoint(camera, point, p);
        p[0] += camera[4]; p[1] += camera[5]; p[2] += camera[6];

        // Compute the center of distortion. The sign change comes from
        // the camera model that Noah Snavely's Bundler assumes, whereby
        // the camera coordinate system has a negative z axis.
        T xp = - p[0] / p[2];
        T yp = - p[1] / p[2];

        // Apply second and fourth order radial distortion.
        const T& l1 = camera[8];
        const T& l2 = camera[9];
        T r2 = xp*xp + yp*yp;
        T distortion = T(1.0) + r2  * (l1 + l2  * r2);

        // Compute final projected point position.
        const T& focal = camera[7];
        T predicted_x = focal * distortion * xp;
        T predicted_y = focal * distortion * yp;

        // The error is the difference between the predicted and observed position.
        residuals[0] = predicted_x - T(pt.x);
        residuals[1] = predicted_y - T(pt.y);
     //   cout <<"Residuals[0] : "<<residuals[0]<<endl;
     //   cout <<"Residuals[1] : "<<residuals[1]<<endl;
        return true;

    }
private:
    Point2f pt;

};


int main(int argc, char** argv) {

    google::InitGoogleLogging(argv[0]);

    if( argc!=2 ) {
        cout << "At least need two parameters , usage as " << argv[0] << " ${file_path}" << endl;
        return 0;
    }


    string path = argv[1];

    optimData sbaData;

    readData rd(path);
    if(rd.getDataFromTxt(sbaData))
        cout<<"Read data process is successful..." <<endl;
    else {
        cout<<"Read data process is failed..."<<endl;
        return 0;
    }

    /*   test the read data
    cout << (sbaData.cam_ptn_pt[48])[7775] <<endl;
    for(int i=0;i<9;++i)
        cout << (sbaData.cams[48])[i] <<endl;
    for(int i=0;i<3;++i)
        cout << (sbaData.pts[7775])[i] <<endl;
    */
#ifdef AUTODIFF

    double** camData = new double* [sbaData.numCam];
    for(int i=0;i<sbaData.numCam;++i){
        camData[i] =new double [9];
    }

#else

    double** camData = new double* [sbaData.numCam];
    for(int i=0;i<sbaData.numCam;++i){
        camData[i] =new double [10];
    }

#endif

    double** pointData = new double* [sbaData.numPoints];
    for(int i=0;i<sbaData.numPoints;++i){
        pointData[i] = new double [3];
    }

   camVector2double(sbaData, camData);
   pointsVector2double(sbaData, pointData);

   Problem problem;
#ifdef AUTODIFF
   for(int i=0;i<sbaData.numCam;++i){
    ceres::LocalParameterization *local_parameterization = new CameraParameterization();
       problem.AddParameterBlock(camData[i],9);
   }
#else

   for(int i=0;i<sbaData.numCam;++i){
       ceres::LocalParameterization *local_parameterization = new ProductParameterization(
               new QuaternionParameterization(),
               new IdentityParameterization(6));
       problem.AddParameterBlock(camData[i],10,local_parameterization);
   }
#endif

   for(int i=0;i<sbaData.numPoints;++i){
       problem.AddParameterBlock(pointData[i],3);
   }

   problem.SetParameterBlockConstant(camData[0]);
   problem.SetParameterBlockConstant(pointData[0]);



#ifdef AUTODIFF

    for(int i=0;i<sbaData.numCam;++i){
       map<int,Point2f>::iterator iter;
       for(iter = (sbaData.cam_ptn_pt)[i].begin();iter != (sbaData.cam_ptn_pt)[i].end(); iter++){
           CostFunction* cost_function =
                   new AutoDiffCostFunction<CameraCostFunctor, 2, 9, 3>(
                           new CameraCostFunctor( (*iter).second ) );
           problem.AddResidualBlock(cost_function, NULL, camData[i], pointData[(*iter).first]);
       }
   }

#else
  /*
    for(int i=0;i<sbaData.numCam;++i){
        map<int,Point2f>::iterator iter;
        for(iter = (sbaData.cam_ptn_pt)[i].begin();iter != (sbaData.cam_ptn_pt)[i].end(); iter++){
            CostFunction* cost_function =
                    new AutoDiffCostFunction<CameraCostFunctor, 2, 10, 3>(
                            new CameraCostFunctor( (*iter).second ) );
            problem.AddResidualBlock(cost_function, NULL, camData[i], pointData[(*iter).first]);
        }
    }
    */

    for(int i=0;i<sbaData.numCam;++i){
        map<int,Point2f>::iterator iter;
        for(iter = (sbaData.cam_ptn_pt)[i].begin();iter != (sbaData.cam_ptn_pt)[i].end(); iter++ ){
            CostFunction* cost_function = new ProjectFactor( (*iter).second );
            problem.AddResidualBlock(cost_function,NULL,camData[i],pointData[(*iter).first]);
        }
   }


#endif

    Solver::Options options;
    options.minimizer_progress_to_stdout = true;
    Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout <<summary.FullReport()<<"\n";



    return 0;
}

#ifdef AUTODIFF
void camVector2double( optimData& data, double** newData )
{
    for(int i=0;i<data.numCam;++i){


        newData[i][0] = ((data.cams)[i])[0];
        newData[i][1] = ((data.cams)[i])[1];
        newData[i][2] = ((data.cams)[i])[2];
        newData[i][3] = ((data.cams)[i])[3];
        newData[i][4] = ((data.cams)[i])[4];
        newData[i][5] = ((data.cams)[i])[5];
        newData[i][6] = ((data.cams)[i])[6];
        newData[i][7] = ((data.cams)[i])[7];
        newData[i][8] = ((data.cams)[i])[8];

    }
}
#else
void camVector2double( optimData& data, double** newData )
{
    for(int i=0;i<data.numCam;++i){
        double x[3];
        x[0] = ((data.cams)[i])[0];
        x[1] = ((data.cams)[i])[1];
        x[2] = ((data.cams)[i])[2];
        AngleAxisToQuaternion( x,newData[i] );
        newData[i][4] = ((data.cams)[i])[3];
        newData[i][5] = ((data.cams)[i])[4];
        newData[i][6] = ((data.cams)[i])[5];
        newData[i][7] = ((data.cams)[i])[6];
        newData[i][8] = ((data.cams)[i])[7];
        newData[i][9] = ((data.cams)[i])[8];

    }
}

#endif

void pointsVector2double( optimData& data, double** newData)
{
    for(int i=0;i<data.numPoints;++i)
    {
        newData[i][0] = ((data.pts)[i])[0];
        newData[i][1] = ((data.pts)[i])[1];
        newData[i][2] = ((data.pts)[i])[2];

    }
}