//
// Created by huchao on 19-5-17.
//

#include "readData.h"
#include <iostream>
#include <string>

bool readData::getDataFromTxt(baData& data) {
    ifstream file;
    file.open(path, ios::in);

    if( !file.is_open() )
        return false;

    int numCam;
    file >> numCam;
    data.numCam = numCam;

    int numPoints;
    file >> numPoints;
    data.numPoints = numPoints;

    int numObservations;
    file >> numObservations;
    data.numObservations = numObservations;

    cout << "The number of camera is : " << data.numCam <<endl;
    cout << "The number of points is : " << data.numPoints <<endl;
    cout << "The number of observations is : " <<data.numObservations <<endl;

    if( data.numCam ==0 ||
        data.numPoints ==0 ||
        data.numObservations ==0)
        return false;

    data.cam_ptn_pt.resize(data.numCam);

    {
        for(int i=0;i<data.numObservations;++i){
            int camSubCoord;
            int pointSubCoord;
            Point2f pt;
            file >> camSubCoord;
            file >> pointSubCoord;
            file >> pt.x;
            file >> pt.y;

            data.cam_ptn_pt[camSubCoord].insert(
                    make_pair(pointSubCoord,pt) );
        }

        int tempNumObservations = 0;
        for(int i=0;i<data.cam_ptn_pt.size();++i)
        {
            tempNumObservations += data.cam_ptn_pt[i].size();
        }
        if( tempNumObservations != data.numObservations ){
            cout<<"The real number of observations is not "<<data.numObservations<<endl;
            return false;
        }

        for(int i=0;i<data.numCam;++i){
            vector<double> cam_param_temp;
            double temp;
            for(int j=0;j<9;++j){
                file >> temp;
                cam_param_temp.push_back(temp);
            }
            data.cams.insert(make_pair(i,cam_param_temp));
        }

        for(int i=0;i<data.numPoints;++i){
            vector<double> points_temp;
            double temp;
            for(int j=0;j<3;j++){
                file >> temp;
                points_temp.push_back(temp);
            }
            data.pts.insert(make_pair(i,points_temp));
        }
    }
    string temp;
    file >> temp;
    if(file.eof())
        return true;
    else
        return false;

}


