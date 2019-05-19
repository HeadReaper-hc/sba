//
// Created by huchao on 19-5-17.
//

#ifndef SBA_READDATA_H
#define SBA_READDATA_H
#include <string>
#include <map>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;

class readData {
public:
    readData(string& _path):path(_path){}
    readData() = delete;

    struct baData{
        vector<map<int,Point2f> > cam_ptn_pt;
        map<int, vector<double> > cams;
        map<int, vector<double> > pts;
        int numCam;
        int numPoints;
        int numObservations;
    };
    bool getDataFromTxt(baData& data);



private:

    string path;
};

typedef readData::baData optimData;


#endif //SBA_READDATA_H
