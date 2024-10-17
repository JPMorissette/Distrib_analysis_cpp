// NodeInput.h

#ifndef NODEINPUT_H
#define NODEINPUT_H

#include "Eigen/Dense"
#include <string>

using namespace Eigen;
using namespace std;

class NodeInput {
public:
    MatrixXd xtx;
    MatrixXd xty;
    MatrixXd yty;
    
    NodeInput(const string &filename);

private:
    void readDataFromCSV(const std::string &filename);
};

#endif // NODEINPUT_H