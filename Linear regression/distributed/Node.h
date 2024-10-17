#ifndef NODE_H
#define NODE_H

#include "Eigen/Dense"
#include <string>
#include <vector>

using namespace Eigen;
using namespace std;

class Node {
public:
    Node(const string &filename, int i);

private:
    MatrixXd data;
    VectorXd outcome;
    MatrixXd intercept_pred;
    MatrixXd xtx;
    MatrixXd xty;
    MatrixXd yty;

    void performCalculations(const MatrixXd& data, int i);
    
    vector<string> split(const string &s, char delimiter);
    void readDataFromCSV(const string &filename);
    void writeDataToCSV(const MatrixXd& xtx, const MatrixXd& yty, const MatrixXd& xty, const string& filename);
};

#endif // NODE_H
