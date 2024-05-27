#ifndef NODE_H
#define NODE_H

#include <Eigen/Dense>
#include <string>

using namespace Eigen;
using namespace std;

class Node {
public:
    Node(const string &filename, int i);

    void calculVD(const string &filename, int i, int it);

private:
    MatrixXd data;
    VectorXd outcome;
    MatrixXd intercept_pred;
    MatrixXd beta;

    MatrixXd betaCoord;
    MatrixXd Vnode;
    MatrixXd Dnode;

    void performCalculations(const MatrixXd& data, int i);
    
    vector<string> split(const string &s, char delimiter);
    MatrixXd readDataFromCSV(const string &filename);

    void writeBetaNToCSV(const MatrixXd &beta, const MatrixXd &nbOfData, const string &filename);
    void writeVDToCSV(const MatrixXd &V, const MatrixXd &D, const string &filename);
};

#endif // NODE_H
