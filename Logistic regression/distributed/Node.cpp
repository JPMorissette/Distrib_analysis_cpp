#include "Node.h"
#include <cmath>
#include <stats.hpp>
#include <iostream>
#include <fstream>
#include <string>

using namespace Eigen;
using namespace std;

// Constructor
Node::Node(const string &filename, int i) {
    data = readDataFromCSV(filename);
    performCalculations(data, i);
}

// Function to perform calculations of V and D matrix
void Node::calculVD(const string &filename, int i, int it) {
    betaCoord = readDataFromCSV(filename);
    
    // Matrix D - Gradient
    VectorXd eta = intercept_pred * betaCoord;
    VectorXd exp_eta = eta.array().exp();
    VectorXd s = exp_eta.array()/(1+exp_eta.array());
    Dnode = intercept_pred.transpose() * (outcome - s);
    
    // Matrix V - Hessian
    VectorXd mu = 1 / (1 + (-eta.array()).exp());
    VectorXd tmp = mu.array() * (1 - mu.array());
    MatrixXd Wtemp = (1/(tmp.array()*(1/tmp.array()).pow(2)));
    MatrixXd W = Wtemp.asDiagonal();
    Vnode = intercept_pred.transpose() * W * intercept_pred;

    // Write in csv file
    string filename_out = "Data_Node_" + to_string(i) + "_iter_" + to_string(it) + "_output.csv";
    writeVDToCSV(Dnode, Vnode, filename_out);
}

// Function to perform initial calculations
void Node::performCalculations(const MatrixXd& data, int i) {
    outcome = data.col(0);
    intercept_pred = data;
    intercept_pred.col(0) = VectorXd::Ones(data.rows());

    Eigen::MatrixXd nbOfData(1, 1);
    nbOfData(0,0) = outcome.rows();

    float tol = 1.0E-6;

    double av_y = outcome.sum()/outcome.rows();
    VectorXd mu = (outcome.array() + av_y) / 2.0;
    VectorXd eta = log(mu.array()/(1-mu.array())); //Logit link

    double dev = 0;
    double deltaDev = 1;
    int a = 1;

    beta = VectorXd::Zero(intercept_pred.rows());

    while (a < 50 && abs(deltaDev) > tol) {
        VectorXd tmp = mu.array() * (1 - mu.array());
        MatrixXd Wtemp = (1/(tmp.array()*(1/tmp.array()).pow(2)));
        MatrixXd W = Wtemp.asDiagonal();
        VectorXd z = eta.array() + (outcome.array() - mu.array()) / tmp.array();

        beta = (intercept_pred.transpose() * W * intercept_pred).ldlt().solve(intercept_pred.transpose() * W * z);

        eta = intercept_pred * beta;
        mu = 1 / (1 + (-eta.array()).exp());

        double dev0 = dev;
        dev = -2 * ((outcome.array() * (mu.array().log())) + ((1 - outcome.array()) * ((1 - mu.array()).log()))).sum();
        double deltaDev = dev - dev0;

        ++a;
    }

    // Write in csv file
    string filename = "Data_Node_" + to_string(i) + "_iter_0_output.csv";
    writeBetaNToCSV(beta, nbOfData, filename);
}

// Needed to read file
vector<string> Node::split(const string &s, char delimiter) {
    vector<string> tokens;
    string token;
    istringstream tokenStream(s);
    while (getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

// Read data in csv and store in matrix data
MatrixXd Node::readDataFromCSV(const string &filename){
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Could not open the file " << filename << endl;
        exit(1);
    }

    vector<vector<string>> table;
    string line;
    while (getline(file, line)) {
        vector<string> fields = split(line, ',');
        table.push_back(fields);
    }
    file.close();

    // Transform vector into eigen matrix
    int rows = table.size();
    int cols = table.empty() ? 0 : table[0].size();
    MatrixXd m(rows-1, cols);

    for (int i = 0; i < rows-1; ++i) {
        for (int j = 0; j < cols; ++j) {
            m(i, j) = stof(table[i+1][j]);
        }
    }

    return m;
}

// Write in csv file accessible to all
void Node::writeBetaNToCSV(const MatrixXd &beta, const MatrixXd &nbOfData, const string &filename){
    ofstream file(filename);
    if (file.is_open()) {
        file << "\"Beta\",\"N\"" << endl;
        
        int max_size = beta.rows();

        for (int i = 0; i < max_size; ++i) {
            if (i < beta.rows())
                file << beta(i, 0);
            file << ",";
            if (i < nbOfData.rows())
                file << nbOfData(i, 0);
            file << ",";
            file << endl;
        }

        file.close();
        cout << "Data written to " << filename << " successfully." << endl;
    } else {
        cerr << "Unable to open file: " << filename << endl;
    }
}

// Write in csv file accessible to all NOTE THIS ONE IS NOT ADPTABLE!!!
void Node::writeVDToCSV(const MatrixXd &V, const MatrixXd &D, const string &filename){
    ofstream file(filename);
    if (file.is_open()) {
        file << "\"Gradient\",\"hessian_intercept\",\"hessian_pred1\",\"hessian_pred2\",\"hessian_pred3\",\"hessian_pred4\"" << endl;
        
        int numRows = Dnode.rows();
        
        // Ensure Vnode has the correct dimensions
        if (Vnode.cols() != 5) {
            cerr << "Vnode must have 5 columns." << endl;
            return;
        }
        
        // Write the data
        for (int i = 0; i < numRows; ++i) {
            if (i < Dnode.rows())
                file << Dnode(i, 0);
            file << ",";
            for (int j = 0; j < 5; ++j) {
                if (i < Vnode.rows())
                    file << Vnode(i, j);
                if (j < 4) // Avoid trailing comma at the end of the line
                    file << ",";
            }
            file << endl;
        }
        file.close();

        cout << "yooo " << filename << "slayy" << endl;
        cout << "Data written to " << filename << " successfully." << endl;
    } else {
        cerr << "Unable to open file: " << filename << endl;
    }
}
