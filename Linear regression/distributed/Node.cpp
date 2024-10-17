#include "Node.h"
#include <cmath>
#include "stats.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>

using namespace Eigen;
using namespace std;

// Constructor
Node::Node(const string &filename, int i) {
    readDataFromCSV(filename);
    performCalculations(data, i);
}

// Function to perform all calculations
void Node::performCalculations(const MatrixXd& data, int i) {
    outcome = data.col(0);
    intercept_pred = data;
    intercept_pred.col(0) = VectorXd::Ones(data.rows());

    xtx = intercept_pred.transpose() * intercept_pred;
    yty = outcome.transpose() * outcome;
    xty = intercept_pred.transpose() * outcome;

    // Write in csv file
    Eigen::Map<Eigen::VectorXd> flat_xtx(xtx.data(), xtx.size());
    string filename = "Node" + to_string(i) + "_output.csv";
    writeDataToCSV(flat_xtx, yty, xty, filename);
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
void Node::readDataFromCSV(const string &filename){
    // Open the file
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

    data = m;
}

// Write xtx, xty and yty in csv file accessible to all
void Node::writeDataToCSV(const MatrixXd &xtx, const MatrixXd &yty, const MatrixXd &xty, const string &filename){
    ofstream file(filename);
    if (file.is_open()) {
        file << "\"xtx\",\"yty\",\"xty\"" << endl;
        
        file << std::fixed << std::setprecision(16);

        int max_size = max({xtx.rows(), yty.rows(), xty.rows()});

        for (int i = 0; i < max_size; ++i) {
            if (i < xtx.rows())
                file << xtx(i, 0);
            file << ",";
            if (i < yty.rows())
                file << yty(i, 0);
            file << ",";
            if (i < xty.rows())
                file << xty(i, 0);
            file << endl;
        }

        file.close();
        cout << "Data written to " << filename << " successfully." << endl;
    } else {
        cerr << "Unable to open file: " << filename << endl;
    }
}
