#define STATS_ENABLE_EIGEN_WRAPPERS

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <Eigen/Dense>
#include <stats.hpp>

using namespace std;
using namespace Eigen;

// Function to split a string into a vector of strings based on a delimiter
vector<string> split(const string& s, char delimiter) {
    vector<string> tokens;
    string token;
    istringstream tokenStream(s);
    while (getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

// +1, -1 à cause du titre dans le csv
MatrixXd convertTableToMatrixXd(const vector<vector<string>>& table) {
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

int main() {

    // READ
    ifstream file("Pooled_data.csv");
    if (!file.is_open()) {
        cerr << "Error: Could not open the file." << endl;
        return 1;
    }

    vector<vector<string>> table;

    string line;
    while (getline(file, line)) {
        vector<string> fields = split(line, ',');
        table.push_back(fields);
    }

    file.close();

    // Convert table to MatrixXd
    MatrixXd m = convertTableToMatrixXd(table);

    // CALCULS
    VectorXd outcome = m.col(0);
    MatrixXd intercept_pred = m;
    intercept_pred.col(0) = VectorXd::Ones(m.rows());

    MatrixXd xtx = intercept_pred.transpose() * intercept_pred;
    MatrixXd xtx_inverse = xtx.inverse();
    MatrixXd yty = outcome.transpose() * outcome;
    MatrixXd xty = intercept_pred.transpose() * outcome;

    VectorXd beta = xtx_inverse * xty;
    MatrixXd varbeta = (1.0/(outcome.rows()-intercept_pred.cols()))*((yty-(beta.transpose()*xty))(0,0))*xtx_inverse;

    VectorXd upper = beta + (stats::qt(1 - 0.05 / 2, m.rows() - intercept_pred.cols()) * varbeta.diagonal().array().sqrt()).matrix();
    VectorXd lower = beta - (stats::qt(1 - 0.05 / 2, m.rows() - intercept_pred.cols()) * varbeta.diagonal().array().sqrt()).matrix();
    
    // WRITE
    // vector<string> row_names = {"Intercept", "Pred1", "Pred2"};

    // More flexible, variable length
    vector<string> row_names = {"Intercept"};
    int num_cols = intercept_pred.cols() - 1;
    for (int i = 1; i <= num_cols; ++i) {
        row_names.push_back("Pred" + to_string(i));
    }

    MatrixXd reponse(beta.size(), 3);
    reponse.col(0) = beta;
    reponse.col(1) = upper;
    reponse.col(2) = lower;
    
    ofstream outputFile("PoolingOrg_results_centralised_lin_reg.csv");
    if (outputFile.is_open()) {
        outputFile << """,Beta,Upper,Lower" << endl;
        
        for (int i = 0; i < reponse.rows(); ++i) {
            outputFile << row_names[i] << ",";
            for (int j = 0; j < reponse.cols(); ++j) {
                outputFile << reponse(i, j);
                if (j < reponse.cols() - 1) {
                    outputFile << ",";
                }
            }
            outputFile << endl;
        }
        outputFile.close();
        cout << "CSV file created successfully." << endl;
    } else {
        cout << "Unable to create CSV file." << endl;
    }

    return 0;
}