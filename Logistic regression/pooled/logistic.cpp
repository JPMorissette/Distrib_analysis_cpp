#define STATS_ENABLE_EIGEN_WRAPPERS

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include "Eigen/Dense"
#include "stats.hpp"

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

MatrixXd readCSVToMatrixXd(const string& filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Could not open the file." << endl;
        exit(1);
    }

    vector<vector<string>> table;

    string line;
    while (getline(file, line)) {
        vector<string> fields = split(line, ',');
        table.push_back(fields);
    }

    file.close();

    return convertTableToMatrixXd(table);
}

int main() {

    // READ
    MatrixXd m1 = readCSVToMatrixXd("Data_node_1.csv");
    MatrixXd m2 = readCSVToMatrixXd("Data_node_2.csv");
    MatrixXd m3 = readCSVToMatrixXd("Data_node_3.csv");

    MatrixXd merged = MatrixXd::Zero(m1.rows() + m2.rows() + m3.rows(), max(max(m1.cols(), m2.cols()), m3.cols()));
    merged.topRows(m1.rows()) = m1;
    merged.middleRows(m1.rows(), m2.rows()) = m2;
    merged.bottomRows(m3.rows()) = m3;

    // CALCULS
    VectorXd outcome = merged.col(0);
    MatrixXd intercept_pred = merged;
    intercept_pred.col(0) = VectorXd::Ones(merged.rows());

    float tol = 1.0E-6;

    double av_y = outcome.sum()/outcome.rows();
    VectorXd mu = (outcome.array() + av_y) / 2.0;
    VectorXd eta = log(mu.array()/(1-mu.array())); //Logit link

    double dev = 0;
    double deltaDev = 1;
    int i = 1;

    VectorXd b = Eigen::VectorXd::Zero(intercept_pred.rows());

    while (i < 50 && abs(deltaDev) > tol) {
        
        VectorXd tmp = mu.array() * (1 - mu.array());
        MatrixXd Wtemp = (1/(tmp.array()*(1/tmp.array().pow(2))));
        MatrixXd W = Wtemp.asDiagonal();
        VectorXd z = eta.array() + (outcome.array() - mu.array()) / tmp.array();

        b = (intercept_pred.transpose() * W * intercept_pred).ldlt().solve(intercept_pred.transpose() * W * z);

        eta = intercept_pred * b;
        mu = 1 / (1 + (-eta.array()).exp());

        double dev0 = dev;
        dev = -2 * ((outcome.array() * (mu.array().log())) + ((1 - outcome.array()) * ((1 - mu.array()).log()))).sum();
        double deltaDev = dev - dev0;

        ++i;
    }

    VectorXd tmp = mu.array() * (1 - mu.array());
    MatrixXd Wtemp = (1/(tmp.array()*(1/tmp.array().pow(2))));
    MatrixXd W = Wtemp.asDiagonal();

    MatrixXd varbeta_tmp = intercept_pred.transpose() * W * intercept_pred;
    MatrixXd varbeta = varbeta_tmp.inverse();

    VectorXd upper = b + (stats::qnorm(1 - 0.05 / 2) * varbeta.diagonal().array().sqrt()).matrix();
    VectorXd lower = b - (stats::qnorm(1 - 0.05 / 2) * varbeta.diagonal().array().sqrt()).matrix();

    std::cout << "Upper Confidence Interval Bound:\n" << upper << std::endl;
    std::cout << "Lower Confidence Interval Bound:\n" << lower << std::endl;

    // WRITE
    vector<string> row_names = {"Intercept"};
    int num_cols = intercept_pred.cols() - 1;
    for (int i = 1; i <= num_cols; ++i) {
        row_names.push_back("Pred" + to_string(i));
    }

    MatrixXd reponse(b.size(), 3);
    reponse.col(0) = b;
    reponse.col(1) = upper;
    reponse.col(2) = lower;
    
    ofstream outputFile("PoolingOrg_results_centralised_logistic_reg.csv");
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