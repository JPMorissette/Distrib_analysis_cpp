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

int main() {

    // # Extracts data from the CSV and creates R data frame
    // pooled_data <- read.csv("Pooled_data.csv")

    // Open the CSV file
    ifstream file("Pooled_data.csv");
    if (!file.is_open()) {
        cerr << "Error: Could not open the file." << endl;
        return 1;
    }

    // Define a vector to store the table data
    vector<vector<string>> table;

    // Read the CSV data line by line
    string line;
    while (getline(file, line)) {
        // Split each line into fields based on the comma delimiter
        vector<string> fields = split(line, ',');

        // Add the fields to the table
        table.push_back(fields);
    }

    // Close the file
    file.close();

    // Display the table
    // for (const auto& row : table) {
    //     for (const auto& field : row) {
    //         cout << field << "\t";
    //     }
    //     cout << endl;
    // }

    // ## Code assumes a data frame where the first column is the outcome
    // ## Creates a data frame with the outcome
    // outcome <- pooled_data[c(1)]
    // ## Code assumes the columns 2 and following are predictors
    // ## Creates a data frame with the predictors
    // predictors <- pooled_data[-c(1)]
    // ## Create a frame with the intercept (here 1) for each observation
    // intercept <- rep(1,nrow(pooled_data))
    // ## joins the intercepts and the predictors
    // intercept_pred <- data.frame(intercept,predictors)

    // Convert table to MatrixXd
    MatrixXd m = convertTableToMatrixXd(table);

    VectorXd outcome = m.col(0);
    MatrixXd intercept_pred = m;
    intercept_pred.col(0) = VectorXd::Ones(m.rows());

    // MatrixXd predictors =  m.rightCols(m.cols() - 1);
    // VectorXd intercept = VectorXd::Ones(outcome.size());
    // MatrixXd intercept_pred(outcome.size(), predictors.cols() + intercept.cols());
    // intercept_pred << intercept, predictors;

    // cout << "outcome:\n" << outcome << endl;
    // cout << "intercept_pred:\n" << intercept_pred << endl;


    // # Summary statistics for the coefficient estimates in linear regression model----

    // xtx <- t(as.matrix(intercept_pred))%*%as.matrix(intercept_pred)
    // xtx_inverse <- solve(xtx)
    // yty <- t(as.matrix(outcome))%*%as.matrix(outcome)
    // xty <- t(as.matrix(intercept_pred))%*%as.matrix(outcome)

    MatrixXd xtx = intercept_pred.transpose() * intercept_pred;
    MatrixXd xtx_inverse = xtx.inverse();
    MatrixXd yty = outcome.transpose() * outcome;
    MatrixXd xty = intercept_pred.transpose() * outcome;

    // # Coefficient estimates in linear regression model-------------------------

    // # Coefficients and Variance matrix

    // beta <- xtx_inverse%*%xty
    // varbeta <- (1/(nrow(pooled_data)-ncol(intercept_pred)))*drop((yty-((t(beta))%*%xty))) * xtx_inverse

    VectorXd beta = xtx_inverse * xty;
    
    //MatrixXd temp = yty-(beta.transpose()*xty);
    MatrixXd varbeta = (1.0/(outcome.rows()-intercept_pred.cols()))*((yty-(beta.transpose()*xty))(0,0))*xtx_inverse;
    
    // cout << "beta:\n" << beta << endl;
    // cout << "varbeta:\n" << varbeta << endl;


    // # Confidence interval with alpha=0.05

    // upper <- beta + qt(p=.05/2, df=nrow(pooled_data)-ncol(intercept_pred), lower.tail=FALSE)*sqrt(diag(varbeta))
    // lower <- beta - qt(p=.05/2, df=nrow(pooled_data)-ncol(intercept_pred), lower.tail=FALSE)*sqrt(diag(varbeta))

    VectorXd upper = beta + (stats::qt(1 - 0.05 / 2, m.rows() - intercept_pred.cols()) * varbeta.diagonal().array().sqrt()).matrix();
    VectorXd lower = beta - (stats::qt(1 - 0.05 / 2, m.rows() - intercept_pred.cols()) * varbeta.diagonal().array().sqrt()).matrix();
    
    std::cout << "Upper Confidence Interval Bound:\n" << upper << std::endl;
    std::cout << "Lower Confidence Interval Bound:\n" << lower << std::endl;


    // output <- setNames(data.frame(beta,upper,lower, row.names = c("Intercept",paste0("Pred", c(1:ncol(predictors))))), c("Beta", "Upper", "Lower"))

    // ## Producing the CSV file containing the final outputs
    // write.csv(output, file="PoolingOrg_results_centralised_lin_reg.csv")

    vector<string> row_names = {"Intercept", "Pred1", "Pred2"};

    MatrixXd reponse(beta.size(), 3);
    reponse.col(0) = beta;
    reponse.col(1) = upper;
    reponse.col(2) = lower;
    
    // Convert the matrix to a data frame-like structure
    ofstream outputFile("PoolingOrg_results_centralised_lin_reg.csv");
    if (outputFile.is_open()) {
        // Write column names
        outputFile << """,Beta,Upper,Lower" << endl;
        
        // Write row names and values
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