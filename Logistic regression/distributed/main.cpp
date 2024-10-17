#define STATS_ENABLE_EIGEN_WRAPPERS

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include "stats.hpp"
#include "Node.h"

using namespace std;
using namespace Eigen;

// Write beta in csv file accessible to nodes
void writeBetaToCSV(const MatrixXd &beta, const string &filename){
    ofstream file(filename);
    if (file.is_open()) {
        file << "\"coefs\"" << endl;
        for (int i = 0; i < beta.rows(); ++i) {
            file << beta(i, 0);
            file << ",";
            file << endl;
        }
        file.close();
        cout << "Data written to " << filename << " successfully." << endl;
    } else {
        cerr << "Unable to open file: " << filename << endl;
    }
}

// Write final results in csv file
void writeResultsToCSV(const string& filename, const vector<string>& row_names, const MatrixXd& reponse) {
    ofstream outputFile(filename);
    if (outputFile.is_open()) {
        outputFile << "Variable,Beta,Lower,Upper" << endl;
        
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
        cout << "CSV file created successfully: " << filename << endl;
    } else {
        cerr << "Unable to create CSV file: " << filename << endl;
    }
}

// Needed to read file
vector<string> split(const string &s, char delimiter) {
    vector<string> tokens;
    string token;
    istringstream tokenStream(s);
    while (getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

// Read csv files
MatrixXd readDataFromCSV(const string &filename){
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

    // Transform vector into eigen matrix, more complex to account for empty spaces in csv (beta-N csv)
    int rows = table.size();
    int cols = table.empty() ? 0 : table[0].size();
    MatrixXd m(rows-1, cols);

    for (int i = 0; i < rows - 1; ++i) {
        for (int j = 0; j < cols; ++j) {
            try {
                if (table[i + 1][j].empty()) {
                    m(i, j) = 0.0f; // If empty, set to 0
                } else {
                    m(i, j) = std::stof(table[i + 1][j]);
                }
            } catch (const std::invalid_argument& e) {
                std::cerr << "Invalid argument: " << e.what() << std::endl;
                exit(1);
            } catch (const std::out_of_range& e) {
                std::cerr << "Out of range: " << e.what() << std::endl;
                exit(1);
            }
        }
    }

    return m;
}

int main() {
    
    // plante après 4 itérations, est-ce lié à la remarque pour Simon? Mes betas ne convergent pas
    // todo: calcul de la pvalue

    // Du côté des nodes. Calcul des betas initiaux.
    int nbOfNodes = 3;
    vector<Node> nodesCalc;
    for (int i = 1; i <= nbOfNodes; ++i) {
        string filename = "Data_node_" + to_string(i) + ".csv";
        nodesCalc.push_back(Node(filename, i));
    }
    
    // Needed to initialise the size of BetaN
    string filename = "Data_Node_1_iter_0_output.csv";
    MatrixXd result = readDataFromCSV(filename);
    
    // Calcul du first beta (average) du coord. node
    MatrixXd betaN = MatrixXd::Zero(result.rows(), 1);
    int nTot = 0;
    for (int i = 1; i <= nbOfNodes; ++i) {
        string filename = "Data_Node_" + to_string(i) + "_iter_0_output.csv";
        MatrixXd result = readDataFromCSV(filename);
        
        betaN += result.col(0) * result(0,1);
        nTot += result(0,1);
    }
    MatrixXd beta = betaN.array()/nTot;

    // Save beta in a file
    filename = "Coord_node_iter_1_primer.csv";
    writeBetaToCSV(beta, filename);

    // Selon le nb d'aller retour entre coord et nodes
    int nbOfIte = 2;
    MatrixXd Vmatrix_inverse;
    for (int it = 1; it <= nbOfIte; ++it) {

        // Du côté des nodes - read beta, calcul de V et D, save in a csv file
        for (int i = 1; i <= nbOfNodes; ++i) {
            string filename = "Coord_node_iter_" + to_string(it) + "_primer.csv";
            nodesCalc[i-1].calculVD(filename, i, it);
        }

        // Read csv files produced by nodes and calcul of V and D global
        MatrixXd Dmatrix = MatrixXd::Zero(result.rows(), 1);
        MatrixXd Vmatrix = MatrixXd::Zero(result.rows(), result.rows());
        for (int i = 1; i <= nbOfNodes; ++i) {
            string filename = "Data_Node_" + to_string(i) + "_iter_" + to_string(it) + "_output.csv";
            MatrixXd result = readDataFromCSV(filename);

            Dmatrix += result.col(0);
            Vmatrix += result.block(0, 1, result.rows(), result.cols() - 1);
        }

        // Calcul du nouveau beta (Newton-Raphson)
        Vmatrix_inverse = Vmatrix.inverse();
        beta = beta + Vmatrix_inverse * Dmatrix;
        
        // Write new beta in csv file
        filename = "Coord_node_iter_" + to_string(it + 1) + "_primer.csv";
        writeBetaToCSV(beta, filename);
    }

    // Calculs finaux (int. de confiance + pvalue)
    VectorXd upper = beta + (stats::qnorm(1 - 0.05 / 2) * Vmatrix_inverse.diagonal().array().sqrt()).matrix();
    VectorXd lower = beta - (stats::qnorm(1 - 0.05 / 2) * Vmatrix_inverse.diagonal().array().sqrt()).matrix();
    //p_vals <- 2*(1 - pnorm(abs(beta)/sqrt(diag(Sigma))))
    // VectorXd p_value = 2*(1-stats::pnorm(beta.array().abs()/Vmatrix_inverse.diagonal().array().sqrt()));

    // WRITE
    vector<string> row_names = {"Intercept"};
    int num_cols = beta.rows() - 1;
    for (int i = 1; i <= num_cols; ++i) {
        row_names.push_back("Pred" + to_string(i));
    }

    MatrixXd reponse(beta.size(), 3);
    reponse.col(0) = beta;
    reponse.col(1) = lower;
    reponse.col(2) = upper;
    
    writeResultsToCSV("Distributed_results_centralised_logistic_reg.csv", row_names, reponse);

    return 0;
}
