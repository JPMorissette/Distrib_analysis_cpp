#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include "gcem.hpp"
#include "stats.hpp"
#include "Node.h"
#include "NodeInput.h"

using namespace std;
using namespace Eigen;

void writeResultsToCSV(const string& filename, const vector<string>& row_names, const MatrixXd& reponse) {
    ofstream outputFile(filename);
    if (outputFile.is_open()) {
        outputFile << "Variable,Beta,Upper,Lower" << endl;
        
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

int main() {
    // Nombre de nodes
    int nbOfNodes = 2;

    // Calculs au niveau des nodes. Crée un fichier csv par node qui contient xtx, yty, xty. Correspond aux données en entrée.
    vector<Node> nodesCalc;
    for (int i = 1; i <= nbOfNodes; ++i) {
        string filename = "Data_node_" + to_string(i) + ".csv";
        nodesCalc.push_back(Node(filename, i));
    }

    // Lecture des fichiers csv produits par les nodes. Stocke chaque valeur dans un objet NodeInput.
    vector<NodeInput> nodes;
    for (int i = 1; i <= nbOfNodes; ++i) {
        string filename = "Node" + to_string(i) + "_output.csv";
        nodes.push_back(NodeInput(filename));
    }

    // CALCULS
    int cols = nodes[0].xtx.cols();
    MatrixXd xtx = MatrixXd::Zero(cols, cols);
    MatrixXd xty = MatrixXd::Zero(cols, 1);
    MatrixXd yty = MatrixXd::Zero(1, 1);

    for (int i = 0; i < nbOfNodes; ++i) {
        xtx += nodes[i].xtx;
        xty += nodes[i].xty;
        yty += nodes[i].yty;
    }

    MatrixXd xtx_inverse = xtx.inverse();
    
    VectorXd beta = xtx_inverse * xty;
    MatrixXd varbeta = (1.0 / (xtx(0,0) - cols)) * ((yty - (beta.transpose() * xty))(0, 0)) * xtx_inverse;

    double t_value = stats::qt(1 - 0.05 / 2, xtx(0,0) - cols);
    VectorXd upper = beta + (t_value * varbeta.diagonal().array().sqrt()).matrix();
    VectorXd lower = beta - (t_value * varbeta.diagonal().array().sqrt()).matrix();

    // WRITE
    vector<string> row_names = {"Intercept"};
    int num_cols = cols - 1;
    for (int i = 1; i <= num_cols; ++i) {
        row_names.push_back("Pred" + to_string(i));
    }

    MatrixXd reponse(beta.size(), 3);
    reponse.col(0) = beta;
    reponse.col(1) = upper;
    reponse.col(2) = lower;
    
    writeResultsToCSV("Distributed_results_centralised_lin_reg.csv", row_names, reponse);

    return 0;
}
