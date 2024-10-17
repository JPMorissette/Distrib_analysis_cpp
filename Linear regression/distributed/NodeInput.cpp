#include "NodeInput.h"
#include <cmath>
#include <stats.hpp>
#include <iostream>
#include <fstream>
#include <string>

// Constructor
NodeInput::NodeInput(const string &filename){
    readDataFromCSV(filename);
}

// Read data and store in object attributes
void NodeInput::readDataFromCSV(const string &filename) {
    ifstream file(filename);

    if (file.is_open()) {
        vector<double> xtx_data, yty_data, xty_data;
        string line;
        getline(file, line);
        while (getline(file, line)) {
            stringstream ss(line);
            string cell;
            getline(ss, cell, ',');
            xtx_data.push_back(stod(cell));
            getline(ss, cell, ',');
            if (!cell.empty())
                yty_data.push_back(stod(cell));
            getline(ss, cell, ',');
            if (!cell.empty())
                xty_data.push_back(stod(cell));
        }

        // Transform vectors into eigen matrix
        int size = sqrt(xtx_data.size());
        xtx.resize(size, size);
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                xtx(i, j) = xtx_data[i * size + j];
            }
        }

        yty.resize(yty_data.size(), 1);
        for (int i = 0; i < yty_data.size(); ++i)
            yty(i, 0) = yty_data[i];

        xty.resize(xty_data.size(), 1);
        for (int i = 0; i < xty_data.size(); ++i)
            xty(i, 0) = xty_data[i];

        cout << "Data read from " << filename << " successfully." << endl;
        file.close();
    } else {
        cerr << "Unable to open file: " << filename << endl;
    }
}

