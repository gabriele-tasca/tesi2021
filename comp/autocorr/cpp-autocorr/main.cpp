
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>

int main() {
    
    std::ifstream csv_file ("/home/gaboloth/D/fisica/tesi/dati/csvsquare/all/Narcanello-2d.csv");

    int xwid; 
    int ywid;
    char spam;
    csv_file >> spam >> xwid >> ywid;
    std::cout << xwid << " " << ywid << std::endl;

    // declare data array
    float z2d[xwid*ywid];

    
    for (int y=0; y<ywid; y++) { 
        for (int x=0; x<xwid; x++) {
            
            csv_file >> z2d[x + y*xwid];

        }
    }

    int rmax =  static_cast<int>(static_cast<float>(std::max(xwid, ywid))/2.0);
    std::cout << rmax; 

    int count = 0;
    float autocorrelations[rmax];
    float counts[rmax];
    for (int y1=0; y1<ywid; y1++) { 
        for (int x1=0; x1<xwid; x1++) {

            float fx1 = static_cast<float>(x1);
            float fy1 = static_cast<float>(y1);

            for (int y2=0; y2<ywid; y2++) { 
                for (int x2=0; x2<xwid; x2++) {

                    float fx2 = static_cast<float>(x2);
                    float fy2 = static_cast<float>(y2);
                    float distance2 = sqrt( pow( (x1 - x2),2) + pow( (y1 - y2),2) );
                    float product = z2d[x1 + xwid*y1] *  z2d[x2 + xwid*y2];
                }
            }
            count++;
            std::cout << " inner cycle " << count << std::endl;

        }
    }
    

    return 0;
}