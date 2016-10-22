#include <iostream>
#include <string>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;

int main(int argc, char** argv)
{
    cv::VideoCapture cap(0);
    if ( !cap.isOpened() ) {
        cerr << "Unable to connect to camera" << endl;
        return 1;
    }

    cv::namedWindow( "Image" );

    cv::Mat frame;
    int finish = 0;

    while(!finish) {
        // Grab a frame
        cap >> frame;
        if (frame.empty()) {
            cerr << "No captured frame" << endl;
            break;
        }
        cv::imshow( "Image", frame );
        int key = cv::waitKey(30) & 0xFF;
        if (key == 27 || key == 'q') {
            finish = 1;
        }
    }
    cv::destroyAllWindows();
    return 0;
}

