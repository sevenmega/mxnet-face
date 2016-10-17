#include <ctime>
#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>

using namespace dlib;
using namespace std;

// WR to speedup
#define FACE_DOWNSAMPLE_RATIO 2

int main(int argc, char** argv)
{
    try
    {
        cv::VideoCapture cap(0);
        if (!cap.isOpened())
        {
            cerr << "Unable to connect to camera" << endl;
            return 1;
        }

        image_window win;
        clock_t start;

        // Load face detection and pose estimation models.
        frontal_face_detector detector = get_frontal_face_detector();
        shape_predictor pose_model;
        deserialize(argv[1]) >> pose_model;

        // Grab and process frames until the main window is closed by the user.
        while(!win.is_closed())
        {
            // Grab a frame
            cv::Mat temp;
            cap >> temp;
            // Turn OpenCV's Mat into something dlib can deal with.  Note that this just
            // wraps the Mat object, it doesn't copy anything.  So cimg is only valid as
            // long as temp is valid.  Also don't do anything to temp that would cause it
            // to reallocate the memory which stores the image as that will make cimg
            // contain dangling pointers.  This basically means you shouldn't modify temp
            // while using cimg.
            cv_image<bgr_pixel> cimg(temp);
#ifdef FACE_DOWNSAMPLE_RATIO
            cv::Mat temp_small;
            cv::resize(temp, temp_small, cv::Size(),
                1.0/FACE_DOWNSAMPLE_RATIO, 1.0/FACE_DOWNSAMPLE_RATIO);
            cv_image<bgr_pixel> cimg_small(temp_small);
#endif

            // Detect faces
            start = clock();
#ifdef FACE_DOWNSAMPLE_RATIO
            std::vector<rectangle> faces_small = detector(cimg_small);
            std::vector<rectangle> faces;
            for (unsigned long i = 0; i < faces_small.size(); ++i) {
                rectangle r(
                    (long)(faces_small[i].left() * FACE_DOWNSAMPLE_RATIO),
                    (long)(faces_small[i].top() * FACE_DOWNSAMPLE_RATIO),
                    (long)(faces_small[i].right() * FACE_DOWNSAMPLE_RATIO),
                    (long)(faces_small[i].bottom() * FACE_DOWNSAMPLE_RATIO)
                );
                faces.push_back(r);
            }
#else
            std::vector<rectangle> faces = detector(cimg);
#endif
            cout << "Detector took " << double(clock() - start) / CLOCKS_PER_SEC << " sec." << endl;

            // Find the pose of each face.
            start = clock();
            std::vector<full_object_detection> shapes;
            for (unsigned long i = 0; i < faces.size(); ++i) {
                shapes.push_back(pose_model(cimg, faces[i]));
            }
            cout << "Predictor took " << double(clock() - start) / CLOCKS_PER_SEC << " sec." << endl;

            // Display it all on the screen
            start = clock();
            win.clear_overlay();
            win.set_image(cimg);
            win.add_overlay(faces);
            win.add_overlay(render_face_detections(shapes));
            cout << "Display took " << double(clock() - start) / CLOCKS_PER_SEC << " sec." << endl;
        }
    }
    catch(serialization_error& e)
    {
        cout << "You need dlib's default face landmarking model file to run this example." << endl;
        cout << "You can get it from the following URL: " << endl;
        cout << "   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << endl;
        cout << endl << e.what() << endl;
    }
    catch(exception& e)
    {
        cout << e.what() << endl;
    }
}

