#include <ctime>
#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/objdetect/objdetect.hpp"
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>

using namespace dlib;
using namespace std;

// WR to speedup
#define FACE_DOWNSAMPLE_RATIO 2
// toggle to use CV Face Detection
// around x2 the speed, however, with a lot more false positive
//#define USE_CV_FACE_DETECTION
#define USE_DLIB_FACE_DETECTION

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
        cv::Mat frame;

#ifdef USE_CV_FACE_DETECTION
        cv::CascadeClassifier face_cascade;

        // Load the cascade
        if (!face_cascade.load(argv[2])) {
            cerr << "Unable to Load CV face cascade" << endl;
            return 1;
        };
#endif

#ifdef USE_DLIB_FACE_DETECTION
        // Load face detection and pose estimation models.
        frontal_face_detector detector = get_frontal_face_detector();
#endif
        shape_predictor pose_model;
        deserialize(argv[1]) >> pose_model;

        // Grab and process frames until the main window is closed by the user.
        while(!win.is_closed())
        {
            // Grab a frame
            cap >> frame;
            if (frame.empty()) {
                cerr << "No captured frame" << endl;
                break;
            }

#ifdef USE_CV_FACE_DETECTION
            std::vector<cv::Rect> cv_faces;
            cv::Mat frame_gray;
            cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
            cv::equalizeHist(frame_gray, frame_gray);

            // CV Detect faces
            start = clock();
            face_cascade.detectMultiScale(frame_gray, cv_faces,
                1.1, 2, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30));
            cout << "CV Detector took " << double(clock() - start) / CLOCKS_PER_SEC << " sec." << endl;

            cout << "CV Detect " << cv_faces.size() << " faces" << endl;
            for (unsigned long i = 0; i < cv_faces.size(); ++i) {
                cout << "  Face " << i << " ["
                     << cv_faces[i].x << ", "
                     << cv_faces[i].y << ", "
                     << cv_faces[i].x + cv_faces[i].width << ", "
                     << cv_faces[i].y + cv_faces[i].height << "]" << endl;
            }
#endif

            // Turn OpenCV's Mat into something dlib can deal with.  Note that this just
            // wraps the Mat object, it doesn't copy anything.  So cimg is only valid as
            // long as frame is valid.  Also don't do anything to frame that would cause it
            // to reallocate the memory which stores the image as that will make cimg
            // contain dangling pointers.  This basically means you shouldn't modify frame
            // while using cimg.
            cv_image<bgr_pixel> cimg(frame);
#ifdef USE_DLIB_FACE_DETECTION
#ifdef FACE_DOWNSAMPLE_RATIO
            cv::Mat frame_small;
            cv::resize(frame, frame_small, cv::Size(),
                1.0/FACE_DOWNSAMPLE_RATIO, 1.0/FACE_DOWNSAMPLE_RATIO);
            cv_image<bgr_pixel> cimg_small(frame_small);
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

            cout << "Detect " << faces.size() << " faces" << endl;
            for (unsigned long i = 0; i < faces.size(); ++i) {
                cout << "  Face " << i << " ["
                     << faces[i].left() << ", "
                     << faces[i].top() << ", "
                     << faces[i].right() << ", "
                     << faces[i].bottom() << "]" << endl;
            }
#else /* not USE_DLIB_FACE_DETECTION */
            // convert CV faces into dlib faces
            std::vector<rectangle> faces;
            for (unsigned long i = 0; i < cv_faces.size(); ++i) {
                rectangle r(
                    (long)(cv_faces[i].x),
                    (long)(cv_faces[i].y),
                    (long)(cv_faces[i].x + cv_faces[i].width),
                    (long)(cv_faces[i].y + cv_faces[i].height)
                );
                faces.push_back(r);
            }
#endif

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

