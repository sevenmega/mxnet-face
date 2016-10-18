#include <ctime>
#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/objdetect/objdetect.hpp"
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <mxnet/c_predict_api.h>

using namespace dlib;
using namespace std;

static float face_align_template[][2] = {
    {0.0792396913815, 0.339223741112}, {0.0829219487236, 0.456955367943},
    {0.0967927109165, 0.575648016728}, {0.122141515615,  0.691921601066},
    {0.168687863544,  0.800341263616}, {0.239789390707,  0.895732504778},
    {0.325662452515,  0.977068762493}, {0.422318282013,  1.04329000149 },
    {0.531777802068,  1.06080371126 }, {0.641296298053,  1.03981924107 },
    {0.738105872266,  0.972268833998}, {0.824444363295,  0.889624082279},
    {0.894792677532,  0.792494155836}, {0.939395486253,  0.681546643421},
    {0.96111933829,   0.562238253072}, {0.970579841181,  0.441758925744},
    {0.971193274221,  0.322118743967}, {0.163846223133,  0.249151738053},
    {0.21780354657,   0.204255863861}, {0.291299351124,  0.192367318323},
    {0.367460241458,  0.203582210627}, {0.4392945113,    0.233135599851},
    {0.586445962425,  0.228141644834}, {0.660152671635,  0.195923841854},
    {0.737466449096,  0.182360984545}, {0.813236546239,  0.192828009114},
    {0.8707571886,    0.235293377042}, {0.51534533827,   0.31863546193 },
    {0.516221448289,  0.396200446263}, {0.517118861835,  0.473797687758},
    {0.51816430343,   0.553157797772}, {0.433701156035,  0.604054457668},
    {0.475501237769,  0.62076344024 }, {0.520712933176,  0.634268222208},
    {0.565874114041,  0.618796581487}, {0.607054002672,  0.60157671656 },
    {0.252418718401,  0.331052263829}, {0.298663015648,  0.302646354002},
    {0.355749724218,  0.303020650651}, {0.403718978315,  0.33867711083 },
    {0.352507175597,  0.349987615384}, {0.296791759886,  0.350478978225},
    {0.631326076346,  0.334136672344}, {0.679073381078,  0.29645404267 },
    {0.73597236153,   0.294721285802}, {0.782865376271,  0.321305281656},
    {0.740312274764,  0.341849376713}, {0.68499850091,   0.343734332172},
    {0.353167761422,  0.746189164237}, {0.414587777921,  0.719053835073},
    {0.477677654595,  0.706835892494}, {0.522732900812,  0.717092275768},
    {0.569832064287,  0.705414478982}, {0.635195811927,  0.71565572516 },
    {0.69951672331,   0.739419187253}, {0.639447159575,  0.805236879972},
    {0.576410514055,  0.835436670169}, {0.525398405766,  0.841706377792},
    {0.47641545769,   0.837505914975}, {0.41379548902,   0.810045601727},
    {0.380084785646,  0.749979603086}, {0.477955996282,  0.74513234612 },
    {0.523389793327,  0.748924302636}, {0.571057789237,  0.74332894691 },
    {0.672409137852,  0.744177032192}, {0.572539621444,  0.776609286626},
    {0.5240106503,    0.783370783245}, {0.477561227414,  0.778476346951}
    };
static int INNER_EYES_AND_BOTTOM_LIP[] = {39, 42, 57};
static int OUTER_EYES_AND_NOSE[] = {36, 45, 33};

// WR to speedup
#define FACE_DOWNSAMPLE_RATIO 2
// toggle to use CV Face Detection
// around x2 the speed, however, with a lot more false positive
//#define USE_CV_FACE_DETECTION
#define USE_DLIB_FACE_DETECTION

// Read file to buffer
class BufferFile {
 public :
    std::string file_path_;
    int length_;
    char* buffer_;

    explicit BufferFile(std::string file_path)
    :file_path_(file_path) {

        std::ifstream ifs(file_path.c_str(), std::ios::in | std::ios::binary);
        if (!ifs) {
            std::cerr << "Can't open the file. Please check " << file_path << ". \n";
            assert(false);
        }

        ifs.seekg(0, std::ios::end);
        length_ = ifs.tellg();
        ifs.seekg(0, std::ios::beg);
        std::cout << file_path.c_str() << " ... "<< length_ << " bytes\n";

        buffer_ = new char[sizeof(char) * length_];
        ifs.read(buffer_, length_);
        ifs.close();
    }

    int GetLength() {
        return length_;
    }
    char* GetBuffer() {
        return buffer_;
    }

    ~BufferFile() {
        delete[] buffer_;
        buffer_ = NULL;
    }
};

static PredictorHandle mx_predictor = 0;
static int mx_input_size = 0;
static int mx_output_size = 0;
static std::vector<mx_float> mx_input_data;
static std::vector<float> mx_output_data;

static void mx_setup(void)
{
    // Models path for your model, you have to modify it
    BufferFile json_data("../model/lightened_cnn/lightened_cnn-symbol.json");
    BufferFile param_data("../model/lightened_cnn/lightened_cnn-0166_cpu.params");

    // Parameters
    int dev_type = 1;  // 1: cpu, 2: gpu
    int dev_id = 0;  // arbitrary.
    mx_uint num_input_nodes = 1;  // 1 for feedforward
    const char* input_key[1] = {"data"};
    const char** input_keys = input_key;
    mx_uint num_output_nodes = 1;
    const char* output_key[1] = {"drop1"};
    const char** output_keys = output_key;

    // Image size and channels
    int width = 128;
    int height = 128;
    int channels = 1;

    const mx_uint input_shape_indptr[2] = { 0, 4 };
    // ( trained_width, trained_height, channel, num)
    const mx_uint input_shape_data[4] = { 1,
                                        static_cast<mx_uint>(channels),
                                        static_cast<mx_uint>(width),
                                        static_cast<mx_uint>(height) };

    //-- Create Predictor
    MXPredCreatePartialOut((const char*)json_data.GetBuffer(),
                 (const char*)param_data.GetBuffer(),
                 static_cast<size_t>(param_data.GetLength()),
                 dev_type,
                 dev_id,
                 num_input_nodes,
                 input_keys,
                 input_shape_indptr,
                 input_shape_data,
                 num_output_nodes,
                 output_keys,
                 &mx_predictor);

    mx_input_size = width * height * channels;
    mx_input_data = std::vector<mx_float>(mx_input_size);

    mx_uint output_index = 0;

    mx_uint *shape = 0;
    mx_uint shape_len;

    //-- Get Output Size
    MXPredGetOutputShape(mx_predictor, output_index, &shape, &shape_len);

    mx_output_size = 1;
    for (mx_uint i = 0; i < shape_len; ++i)
        mx_output_size *= shape[i];
    mx_output_data = std::vector<float>(mx_output_size);
}

static void mx_cleanup(void)
{
    // Release Predictor
    MXPredFree(mx_predictor);
}

static void GetImageFile(const std::string image_file, mx_float* image_data) {
#if 1
    cv::Mat im = cv::imread(image_file, 0);  // 0 is Gray
#else
    cv::Mat im_color = cv::imread(image_file, 1);
    cv::Mat im;
    cv::cvtColor(im_color, im, cv::COLOR_BGR2GRAY);
#endif

    if (im.empty()) {
        std::cerr << "Can't open the image. Please check " << image_file << ". \n";
        assert(false);
    }

    int size = im.rows * im.cols;

#if 0
    cout << "image size = " << size << endl;
    cout << "im = "<< endl << " "  << im << endl << endl;
#endif

    mx_float* ptr_image = image_data;

    for (int i = 0; i < im.rows; i++) {
        for (int j = 0; j < im.cols; j++) {
            image_data[i * im.cols + j] = *im.ptr<uchar>(i, j)/255.0;
        }
    }
}

static void GetImage(const cv::Mat im_color, mx_float* image_data) {
    cv::Mat im;
    cv::cvtColor(im_color, im, cv::COLOR_BGR2GRAY);

    int size = im.rows * im.cols;

#if 0
    cout << "image size = " << size << endl;
    cout << "im = "<< endl << " "  << im << endl << endl;
#endif

    mx_float* ptr_image = image_data;

    for (int i = 0; i < im.rows; i++) {
        for (int j = 0; j < im.cols; j++) {
            image_data[i * im.cols + j] = *im.ptr<uchar>(i, j)/255.0;
        }
    }
}

static void mx_forward(void)
{
    //-- Set Input Image
    MXPredSetInput(mx_predictor, "data", mx_input_data.data(), mx_input_size);

    //-- Do Predict Forward
    MXPredForward(mx_predictor);

    //-- Get Output Result
    mx_uint output_index = 0;

    MXPredGetOutput(mx_predictor, output_index, mx_output_data.data(), mx_output_size);
}

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

        image_window win, win_faces;
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

        mx_setup();

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
            cout << "CV Detector took "
                << double(clock() - start) / CLOCKS_PER_SEC << " sec." << endl;

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
            cout << "Detector took "
                << double(clock() - start) / CLOCKS_PER_SEC << " sec." << endl;

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
                full_object_detection landmarks = pose_model(cimg, faces[i]);
                cout << "Landmark num_parts " << landmarks.num_parts() << endl;
                shapes.push_back(landmarks);
            }
            cout << "Predictor took "
               << double(clock() - start) / CLOCKS_PER_SEC << " sec." << endl;

            // Display it all on the screen
            start = clock();
            win.clear_overlay();
            win.set_image(cimg);
            win.add_overlay(faces);
            win.add_overlay(render_face_detections(shapes));
            cout << "Display took "
                << double(clock() - start) / CLOCKS_PER_SEC << " sec." << endl;

            if (shapes.size() == 0) {
                continue;
            }
#if 0
            // We can also extract copies of each face that are cropped, rotated upright,
            // and scaled to a standard size as shown here:
            dlib::array<array2d<rgb_pixel> > face_chips;
            extract_image_chips(cimg, get_face_chip_details(shapes), face_chips);
            win_faces.set_image(tile_images(face_chips));
#else
            // use AffineTransform
            cv::Point2f srcTri[3];
            cv::Point2f dstTri[3];

            srcTri[0] = cv::Point2f( shapes[0].part(36).x(), shapes[0].part(36).y() );
            srcTri[1] = cv::Point2f( shapes[0].part(45).x(), shapes[0].part(45).y() );
            srcTri[2] = cv::Point2f( shapes[0].part(33).x(), shapes[0].part(33).y() );
            cout << "srcTri: ["
                << srcTri[0].x << ", " << srcTri[0].y << "], [ "
                << srcTri[1].x << ", " << srcTri[1].y << "], [ "
                << srcTri[2].x << ", " << srcTri[2].y << "]" << endl;
            // hard-code for 128x128, {36, 45, 33}
            dstTri[0] = cv::Point2f( 24.85209656,   21.66616631 );
            dstTri[1] = cv::Point2f( 100.97396851,  20.24590683 );
            dstTri[2] = cv::Point2f( 63.35371399,   65.84849548 );
            cv::Mat warp_mat = cv::getAffineTransform(srcTri, dstTri);
            cv::Mat face_warp = cv::Mat::zeros( 128, 128,  frame.type() );
            cv::warpAffine(frame, face_warp, warp_mat, face_warp.size() );
            cv_image<bgr_pixel> cimg_face_align(face_warp);
            win_faces.set_image(cimg_face_align);
#endif

            GetImage(face_warp, mx_input_data.data());
#if 0
            //-- Read Image Data from file
            std::string test_file = std::string("../data/my-align/larry/image-10.png");
            GetImageFile(test_file, mx_input_data.data());
#endif
            mx_forward();
#if 0
            cout << "output size = " << mx_output_data.size() << endl;
            for (int i = 0; i < mx_output_data.size(); i++) {
                cout << mx_output_data[i] << " ";
            }
            cout << endl;
#endif
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
    mx_cleanup();
}

