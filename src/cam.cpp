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
#include <numeric>
#include <msgpack.hpp>
#include <string>
#include <iostream>
#include <sstream>
extern "C" {
#include "ccv.h"
}

/*
 * Config
 */
#define HAVE_OPENCV  /* this is must */
#define HAVE_DLIB
#define HAVE_CCV
#define HAVE_MXNET

#define INPUT_USE_OPENCV_CAMERA
//#define INPUT_USE_OPENCV_FILE
//#define DETECT_USE_OPENCV
//#define DETECT_USE_DLIB
#define DETECT_USE_CCV
#define LANDMARK_USE_DLIB
#define ALIGN_USE_OPENCV
#define FEATURE_USE_MXNET
#define DATABASE_USE_MSGPACK
#define CLASSIFIER_USE_NAIVE
#define DISPLAY_USE_OPENCV
//#define DISPLAY_USE_DLIB

#define DETECT_DOWNSAMPLE_RATIO     2
//#define FACE_USE_GRAY

using namespace std;

#ifdef ALIGN_USE_DLIB
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
#endif

#ifdef FEATURE_USE_MXNET
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

static PredictorHandle mx_setup(int &input_size, int &output_size)
{
    PredictorHandle predictor = 0;

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
                 &predictor);

    input_size = width * height * channels;
    mx_uint output_index = 0;
    mx_uint *shape = 0;
    mx_uint shape_len;
    //-- Get Output Size
    MXPredGetOutputShape(predictor, output_index, &shape, &shape_len);
    output_size = 1;
    for (mx_uint i = 0; i < shape_len; ++i)
        output_size *= shape[i];

    return predictor;
}

static void mx_cleanup(PredictorHandle predictor)
{
    // Release Predictor
    MXPredFree(predictor);
}

// return vector only in c++11
// convert form U8 to float element-wise
static std::vector<float> mx_get_image(const cv::Mat im) {
    int size = im.rows * im.cols;
#if 0
    cout << "im size = " << size << endl;
    cout << "im = "<< endl << " "  << im << endl << endl;
#endif
    std::vector<float> input_data (size);
    float *ptr_data = input_data.data();
    for (int i = 0; i < im.rows; i++) {
        for (int j = 0; j < im.cols; j++) {
            ptr_data[i * im.cols + j] = (float)*im.ptr<uchar>(i, j)/255.0;
        }
    }
    return input_data;
}

// return vector only in c++11
static std::vector<float> mx_forward(PredictorHandle predictor,
    std::vector<float> input_data, int input_size, int output_size)
{
    MXPredSetInput(predictor, "data", input_data.data(), input_size);
    MXPredForward(predictor);
    mx_uint output_index = 0;
    std::vector<float> output_data (output_size);
    MXPredGetOutput(predictor, output_index, output_data.data(), output_size);
    return output_data;
}
#endif

static std::vector<std::string> name_list;
static std::vector<std::vector<float> > rep_list;
static std::vector<std::vector<float> > rep_list_norm;

#ifdef DATABASE_USE_MSGPACK
static void calc_norm(std::vector<float> &in, std::vector<float> &out)
{
    double sum = 0.0f;
	for (std::vector<float>::iterator it = in.begin(); it != in.end(); ++it) {
        sum += pow(*it, 2);
    }
    double scale = 1/sqrt(sum);
	for (std::vector<float>::iterator it = in.begin(); it != in.end(); ++it) {
        sum += pow(*it, 2);
    }
    std::vector<float>::iterator it_out= out.begin();
	for (std::vector<float>::iterator it = in.begin(); it != in.end(); ++it) {
        *it_out = *it * scale;
        ++it_out;
    }
}

static void add_name_rep_double(std::string &name, std::vector<double> &rep_double)
{
    name_list.push_back(name);
    std::vector<float> rep(rep_double.begin(), rep_double.end());
    rep_list.push_back(rep);

	std::vector<float> rep_norm(256);
    calc_norm(rep, rep_norm);
    rep_list_norm.push_back(rep_norm);
}

static void load_face_db(void)
{
    std::ifstream file( "../data/feature/feature_db.mp" );
    std::stringstream buffer;
    buffer << file.rdbuf();
    file.close();

    // deserialize the buffer into msgpack::object instance.
    std::string str(buffer.str());

    msgpack::object_handle oh =
        msgpack::unpack(str.data(), str.size());

    // deserialized object is valid during the msgpack::object_handle instance is alive.
    msgpack::object obj = oh.get();

    // msgpack::object supports ostream.
    //std::cout << obj << std::endl;

    msgpack::object_kv*  pkv;
    msgpack::object_kv*  pkv_end;
    msgpack::object      pk, pv;

    msgpack::object_kv*  p2kv;
    msgpack::object_kv*  p2kv_end;
    msgpack::object      p2k, p2v;
    double *data_ptr;

    if(obj.via.map.size > 0)
    {
        pkv = obj.via.map.ptr;
        pkv_end = obj.via.map.ptr + obj.via.map.size;

        do
        {
            pk = pkv->key;
            pv = pkv->val;

            //cout << pk << ", " << pv << endl;

            ++pkv;

            if (pv.via.map.size > 0)
            {
                p2kv = pv.via.map.ptr;
                p2kv_end = pv.via.map.ptr + pv.via.map.size;

                do
                {
                    p2k = p2kv->key;
                    p2v = p2kv->val;

                    //cout << p2k << ", " << p2v << " size " << p2v.via.map.size << endl;
                    //if (p2v.via.map.size == 0x800)
                    if ( p2k.as<std::string>() == std::string("data") )
                    {
                        std::string name = pk.as<std::string>();
                        data_ptr = (double *)(p2v.via.map.ptr);
                        //cout << "name " << name << " rep " << data_ptr[0] << endl;
                        int size = p2v.via.map.size/sizeof(double);
                        std::vector<double> rep_double(data_ptr, data_ptr+size);
                        add_name_rep_double(name, rep_double);
                    }
                    ++p2kv;
                }
                while (p2kv < p2kv_end);
            }
        }
        while (pkv < pkv_end);
    }

    // convert msgpack::object instance into the original type.
    // if the type is mismatched, it throws msgpack::type_error exception.
    //msgpack::type::tuple<int, bool, std::string> dst;
    //deserialized.convert(dst);
}
#endif

#ifdef CLASSIFIER_USE_NAIVE
static double calc_dis(std::vector<float> &A, std::vector<float> &B)
{
    double sum = 0.0f;
    std::vector<float>::iterator it_B = B.begin();
    for (std::vector<float>::iterator it_A = A.begin(); it_A != A.end(); ++it_A) {
        sum += (*it_A) * (*it_B);
        ++it_B;
    }
    return sum;
}

static double calc_norm_value(std::vector<float> &in)
{
    double sum = 0.0f;
	for (std::vector<float>::iterator it = in.begin(); it != in.end(); ++it) {
        sum += pow(*it, 2);
    }
    return sqrt(sum);
}

static int find_nearest_face(std::vector<float> &rep)
{
    int id = 0;
    double dis_min = 0.0f;
    int best_id = -1;

    for (std::vector< std::vector<float> >::iterator it = rep_list.begin(); it != rep_list.end(); ++it) {
#if 0
        cout << "face = " << it->size() << endl;
        for (std::vector<float>::iterator iit = it->begin(); iit != it->end(); ++iit) {
            cout << *iit << " ";
        }
        cout << endl;
#endif
        double dis = cv::norm( *it , rep , cv::NORM_L2);
        cout << "dis = " << dis << endl;

        if (best_id < 0 || dis < dis_min) {
            dis_min = dis;
            best_id = id;
        }
        id++;
    }
    return best_id;
}

static int find_nearest_face_norm(std::vector<float> &rep)
{
    int id = 0;
    double dis_max = 0.0f;
    int best_id = -1;

    // dis = np.dot(output[0], output[1])/np.linalg.norm(output[0])/np.linalg.norm(output[1])
    for (std::vector< std::vector<float> >::iterator it = rep_list_norm.begin(); it != rep_list_norm.end(); ++it) {
#if 0
        cout << "face_norm = " << it->size() << endl;
        for (std::vector<float>::iterator iit = it->begin(); iit != it->end(); ++iit) {
            cout << *iit << " ";
        }
        cout << endl;
#endif
        std::vector<float> output_norm(256);
        calc_norm(rep, output_norm);
        double dis = calc_dis(output_norm, *it);
        cout << "dis_1 = " << dis << endl;

        if (best_id < 0 || dis > dis_max) {
            dis_max = dis;
            best_id = id;
        }
        id++;
    }
    return best_id;
}

static int find_nearest_face_substract_norm(std::vector<float> &rep)
{
    int id = 0;
    double dis_min = 0.0f;
    int best_id = -1;

    std::vector<float> diff(256);
    // dist = np.linalg.norm(rep - clf.means_[maxI])
    for (std::vector< std::vector<float> >::iterator it = rep_list_norm.begin(); it != rep_list_norm.end(); ++it) {
        //diff = mx_output_data - *it;
        std::set_difference(
            rep.begin(), rep.end(),
            (*it).begin(), (*it).end(),
            std::back_inserter( diff ));
        double dis = calc_norm_value(diff);
        cout << "dis_2 = " << dis << endl;

        if (best_id < 0 || dis < dis_min) {
            dis_min = dis;
            best_id = id;
        }
        id++;
    }
    return best_id;
}
#endif

#ifdef DISPLAY_USE_OPENCV
void draw_polyline(cv::Mat &img, const dlib::full_object_detection& d, const int start, const int end, bool isClosed = false)
{
    std::vector <cv::Point> points;
    for (int i = start; i <= end; ++i)
    {
        points.push_back(cv::Point(d.part(i).x(), d.part(i).y()));
    }
    cv::polylines(img, points, isClosed, cv::Scalar(0,0,255), 1, 16);
}

void render_face(cv::Mat &img, const dlib::full_object_detection& d)
{
    DLIB_CASSERT
    (
     d.num_parts() == 68,
     "\n\t Invalid inputs were given to this function. "
     << "\n\t d.num_parts():  " << d.num_parts()
     );

    draw_polyline(img, d, 0, 16);           // Jaw line
    draw_polyline(img, d, 17, 21);          // Left eyebrow
    draw_polyline(img, d, 22, 26);          // Right eyebrow
    draw_polyline(img, d, 27, 30);          // Nose bridge
    draw_polyline(img, d, 30, 35, true);    // Lower nose
    draw_polyline(img, d, 36, 41, true);    // Left eye
    draw_polyline(img, d, 42, 47, true);    // Right Eye
    draw_polyline(img, d, 48, 59, true);    // Outer lip
    draw_polyline(img, d, 60, 67, true);    // Inner lip
}
#endif

int main(int argc, char** argv)
{
#ifdef INPUT_USE_OPENCV_CAMERA
    cv::VideoCapture cap(0);
    if ( !cap.isOpened() ) {
        cerr << "Unable to connect to camera" << endl;
        return 1;
    }
#elif defined(INPUT_USE_OPENCV_FILE)
#else
#error "No input defined"
#endif

#ifdef DISPLAY_USE_OPENCV
    cv::namedWindow( "Image" );
    cv::namedWindow( "Faces" );
#elif defined(DISPLAY_USE_DLIB)
    dlib::image_window win, win_faces;
#endif

#ifdef DETECT_USE_OPENCV
    cv::CascadeClassifier face_cascade;
    // Load the cascade
    if ( !face_cascade.load(argv[2]) ) {
        cerr << "Unable to Load CV face cascade" << endl;
        return 1;
    };
#elif defined(DETECT_USE_DLIB)
    // Load face detection and pose estimation models.
    dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
#elif defined(DETECT_USE_CCV)
    ccv_scd_classifier_cascade_t *cascade =
        ccv_scd_classifier_cascade_read(argv[2]);
#else
#error "No detect defined"
#endif

#ifdef LANDMARK_USE_DLIB
    dlib::shape_predictor pose_model;
    dlib::deserialize(argv[1]) >> pose_model;
#endif

#ifdef FEATURE_USE_MXNET
    int mx_input_size, mx_output_size;
    PredictorHandle mx_predictor = mx_setup(mx_input_size, mx_output_size);
#endif

#ifdef DATABASE_USE_MSGPACK
    load_face_db();
#endif

    // always pass frames in cv::Mat
    cv::Mat frame;
    cv::Mat frame_gray;
    clock_t start;
    int finish = 0;

    while(!finish) {
#ifdef INPUT_USE_OPENCV_CAMERA
        // Grab a frame
        cap >> frame;
        if (frame.empty()) {
            cerr << "No captured frame" << endl;
            break;
        }
#elif defined(INPUT_USE_OPENCV_FILE)
        frame = cv::imread(argv[3], CV_LOAD_IMAGE_COLOR);
#else
#error "No input defined"
#endif
        cout << "Input frame [" << frame.rows << "x" << frame.cols
             << "@" << frame.step[0]/frame.cols << "]" << endl;

        // get gray
        cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
        // always pass face bbs in vector of cv::Rect
        std::vector<cv::Rect> cv_faces;

#ifdef DETECT_USE_OPENCV
        cv::equalizeHist(frame_gray, frame_gray);

        // CV Detect faces
        start = clock();
        face_cascade.detectMultiScale(frame_gray, cv_faces,
            1.1, 2, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30));
        cout << "CV Detector took "
             << double(clock() - start) / CLOCKS_PER_SEC << " sec." << endl;
#elif defined(DETECT_USE_DLIB)
        dlib::cv_image<dlib::bgr_pixel> cimg(frame);
#ifdef DETECT_DOWNSAMPLE_RATIO
        cv::Mat frame_small;
        cv::resize(frame, frame_small, cv::Size(),
            1.0/DETECT_DOWNSAMPLE_RATIO, 1.0/DETECT_DOWNSAMPLE_RATIO);
        dlib::cv_image<dlib::bgr_pixel> cimg_small(frame_small);
#endif

        // Detect faces
        start = clock();
#ifdef DETECT_DOWNSAMPLE_RATIO
        std::vector<dlib::rectangle> faces_small = detector(cimg_small);
        std::vector<dlib::rectangle> faces;
        for (unsigned long i = 0; i < faces_small.size(); ++i) {
            dlib::rectangle r(
                (long)(faces_small[i].left() * DETECT_DOWNSAMPLE_RATIO),
                (long)(faces_small[i].top() * DETECT_DOWNSAMPLE_RATIO),
                (long)(faces_small[i].right() * DETECT_DOWNSAMPLE_RATIO),
                (long)(faces_small[i].bottom() * DETECT_DOWNSAMPLE_RATIO)
            );
            faces.push_back(r);
        }
#else
        std::vector<dlib::rectangle> faces = detector(cimg);
#endif
        cout << "DLIB Detector took "
             << double(clock() - start) / CLOCKS_PER_SEC << " sec." << endl;

        // cout << "DLIB Detect " << faces.size() << " faces" << endl;
        // convert to cv_faces
        for (unsigned long i = 0; i < faces.size(); ++i) {
            //cout << "  DLIB Face " << i << " ["
            //     << faces[i].left() << ", "
            //     << faces[i].top() << ", "
            //     << faces[i].right() << ", "
            //     << faces[i].bottom() << "]" << endl;
            cv::Rect cv_face(
                faces[i].left(), faces[i].top(),
                faces[i].right() - faces[i].left(),
                faces[i].bottom() - faces[i].top()
            );
            cv_faces.push_back(cv_face);
        }
#elif defined(DETECT_USE_CCV)
        ccv_dense_matrix_t* ccv_image = 0;
        ccv_read(frame.data, &ccv_image,
            //CCV_IO_ANY_RAW | CCV_IO_BGR_RAW,
            CCV_IO_BGR_RAW | CCV_IO_ANY_RAW | CCV_IO_GRAY,
            frame.rows, frame.cols, frame.step[0]);
        start = clock();
        ccv_array_t *ccv_faces = ccv_scd_detect_objects(ccv_image, &cascade,
            1, ccv_scd_default_params);
        cout << "CCV Detector took "
             << double(clock() - start) / CLOCKS_PER_SEC << " sec." << endl;

        //cout << "CCV Detect " << ccv_faces->rnum << " faces" << endl;
        // convert to cv_faces
        for (int i = 0; i < ccv_faces->rnum; i++)
        {
            ccv_comp_t* ccv_face = (ccv_comp_t*)ccv_array_get(ccv_faces, i);
            //cout << "  CCV Face " << i << " ["
            //     << ccv_face->rect.x << ", "
            //     << ccv_face->rect.y << ", "
            //     << ccv_face->rect.width << ", "
            //     << ccv_face->rect.height << "]" << endl;
            cv::Rect cv_face(
                ccv_face->rect.x,
                ccv_face->rect.y,
                ccv_face->rect.width,
                ccv_face->rect.height);
            cv_faces.push_back(cv_face);
        }
        ccv_array_free(ccv_faces);
        ccv_matrix_free(ccv_image);
#else
#error "No detect defined"
#endif
        cout << "Detect " << cv_faces.size() << " faces" << endl;
        for (unsigned long i = 0; i < cv_faces.size(); ++i) {
            cout << "  Face " << i << " ["
                 << cv_faces[i].x << ", "
                 << cv_faces[i].y << ", "
                 << cv_faces[i].x + cv_faces[i].width << ", "
                 << cv_faces[i].y + cv_faces[i].height << "]" << endl;
        }

#if !defined(DETECT_USE_DLIB) && defined(LANDMARK_USE_DLIB)
        // get dlib::cv_image
        dlib::cv_image<dlib::bgr_pixel> cimg(frame);
        // convert CV faces into dlib faces
        std::vector<dlib::rectangle> faces;
        for (unsigned long i = 0; i < cv_faces.size(); ++i) {
            dlib::rectangle r(
                (long)(cv_faces[i].x),
                (long)(cv_faces[i].y),
                (long)(cv_faces[i].x + cv_faces[i].width),
                (long)(cv_faces[i].y + cv_faces[i].height)
            );
            faces.push_back(r);
        }
#endif

#ifdef LANDMARK_USE_DLIB
        // Find the pose of each face.
        std::vector<dlib::full_object_detection> shapes;
        for (int i = 0; i < faces.size(); i++) {
            start = clock();
            dlib::full_object_detection landmarks = pose_model(cimg, faces[i]);
            cout << "Landmark face " << i << " took "
                << double(clock() - start) / CLOCKS_PER_SEC << " sec." << endl;
            shapes.push_back(landmarks);
        }

        if (shapes.size() == 0) {
            cout << "No landmark found" << endl;
        }
#endif

#ifdef ALIGN_USE_OPENCV
        std::vector<cv::Mat> faces_align;
        cv::Point2f srcTri[3], dstTri[3];
        // hard-code for 128x128, {36, 45, 33}
        dstTri[0] = cv::Point2f( 24.85209656,   21.66616631 );
        dstTri[1] = cv::Point2f( 100.97396851,  20.24590683 );
        dstTri[2] = cv::Point2f( 63.35371399,   65.84849548 );

        for (int i = 0; i < shapes.size(); i++) {
            // use AffineTransform
            srcTri[0] = cv::Point2f( shapes[i].part(36).x(), shapes[i].part(36).y() );
            srcTri[1] = cv::Point2f( shapes[i].part(45).x(), shapes[i].part(45).y() );
            srcTri[2] = cv::Point2f( shapes[i].part(33).x(), shapes[i].part(33).y() );
            //cout << "srcTri: ["
            //     << srcTri[0].x << ", " << srcTri[0].y << "], [ "
            //     << srcTri[1].x << ", " << srcTri[1].y << "], [ "
            //     << srcTri[2].x << ", " << srcTri[2].y << "]" << endl;

            start = clock();
            cv::Mat warp_mat = cv::getAffineTransform(srcTri, dstTri);
#ifdef FACE_USE_GRAY
            cv::Mat face_warp = cv::Mat::zeros( 128, 128, frame_gray.type() );
            cv::warpAffine(frame_gray, face_warp, warp_mat, face_warp.size() );
#else
            cv::Mat face_warp = cv::Mat::zeros( 128, 128,  frame.type() );
            cv::warpAffine(frame, face_warp, warp_mat, face_warp.size() );
#endif
            faces_align.push_back(face_warp);
            cout << "Align face " << i << " took "
                << double(clock() - start) / CLOCKS_PER_SEC << " sec." << endl;
        }
#endif

#ifdef FEATURE_USE_MXNET
        std::vector<std::vector<float> > face_reps;
        for (int i = 0; i < faces_align.size(); i++) {
            cv::Mat face_gray;
#ifdef FACE_USE_GRAY
            face_gray = faces_align[i];
#else
            cv::cvtColor(faces_align[i], face_gray, cv::COLOR_BGR2GRAY);
#endif
            std::vector<float> mx_input_data = mx_get_image(face_gray);
            start = clock();
            std::vector<float> mx_output_data = mx_forward(mx_predictor,
                mx_input_data, mx_input_size, mx_output_size);
            cout << "Forward face " << i << " took "
                << double(clock() - start) / CLOCKS_PER_SEC << " sec." << endl;
#if 0
            cout << "output size = " << mx_output_data.size() << endl;
            for (int j = 0; j < mx_output_data.size(); j++) {
                cout << mx_output_data[j] << " ";
            }
            cout << endl;
#endif
            face_reps.push_back(mx_output_data);
        }
#endif

#ifdef CLASSIFIER_USE_NAIVE
        std::vector<int> face_ids;
        std::vector<std::string> face_names;
        for (int i = 0; i < face_reps.size(); i++) {
            start = clock();
            int id0 = find_nearest_face(face_reps[i]);
            int id1 = find_nearest_face_norm(face_reps[i]);
            int id2 = find_nearest_face_substract_norm(face_reps[i]);  // no good
            int id = id1;
            cout << "Classifier face " << i << " took "
                << double(clock() - start) / CLOCKS_PER_SEC << " sec." << endl;
            cout << "Face " << i << " Best Match is " << id
                << " name " << name_list[id] << endl;
            face_ids.push_back(id);
            face_names.push_back(name_list[id]);
        }
#endif

#ifdef DISPLAY_USE_OPENCV
        start = clock();
        cv::Mat face_show;
        for (int i = 0; i < shapes.size(); i++) {
            render_face( frame, shapes[i] );
            cv::rectangle( frame, cv_faces[i], cv::Scalar(0,255,0), 1, 16 );
            cv::Mat face_align = faces_align[i];
            cv::putText( face_align, face_names[i], cv::Point(5, 80),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255,255,0), 2 );
            if (i==0) {
                face_show = face_align;
            } else {
                cv::vconcat(face_show, face_align, face_show);
            }
        }
        cv::imshow( "Image", frame );
        if (shapes.size())
            cv::imshow( "Faces", face_show );
        // waitKey is must for display
        int key = cv::waitKey(30) & 0xFF;  // trigger display and hold for 30ms
        //cout << "key = " << key << endl;
        if (key == 27 || key == 'q') {
            finish = 1;
        }
        cout << "CV Display took "
             << double(clock() - start) / CLOCKS_PER_SEC << " sec." << endl;
#elif defined(DISPLAY_USE_DLIB)
        start = clock();

        win.clear_overlay();
        win.set_image(cimg);
        win.add_overlay(faces);
        win.add_overlay(render_face_detections(shapes));
        // We can also extract copies of each face that are cropped, rotated upright,
        // and scaled to a standard size as shown here:
        dlib::array<dlib::array2d<dlib::rgb_pixel> > face_chips;
        dlib::extract_image_chips(cimg, dlib::get_face_chip_details(shapes), face_chips);
        win_faces.set_image(dlib::tile_images(face_chips));
        //dlib::cv_image<dlib::bgr_pixel> cimg_face_align(face_warp);
        //win_faces.set_image(cimg_face_align);
        for (int i = 0; i < face_ids.size(); i++) {
            const dlib::rectangle r;
            win_faces.add_overlay(r, dlib::rgb_pixel(255,0,0), face_names[i]);
        }
        if (win.is_closed() || win_faces.is_closed()) {
            finish = 1;
        }
        cout << "DLIB Display took "
             << double(clock() - start) / CLOCKS_PER_SEC << " sec." << endl;
#endif
    }

#ifdef DISPLAY_USE_OPENCV
    cv::destroyAllWindows();
#endif
#ifdef FEATURE_USE_MXNET
    mx_cleanup(mx_predictor);
#endif
#ifdef DETECT_USE_CCV
    ccv_scd_classifier_cascade_free(cascade);
#endif

    return 0;
}

