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

#define USE_CV_DISPLAY
//#define USE_DLIB_DISPLAY

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

// clapton
static float face_0[] = {
 -2.08490801,  0.32610536, -3.00524211, -0.27926248, -1.63280988,  3.53943563,
  2.76036835,  1.94319832,  1.27122879,  0.87835449, -0.57294291, -0.95302075,
 -1.65047002, -2.44099522,  2.53915262,  0.1782926 ,  0.69484913,  1.74678087,
 -0.57080054,  0.34881702,  4.67621136, -2.17780447, -4.79837465, -0.64135623,
  1.08822441,  5.98691273, -2.1218133 , -3.24501348, -0.35389686, -0.26517099,
 -0.23763637, -0.29417944, -3.73062658, -1.5633508 ,  1.37613571, -0.69218177,
  0.39765531,  1.41823697,  6.67447853,  1.36243451, -1.60042822, -2.31742954,
  3.03940606,  0.39249474,  0.92525637, -3.00282884, -1.6557827 ,  0.21680425,
 -1.40244639,  1.15771198,  0.57471776, -2.03251743, -0.18429792, -1.11193752,
 -3.6900897 , -0.42318344,  0.77306539,  0.18518028, -3.8874836 ,  0.80264747,
 -3.40757036, -1.89626813,  1.49638498,  1.29540658,  0.14980197,  0.89455497,
  1.2807672 , -2.02509689,  1.31428838, -2.95257878, -0.45121118, -0.51760823,
 -2.48558736,  2.76907253, -1.21771991, -2.58135128, -0.52703804,  0.45528018,
  5.65469217,  0.83474106, -1.90428376,  1.5175401 ,  5.38463926, -0.87673318,
  0.70441735,  1.38026059, -2.77842188, -1.10586452,  1.65322149, -0.17740092,
 -0.67700797, -3.09355569, -2.40529871, -2.26281357,  3.96753192, -0.44836584,
 -0.82445377,  0.24015234,  0.93986768, -1.57125139, -1.7579149 , -1.5084486 ,
 -0.72154868,  0.13495338,  4.22349977,  0.70190054, -0.61139011, -2.98568106,
 -3.58896613, -0.48711255, -0.75996584, -0.41011861, -1.11567211, -0.85553724,
  3.09034801, -2.85003805,  0.13948467,  2.83420873,  2.4701364 ,  2.20275807,
 -2.10100484,  0.71348161,  3.27471018,  0.65061545, -1.90934491, -0.18615976,
 -0.5816226 ,  7.50447178, -4.25202847,  0.46764487,  0.96505296, -0.65498507,
  6.83070946,  2.59572339,  1.75672591, -5.2900629 ,  0.8490085 ,  0.11337809,
  0.43284589,  2.87631178, -0.08040479, -1.13699484,  1.16144514, -3.36553049,
 -1.57967067,  1.37424541, -2.16314912, -3.99591684, -0.07387669,  0.23053715,
 -2.0277338 ,  1.28529942,  3.00490046,  1.03512561, -3.33180976,  5.55180645,
 -0.70154148,  1.37254238, -2.73921418, -1.76555479,  0.33173612,  0.18613051,
  3.20832443,  0.9995842 , -0.63536429, -2.39676857, -1.0793221 ,  1.92625284,
  0.81928134,  0.36294597, -0.88734871,  2.95083237,  2.51059985,  0.10461915,
 -3.35733008,  2.39820051,  3.25534916, -3.36600757,  0.91198981, -0.2320392 ,
 -0.56751341, -2.42398691,  3.26838994,  2.7308743 ,  4.45278406, -0.25010791,
 -0.2201445 , -0.32150617, -0.58727145,  1.84273028,  0.92047888, -0.05487311,
  2.49427819, -2.65924978,  2.68576002, -0.60425717, -0.634588  ,  3.11469245,
 -3.25098038, -2.75686646,  1.53631282, -0.20252278,  1.6682328 ,  0.77590895,
 -1.7410996 ,  5.36573648, -0.64414227, -0.76722199, -0.12974343, -0.65861404,
 -2.82593894,  1.37612784, -1.78943121, -1.30376029,  4.60373735, -1.06223273,
 -2.11250329,  0.5825296 ,  3.402812  , -0.6509034 ,  2.96956015,  0.27779946,
 -2.75657797,  2.77492142, -0.17212923,  0.40471444, -0.83278513, -1.40083981,
 -0.88022566,  3.29016232, -1.88341856, -1.70517433, -4.29120874, -0.4144415 ,
 -3.76374555,  0.09155056,  0.42873064, -0.80744779, -1.58663762,  1.99392009,
 -0.52519113,  0.14921844,  0.8415885 ,  1.91287518,  2.67854357,  0.37166613,
 -2.73627496, -0.26684597,  0.05675699,  0.59855688,  1.8693676 , -1.73460782,
  1.23114061,  2.59832406, -0.12608398, -2.10600042};

// lennon
static float face_1[] = {
 -2.28701806, -0.92195076, -1.02591121, -2.776227  ,  0.45085165, -0.75832176,
  0.69099724,  1.88924086,  0.82209069, -0.2514486 , -3.94417405, -0.12126279,
  0.19122636, -4.1079874 ,  1.46222973, -0.518471  ,  1.48726428,  0.05272916,
 -1.19861186, -2.21243   , -0.67990106, -4.02502251,  1.07672596,  0.19440815,
  2.01730275,  2.47030902, -1.63816059,  0.21055368,  0.9867897 , -3.74596214,
 -0.05194803,  3.71783757, -2.48898745,  0.74697387,  1.78985023, -0.03611015,
 -0.58194631,  0.57680714,  0.13702875,  1.23593199, -1.73876989, -5.5212841 ,
  6.86666679,  2.38676929, -0.74429846, -0.89391679, -1.47499037,  0.26756403,
  0.89743263,  0.54498762, -0.72338426, -2.96633482,  1.70078957, -0.0562173 ,
  1.69401395,  0.44377288,  0.07858697,  2.7280426 , -2.34138179, -0.27984825,
 -2.04357433,  1.6444484 ,  0.39764491, -0.97191703, -0.01007709,  1.05316699,
 -1.15629613,  1.1659807 ,  2.18705893, -3.46558309,  3.54437351,  0.43311441,
 -2.2417922 , -1.54937577, -3.63010883, -2.88203478,  0.92706525,  1.34969783,
 -1.4687326 ,  0.72374976, -2.14361453,  1.95054781,  2.8911376 , -1.27469897,
  1.06515276, -2.45210409,  0.026438  , -1.90904856,  2.57594037, -2.47498965,
 -0.7892164 , -1.39096451, -0.72834194,  0.09202421, -1.14996517, -0.48925543,
 -0.21931927, -1.03408992,  3.82661557, -2.52285314,  0.89583933, -2.50193977,
  0.16320047, -0.29452229,  3.20302653,  1.20936966,  0.59304571,  2.2341969 ,
 -0.40079284,  1.09360337,  2.4123826 , -0.65465462,  0.38456705, -0.58743125,
  3.61263776,  3.91512918, -2.42498994,  2.71718311,  1.09920347,  1.78623343,
 -0.36959335,  2.73924875,  2.24352026, -3.12888932, -1.17973959, -1.1910677 ,
  1.22996283,  5.14543438, -3.12217474, -1.70772743, -1.11623025, -0.7343992 ,
 -1.03397703,  2.60965991,  2.82613206, -4.04577923, -1.90314817, -1.94321406,
 -1.26708567, -2.13651538,  1.53648412,  1.68362069,  0.21427965,  1.10590172,
 -1.83917165,  0.89477575, -0.88490862, -2.66050982, -1.69241798,  1.28992212,
 -1.52874184, -2.48721027,  0.59144056, -0.6931591 , -1.46265614,  1.49176908,
 -0.16358316,  0.14201121, -2.70507407, -0.24779081, -0.74997258,  0.9676317 ,
 -0.06238723,  3.20428848, -0.16060752, -1.63331711,  1.88437915,  0.2087263 ,
 -1.57390738,  1.62735224,  0.00707655,  2.66769648, -0.74623811, -2.7124126 ,
 -0.81723487, -0.68488622, -1.3337872 , -1.90776837,  0.67871511,  0.27384573,
 -1.18797588, -4.44696379,  1.16597235,  2.13851643,  1.82935107, -2.59690619,
 -3.6642766 ,  1.49310303, -2.98352814, -1.90341961, -2.23496604,  3.84677505,
  2.00209761, -4.37277174,  3.24249911,  0.27949578,  0.90778232,  3.3862133 ,
 -2.57509041, -0.33039522, -0.30550268,  1.01116443,  3.84858918,  0.32402843,
 -3.10823846,  3.27931762, -1.91556656, -0.31702974, -0.79232866, -1.56933963,
 -2.54087543, -0.97416174,  1.38869691, -3.54760742,  1.91227031,  1.7988112 ,
  1.84893274, -1.56415045,  1.60334265,  2.2681396 ,  2.04333425, -0.57454455,
 -1.97869074,  2.25210333, -0.48192966,  0.75409454,  1.86444116, -0.07089809,
 -0.52696139,  2.50870228,  0.49434602, -1.85691869, -1.12871957, -0.74166983,
 -0.21072586, -0.62929082,  3.67410159,  1.43114579, -1.2732476 , -0.30762705,
  2.79313111,  5.70862913,  0.31131321, -0.73388529,  1.29480171, -0.41381273,
  0.62176955,  2.54269075,  1.27226734,  1.30276799,  0.9668566 ,  1.68098569,
  1.83290851,  0.41971231,  2.14605141,  0.23919784};

// larry
static float face_2[] = {
 -3.66827989, -1.14918566, -1.53059268, -0.57313567, -0.01271075, -3.54587555,
  0.12525059, -1.02816105, -1.61573648, -0.49629095,  2.51805782,  0.50748938,
 -2.01453733, -0.75496757, -0.54644769,  0.15509619,  1.82493401, -2.11173534,
  0.92033213,  1.85458422,  0.50416821,  1.21704185, -0.52944958, -0.5044536 ,
  1.55561459, -1.11102557, -1.51064062,  1.60676634, -2.12443948,  0.86228108,
  2.48120713, -0.34442991, -0.4281655 , -0.80411971, -2.37309861, -0.20412311,
  1.57549703, -1.18137002,  0.50981224, -1.13908434,  2.76346564,  0.41285259,
  4.14597702, -0.9929837 ,  1.63158333, -0.4295882 ,  2.10021186, -0.87730271,
 -3.74970818,  3.33026338,  1.32309747,  1.3101995 , -4.1629138 , -0.78471965,
 -1.16617775, -3.49914861, -2.36741447, -0.29205096,  0.50736845, -0.1176132 ,
 -1.43694341,  0.05667621,  1.67724431, -0.40789339,  0.18124589,  0.07374322,
 -0.77893126,  0.9701578 ,  0.01351878, -0.88110471,  1.47248554, -0.82773525,
  2.65295935, -0.1138244 ,  1.55530012,  1.27528501,  0.57953578,  1.24639332,
  0.10826236,  0.39784929, -0.06384483,  0.26598644,  1.63373315, -1.9234221 ,
  1.47553015, -3.13070822, -1.554811  ,  0.00874291, -1.10111666, -1.55337083,
  3.0455699 , -1.00961936,  2.04972219, -3.09851933, -0.71157491,  1.50916719,
 -1.76370716, -0.12552465, -2.27163768, -0.63121277,  0.03379773, -3.02989721,
  1.13495827,  3.65842843, -0.49564895,  0.48082948,  1.86655307,  0.37633455,
 -0.33966023,  1.0943836 , -1.47661984, -1.28605378, -1.55934346,  1.70353305,
  0.26583084, -1.12781572,  0.40197465,  1.16529107, -1.45445716, -3.93930793,
 -4.21830702,  1.10427952, -0.55534762, -3.77270031, -4.15401888,  1.27321243,
  0.41662192, -1.77038419, -4.23207188, -0.66560626, -0.63325477, -0.55612767,
  3.06384754,  0.67271   , -2.75549078, -4.28457594, -1.32847857, -0.34912288,
  0.04324873,  2.35485244,  1.42720044,  0.3817758 ,  1.27114129, -0.65255487,
  1.18198693,  1.18807805, -1.54789138,  0.22528622,  0.75869721, -0.25558662,
  0.25866577,  3.10223675,  3.77679157, -0.69645905,  1.38284636,  0.92152441,
  2.31247783, -0.33074927, -1.12032235,  0.97939038,  1.40573502,  2.89241147,
  0.59571713,  0.71763027, -1.02753711,  0.76413941,  0.41167599,  0.41713607,
 -0.84591031,  1.72047484, -0.8858788 , -0.08745509,  0.94625545, -3.80456948,
 -0.69370705,  1.21788132, -0.2333208 , -2.99710894, -0.94713938, -0.85553318,
  1.41001809, -0.778     , -1.42366791,  1.70191824,  3.66441131,  1.21904147,
 -0.14890453, -2.51297402, -3.21897864, -0.46842459, -0.48664284,  0.71721852,
  1.32021976, -0.50873411,  1.2234776 , -3.75562143,  1.58899164,  1.15810549,
 -1.86530817,  0.18837956, -0.3214561 ,  0.74272311,  2.09262228, -0.0067708 ,
 -0.98042685,  1.28808808, -1.45794809, -0.70586067,  0.098705  , -1.7467736 ,
 -2.47653818,  1.41501009, -1.44526124, -2.19952106,  2.09506059,  0.69984704,
  2.26488161, -2.29482007,  3.25995612, -0.14010394,  0.33578682,  0.1609247 ,
 -1.19248223,  4.2456522 ,  0.60974377, -0.84022027, -1.10662961, -0.35428166,
 -1.95127547, -0.01679459, -1.06145525, -1.58585346, -1.64490509,  1.95901179,
 -0.61004555, -3.0436337 ,  3.38588524,  1.84345877, -0.70952106,  5.2766943 ,
  0.4424395 ,  0.30676168, -0.28675437,  0.94779998,  0.40057695, -2.07611275,
 -0.33444577,  1.57623661, -2.46969652, -0.1470035 , -2.34852266, -0.7568711 ,
  0.0999296 ,  2.33184195,  3.65313482, -0.24678747};

static std::vector<std::vector<float> > rep_db;
static std::vector<std::vector<float> > rep_db_norm;

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

static void create_face_db(void)
{
	std::vector<float> tmp_0 (face_0, face_0 + 256 );
    rep_db.push_back(tmp_0);
	std::vector<float> tmp_1 (face_1, face_1 + 256 );
    rep_db.push_back(tmp_1);
	std::vector<float> tmp_2 (face_2, face_2 + 256 );
    rep_db.push_back(tmp_2);

	std::vector<float> tmp_norm_0(256);
    calc_norm(tmp_0, tmp_norm_0);
    rep_db_norm.push_back(tmp_norm_0);
	std::vector<float> tmp_norm_1(256);
    calc_norm(tmp_1, tmp_norm_1);
    rep_db_norm.push_back(tmp_norm_1);
	std::vector<float> tmp_norm_2(256);
    calc_norm(tmp_2, tmp_norm_2);
    rep_db_norm.push_back(tmp_norm_2);
}

static std::string name[] = {
    "Clapton", "lennon", "Larry"};

static int find_nearest_face(void)
{
    int id = 0;
    double dis_min = 0.0f;
    int best_id = -1;

    for (std::vector< std::vector<float> >::iterator it = rep_db.begin(); it != rep_db.end(); ++it) {
#if 0
        cout << "face = " << it->size() << endl;
        for (std::vector<float>::iterator iit = it->begin(); iit != it->end(); ++iit) {
            cout << *iit << " ";
        }
        cout << endl;
#endif
        double dis = cv::norm( *it , mx_output_data , cv::NORM_L2);
        cout << "dis = " << dis << endl;

        if (best_id < 0 || dis < dis_min) {
            dis_min = dis;
            best_id = id;
        }
        id++;
    }
    return best_id;
}

static int find_nearest_face_norm(void)
{
    int id = 0;
    double dis_max = 0.0f;
    int best_id = -1;

    // dis = np.dot(output[0], output[1])/np.linalg.norm(output[0])/np.linalg.norm(output[1])
    for (std::vector< std::vector<float> >::iterator it = rep_db_norm.begin(); it != rep_db_norm.end(); ++it) {
#if 0
        cout << "face_norm = " << it->size() << endl;
        for (std::vector<float>::iterator iit = it->begin(); iit != it->end(); ++iit) {
            cout << *iit << " ";
        }
        cout << endl;
#endif
        std::vector<float> output_norm(256);
        calc_norm(mx_output_data, output_norm);
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

static int find_nearest_face_substract_norm(void)
{
    int id = 0;
    double dis_min = 0.0f;
    int best_id = -1;

    std::vector<float> diff(256);
    // dist = np.linalg.norm(rep - clf.means_[maxI])
    for (std::vector< std::vector<float> >::iterator it = rep_db_norm.begin(); it != rep_db_norm.end(); ++it) {
        //diff = mx_output_data - *it;
        std::set_difference(
            mx_output_data.begin(), mx_output_data.end(),
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

#ifdef USE_CV_DISPLAY
void draw_polyline(cv::Mat &img, const dlib::full_object_detection& d, const int start, const int end, bool isClosed = false)
{
    std::vector <cv::Point> points;
    for (int i = start; i <= end; ++i)
    {
        points.push_back(cv::Point(d.part(i).x(), d.part(i).y()));
    }
    cv::polylines(img, points, isClosed, cv::Scalar(0,0,255), 1, 16);
}

void render_face (cv::Mat &img, const dlib::full_object_detection& d)
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
    try
    {
        cv::VideoCapture cap(0);
        if (!cap.isOpened())
        {
            cerr << "Unable to connect to camera" << endl;
            return 1;
        }

#ifdef USE_DLIB_DISPLAY
        image_window win, win_faces;
#endif
#ifdef USE_CV_DISPLAY
        cv::namedWindow( "Cam");
        cv::namedWindow( "Face");
#endif
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
        create_face_db();

        // Grab and process frames until the main window is closed by the user.
        while(1)
        {
#ifdef USE_DLIB_DISPLAY
            if (win.is_closed()) {
                break;
            }
#endif

            // Grab a frame
            cap >> frame;
            if (frame.empty()) {
                cerr << "No captured frame" << endl;
                break;
            }

            std::vector<cv::Rect> cv_faces;
#ifdef USE_CV_FACE_DETECTION
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
                cv::Rect cv_face(
                    faces[i].left(), faces[i].top(),
                    faces[i].right() - faces[i].left(),
                    faces[i].bottom() - faces[i].top()
                );
                cv_faces.push_back(cv_face);
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

            if (shapes.size() == 0) {
                continue;
            }

            start = clock();
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
#endif
            cout << "Align took "
               << double(clock() - start) / CLOCKS_PER_SEC << " sec." << endl;

            GetImage(face_warp, mx_input_data.data());
#if 0
            //-- Read Image Data from file
            std::string test_file = std::string("../data/my-align/larry/image-10.png");
            GetImageFile(test_file, mx_input_data.data());
#endif
            start = clock();
            mx_forward();
            cout << "Forward took "
               << double(clock() - start) / CLOCKS_PER_SEC << " sec." << endl;
#if 0
            cout << "output size = " << mx_output_data.size() << endl;
            for (int i = 0; i < mx_output_data.size(); i++) {
                cout << mx_output_data[i] << " ";
            }
            cout << endl;
#endif

            int id0 = find_nearest_face();
            int id1 = find_nearest_face_norm();
            int id2 = find_nearest_face_substract_norm();
            int id = id1;
            cout << "Best Match is " << id << " name " << name[id] << endl;

            start = clock();
#ifdef USE_DLIB_DISPLAY
            // Display it all on the screen
            win.clear_overlay();
            win.set_image(cimg);
            win.add_overlay(faces);
            win.add_overlay(render_face_detections(shapes));

            cv_image<bgr_pixel> cimg_face_align(face_warp);
            win_faces.set_image(cimg_face_align);
            const rectangle r;
            win_faces.add_overlay(r, rgb_pixel(255,0,0), name[id]);
#endif
#ifdef USE_CV_DISPLAY
            render_face( frame, shapes[0] );
            cv::rectangle( frame, cv_faces[0], cv::Scalar(0,255,0), 1, 16 );
            cv::imshow( "Cam", frame );
            cv::putText( face_warp, name[id], cv::Point(10, 80),
                cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255,255,0), 2 );
            cv::imshow( "Face", face_warp );
            if((cv::waitKey(1) & 0xFF) == 'q')
                break;
#endif
            cout << "Display took "
                << double(clock() - start) / CLOCKS_PER_SEC << " sec." << endl;
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
#ifdef USE_CV_DISPLAY
    cv::destroyAllWindows();
#endif
}

