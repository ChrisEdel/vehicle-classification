#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/io/ply_io.h>
#include <pcl/common/centroid.h>
#include <pcl/common/distances.h>
#include <pcl/common/transforms.h>

// pass-through-filter:
#include <pcl/filters/passthrough.h>

// voxel-grid-filter:
#include <pcl/filters/voxel_grid.h>

// statistical-outlier-removal-filter:
#include <pcl/filters/statistical_outlier_removal.h>

// radius-outlier-removal-filter:
#include <pcl/filters/radius_outlier_removal.h>

// smoothing:
#include <pcl/surface/mls.h>
#include <pcl/search/kdtree.h>

#include <boost/program_options.hpp>
#include <fmt/core.h>
#include <limits>

#include <pcl/filters/random_sample.h>

namespace po = boost::program_options;


void print_usage(const std::string& prog_name, const po::options_description& opt) {
    std::cout << "\nUsage: " << prog_name << " [options] -i FILENAME\n\n" << opt << std::endl;
}


// Inspiration: https://stackoverflow.com/questions/3018313/algorithm-to-convert-rgb-to-hsv-and-hsv-to-rgb-in-range-0-255-for-both
struct rgb {
    uint8_t r;       // a fraction between 0 and 1
    uint8_t g;       // a fraction between 0 and 1
    uint8_t b;       // a fraction between 0 and 1
};

struct hsv {
    float h;       // angle in degrees
    float s;       // a fraction between 0 and 1
    float v;       // a fraction between 0 and 1
};


// https://www.rapidtables.com/convert/color/rgb-to-hsv.html
hsv rgb2hsv(rgb in) {
    float r_, g_, b_, cmax, cmin, delta;
    hsv out;

    if (in.r < 0 or in.r > 255 or
        in.g < 0 or in.g > 255 or
        in.b < 0 or in.b > 255)
        throw std::invalid_argument("Invalid argument for rgb2hsv");

    r_ = in.r / 255.0f;
    g_ = in.g / 255.0f;
    b_ = in.b / 255.0f;

    cmax = std::max({r_, g_, b_});
    cmin = std::min({r_, g_, b_});

    delta = cmax - cmin;

    if (delta == 0)
        out.h = 0;
    else if (cmax == r_)
        out.h = 60.0f * std::fmod((g_ - b_) / delta, 6);
    else if (cmax == g_)
        out.h = 60.0f * (((b_ - r_) / delta) + 2);
    else if (cmax == b_)
        out.h = 60.0f * (((r_ - g_) / delta) + 4);

    if (cmax == 0)
        out.s = 0;
    else
        out.s = delta / cmax;

    out.v = cmax;

    return out;
}


// https://www.rapidtables.com/convert/color/hsv-to-rgb.html
rgb hsv2rgb(hsv in) {
    float c, x, m, r_, g_, b_;
    rgb out;

    if (in.h < 0 or in.h > 359 or
        in.s < 0 or in.s > 1 or
        in.v < 0 or in.v > 1)
        throw std::invalid_argument("Invalid argument in hsv2rgb");

    c = in.v * in.s;
    x = c * (1 - std::abs(std::fmod(in.h / 60.0f, 2) - 1));
    m = in.v - c;

    if (in.h < 60.0f) {
        r_ = c;
        g_ = x;
        b_ = 0;
    } else if (in.h < 120.0f) {
        r_ = x;
        g_ = c;
        b_ = 0;
    } else if (in.h < 180.0f) {
        r_ = 0;
        g_ = c;
        b_ = x;
    } else if (in.h < 240.0f) {
        r_ = 0;
        g_ = x;
        b_ = c;
    } else if (in.h < 300.0f) {
        r_ = x;
        g_ = 0;
        b_ = c;
    } else if (in.h < 360.0f) {
        r_ = c;
        g_ = 0;
        b_ = x;
    }

    out.r = (r_ + m) * 255.0f;
    out.g = (g_ + m) * 255.0f;
    out.b = (b_ + m) * 255.0f;

    return out;
}


rgb rgb2greyscale_rgb(rgb in) {
    rgb out;

    int greyscale_value = (in.r + in.g + in.b) / 3;

    out.r = greyscale_value;
    out.g = greyscale_value;
    out.b = greyscale_value;

    return out;
}


rgb rgb2black_white_rgb(rgb in) {
    rgb out;

    rgb greyscale = rgb2greyscale_rgb(in);

    if (greyscale.r < 128)
        out.r = 0;
    else
        out.r = 255;

    if (greyscale.g < 128)
        out.g = 0;
    else
        out.g = 255;

    if (greyscale.b < 128)
        out.b = 0;
    else
        out.b = 255;

    return out;
}


pcl::PointXYZ* get_centroid(const pcl::PointCloud<pcl::PointXYZRGBL>::Ptr& pointcloud) {
    auto centroid_calc = pcl::CentroidPoint<pcl::PointXYZRGBL>();
    for (auto const& point: *pointcloud)
        centroid_calc.add(point);
    auto centroid = new pcl::PointXYZ();
    centroid_calc.get(*centroid);
    return centroid;
}


void visualise(const pcl::PointCloud<pcl::PointXYZRGBL>::Ptr& pointcloud, const std::string& name, const bool coordinateSystem) {
    // transform pointcloud: the centroid has to be at the origin
    auto centroid = get_centroid(pointcloud);
    Eigen::Affine3f transform = Eigen::Affine3f::Identity();
    transform.translation() << -centroid->x, -centroid->y, -centroid->z;

    // apply transformation
    pcl::transformPointCloud(*pointcloud, *pointcloud, transform);

    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer (name));
    viewer->setBackgroundColor(0.2, 0.2, 0.2);
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBL> rgb(pointcloud);
    viewer->addPointCloud<pcl::PointXYZRGBL> (pointcloud, rgb, name);
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, name);
    if (coordinateSystem)
        viewer->addCoordinateSystem(1.0); // adds reference axis
    viewer->initCameraParameters();
    viewer->setCameraPosition(0, 0, 4,    0, 0, 0,   0, 1, 0);

    viewer->spin();
}


void floor_filter(const pcl::PointCloud<pcl::PointXYZRGBL>::Ptr& pointcloud, const float ff_threshold) {
    // find minimum height value (y value)
    // comp function should return true, iff the fist argument is less than the second argument
    auto comp = [] (const auto &p1, const auto &p2) {
        return p1.y < p2.y;
    };
    float removal_threshold = std::min_element(pointcloud->begin(), pointcloud->end(), comp)->y + ff_threshold;

    // remove all points that are too low
    auto removal_predicate = [removal_threshold] (const auto &point) {
        return point.y < removal_threshold;
    };
    pointcloud->erase(std::remove_if(pointcloud->begin(), pointcloud->end(), removal_predicate), pointcloud->end());
}


void distance_filter(pcl::PointCloud<pcl::PointXYZRGBL>::Ptr& pointcloud, const float df_threshold) {
    // find centroid
    auto centroid = get_centroid(pointcloud);

    // calculate distance vector
    auto distance_vector = std::vector<std::pair<float, pcl::PointXYZRGBL>>();
    for (auto const& point: *pointcloud)
        distance_vector.emplace_back(pcl::euclideanDistance(*centroid, point), point);

    // comp function should return true, iff the fist argument is less than the second argument
    auto comp = [] (auto const& pair1, auto const& pair2) {
        return pair1.first < pair2.first;
    };
    std::sort(distance_vector.begin(), distance_vector.end(), comp);

    // create new pointcloud with only the desired values
    int index = 0;
    int max_index = pointcloud->size() * (1 - df_threshold);
    pcl::PointCloud<pcl::PointXYZRGBL>::Ptr res_pointcloud (new pcl::PointCloud<pcl::PointXYZRGBL>);
    for (auto const& pair: distance_vector) {
        if (index >= max_index) {
            break;
        }
        res_pointcloud->push_back(pair.second);
        index++;
    }

    pointcloud = res_pointcloud;
}


void subsample(pcl::PointCloud<pcl::PointXYZRGBL>::Ptr& pointcloud, const int subample_val) {

    pcl::PointCloud<pcl::PointXYZRGBL>::Ptr res_pointcloud (new pcl::PointCloud<pcl::PointXYZRGBL>);
    std::sample(pointcloud->begin(), pointcloud->end(), std::back_inserter(*res_pointcloud), subample_val,
                std::mt19937{std::random_device{}()});

    pointcloud = res_pointcloud;
}


uint8_t check_rgb_value(float val) {
    auto val_temp = (int) (val * 255);

    if (val_temp < 0)
        return 0;
    if (val_temp > 255)
        return 255;

    return val_temp;
}

std::vector<std::string> header;
std::string const END_HEADER = "end_header";

pcl::PointCloud<pcl::PointXYZRGBL>::Ptr read_point_cloud(const std::string& filepath, int confidence) {
    float x, y, z, r, g, b, c;
    std::string line;
    
    std::ifstream infile(filepath);

    // parse header
    while(std::getline(infile, line)) {
        if (line == END_HEADER)
            break;

        header.push_back(line + "\n");
    }

    // create empty pointcloud:
    pcl::PointCloud<pcl::PointXYZRGBL>::Ptr pointcloud (new pcl::PointCloud<pcl::PointXYZRGBL>);

    while(infile >> x >> y >> z >> r >> g >> b >> c)
        if (c >= confidence) {
            if (!(x == 0 && y == 0 && z == 0 && r == 0 && g == 0 && b == 0 && c == 0)) {
                auto p = *(new pcl::PointXYZRGBL(x, y, z,
                                                 check_rgb_value(r), check_rgb_value(g), check_rgb_value(b),
                                                 c));
                pointcloud->push_back(p);
            }
        }

    infile.close();

    return pointcloud;
}

float rgb_uint8_to_float(uint8_t val) {
    if (0 <= val && val <= 255)
        return val / 255.0f;
    else
        throw std::invalid_argument("RGB value out of range");
}


template<typename T>
std::string vector_to_string(const std::vector<T>& vec) {

    std::ostringstream result_stream;
    for (const auto element: vec)
        result_stream << fmt::format("{} ", element);

    std::string result_string = result_stream.str();

    return result_string.substr(0, result_string.size() - 1);
}


void write_point_cloud(const pcl::PointCloud<pcl::PointXYZRGBL>::Ptr& pointcloud, const std::string& filepath) {
    std::ofstream outfile(filepath);

    // write header
    for (const auto line: header) {
        std::istringstream iss(line);
        std::vector<std::string> lineSeparatedByWhitespace(std::istream_iterator<std::string>{iss},
                                         std::istream_iterator<std::string>());

        // update size of point cloud in the output file
        if (lineSeparatedByWhitespace.at(0) == "element" && lineSeparatedByWhitespace.at(1) == "vertex") {
            outfile << "element vertex " << pointcloud->size() << "\n";
            continue;
        }
        
        outfile << line;
    }

    outfile << END_HEADER << "\n";

    // write points
    std::string sep = " ";
    for (auto point: *pointcloud) {
        outfile << point.x << sep << point.y << sep << point.z << sep;
        outfile << rgb_uint8_to_float(point.r) << sep << rgb_uint8_to_float(point.g) << sep << rgb_uint8_to_float(point.b) << sep;
        outfile << (int) point.label << "\n";
    }

    outfile.close();
}


int main(int argc, char** argv) {

    // parse options

    // required arguments
    std::string infile;
    po::options_description req("Required options");
    req.add_options()
            ("infile,i", po::value<std::string>(&infile)->required(), "Input file");

    // optional arguments
    float df_threshold, ff_threshold, search_radius;
    std::vector<std::string> ptf, sorf, rorf;
    std::vector<float> vgf, torf, rotorf, trorf;
    int confidence, subsample_val, hue_val;
    std::string outfile;
    po::options_description opt("Options");
    opt.add_options()
            ("help,h", "produce help message")
            ("distance-filter,d", po::value<float>(&df_threshold)->default_value(0.0f),
             "Threshold for distance filter (disabled by default)")
            ("floor-filter,f", po::value<float>(&ff_threshold)->default_value(0.0f),
             "Threshold for floor filter (disabled by default)")
            ("pass-through-filter,p", po::value<std::vector<std::string>>(&ptf)->multitoken(),
             "Takes three arguments: coordinate to filter by (\"x\", \"y\" or \"z\"), lower limit (non-negative float) and upper limit (non-negative float) (disabled by default)")
            ("voxel-grid-filter,v", po::value<std::vector<float>>(&vgf)->multitoken(),
             "Takes three non-negative float values (leaf sizes) as arguments (disabled by default)")
            ("statistical-outlier-removal-filter, O", po::value<std::vector<std::string>>(&sorf)->multitoken(),
             "Takes a non-negative int value (number of neighbors to analyze for each point) and a non-negative float value (standard deviation multiplier) as arguments (disabled by default)")
            ("radius-outlier-removal-filter", po::value<std::vector<std::string>>(&rorf)->multitoken(),
             "Takes a non-negative double value (radius) and a non-negative int value (minimal number of neighbors) as arguments (disabled by default)")
            ("search-radius,r", po::value<float>(&search_radius)->default_value(0.0f),
             "Search radius for the smoothing operation (disabled by default)")
            ("condfidence,c", po::value<int>(&confidence)->default_value(2),
             "Lower limit for confidence (2 by default)")
            ("subsample,s", po::value<int>(&subsample_val)->default_value(0),
             "The number of points to subsample the point cloud to; if the given point cloud is smaller or equal to the given value no"
             "subsampling is performed (disabled by default)")
            ("outfile-named,n", po::value<std::string>(&outfile)->default_value(""),
             "Ply file to write the transformed output to")
            ("outfile,o", "The file to output to will be inferred from the input filename and other options, if no "
                          "filename is specified")
            ("translate", po::value<std::vector<float>>(&torf)->multitoken(), "Takes three float values (unit: meters) as arguments. The first one defines the translation on the x axis, " 
            "the second one on the y axis and the third one on the z axis.")
            ("rotate", po::value<std::vector<float>>(&rotorf)->multitoken(), "Takes three float values (unit: degrees) as arguments. The first one defines the rotation on the x axis, " 
            "the second one on the y axis and the third one on the z axis.")
            ("transform-random", po::value<std::vector<float>>(&trorf)->multitoken(), "Takes two float values (unit: meters) as arguments. The first one defines the lower bound of the range, "
            "the second one the upper bound.")
            ("random-sample", "Applies random sampling.")
            ("hue", po::value<int>(&hue_val), "Change hue randomly; if no value is provided, -1 will generate a random value; the valid range is [0, 359]")
            ("greyscale", "Transform colors to greyscale.")
            ("black-white", "Transform colors to black-white.")
            ("headless", "Executes the program without visualisation");

    po::options_description all("");
    all.add(req).add(opt);

    po::variables_map vm;

    try {
        po::store(po::command_line_parser(argc, argv).options(all).run(), vm);

        if (vm.count("help")) {
            print_usage(argv[0], all);
            return 0;
        }

        po::notify(vm);

    } catch (const boost::wrapexcept<boost::program_options::required_option>& e) {
        std::cerr << "Error: No input file was provided!" << std::endl;
        print_usage(argv[0], all);
        return 1;
    }

    std::cout << "Reading pointcloud: " << infile << std::endl;
    pcl::PointCloud<pcl::PointXYZRGBL>::Ptr pointcloud = read_point_cloud(infile, confidence);
    std::cout << "Pointcloud imported: " << pointcloud->size() << " points" << std::endl;

    // distance filter
    if (df_threshold != 0) {
        if (df_threshold < 0 or df_threshold > 1)
            throw std::invalid_argument("Distance filter threshold must be a value from (0, 1]");

        std::cout << "Applying distance filter with threshold " << df_threshold << std::endl;
        distance_filter(pointcloud, df_threshold);

        header.push_back(fmt::format("distance_filter {}\n", df_threshold));
    }

    // floor filter
    if (ff_threshold != 0) {
        if (ff_threshold < 0)
            throw std::invalid_argument("Floor filter threshold must be a positive number");

        std::cout << "Applying floor filter with threshold " << ff_threshold << std::endl;
        floor_filter(pointcloud, ff_threshold);

        header.push_back(fmt::format("floor_filter {}\n", ff_threshold));
    }

    // pass-through-filter
    if (!ptf.empty()) {
        if (ptf.size() != 3)
            throw std::invalid_argument("Invalid number of arguments for the pass-through-filter. Please refer to the usage.");

        std::string coordinate;
        float lower_limit, upper_limit;

        try {
            coordinate = ptf.at(0);
            lower_limit = std::stof(ptf.at(1));
            upper_limit = std::stof(ptf.at(2));
        } catch (const std::exception&) {
            throw std::invalid_argument("Invalid arguments for the pass-through-filter. Please refer to the usage.");
        }

        std::cout << "Applying pass-through-filter on coordinate " << coordinate << " with lower limit " << lower_limit << " and upper limit " << upper_limit << std::endl;

        pcl::PointCloud<pcl::PointXYZRGBL>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZRGBL>);

        // Create the filtering object
        pcl::PassThrough<pcl::PointXYZRGBL> pass;
        pass.setInputCloud (pointcloud);
        pass.setFilterFieldName (coordinate);
        pass.setFilterLimits (lower_limit, upper_limit);
        //pass.setFilterLimitsNegative (true);
        pass.filter (*cloud_filtered);

        pointcloud = cloud_filtered;

        header.push_back(fmt::format("pass-through-filter {}\n", vector_to_string(ptf)));
    }

    // voxel-grid-filter
    if (!vgf.empty()) {
        if (vgf.size() != 3) {
            throw std::invalid_argument("Invalid number of arguments for the voxel-grid-filter. Please refer to the usage.");
        }

        std::cout << "Applying voxel-grid-filter with leaf size (" << vgf.at(0) << ", " << vgf.at(1) << ", " << vgf.at(2) << ")" << std::endl;

        pcl::PointCloud<pcl::PointXYZRGBL>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZRGBL>);

        // Create the filtering object
        pcl::VoxelGrid<pcl::PointXYZRGBL> sor;
        sor.setInputCloud (pointcloud);
        sor.setLeafSize (vgf.at(0), vgf.at(1), vgf.at(2));
        sor.filter (*cloud_filtered);

        pointcloud = cloud_filtered;

        header.push_back(fmt::format("voxel-grid-filter {}\n", vector_to_string(vgf)));
    }

    // statistical-outlier-removal-filter
    if (!sorf.empty()) {
        if (sorf.size() != 2) {
            throw std::invalid_argument("Invalid number of arguments for the statistical-outlier-removal-filter. Please refer to the usage.");
        }

        int meanK;
        float stddevMulThresh;

        try {
            meanK = std::stoi(sorf.at(0));
            stddevMulThresh = std::stof(sorf.at(1));
        } catch (const std::exception&) {
            throw std::invalid_argument("Invalid arguments for the statistical-outlier-removal-filter. Please refer to the usage.");
        }

        std::cout << "Applying statistical-outlier-removal-filter with meanK " << meanK << " and stddevMulThresh " << stddevMulThresh << std::endl;

        pcl::PointCloud<pcl::PointXYZRGBL>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZRGBL>);

        // Create the filtering object
        pcl::StatisticalOutlierRemoval<pcl::PointXYZRGBL> sor;
        sor.setInputCloud (pointcloud);
        sor.setMeanK (meanK);
        sor.setStddevMulThresh (stddevMulThresh);
        sor.filter (*cloud_filtered);

        pointcloud = cloud_filtered;

        header.push_back(fmt::format("statistical-outlier-removal-filter {}\n", vector_to_string(sorf)));
    }

    // radius-outlier-removal-filter
    if (!rorf.empty()) {
        if (rorf.size() != 2) {
            throw std::invalid_argument("Invalid number of arguments for the radius-outlier-removal-filter. Please refer to the usage.");
        }

        double radius;
        float minNeighborsInRadius;

        try {
            radius = std::stod(rorf.at(0));
            minNeighborsInRadius = std::stof(rorf.at(1));
        } catch (const std::exception&) {
            throw std::invalid_argument("Invalid arguments for the radius-outlier-removal-filter. Please refer to the usage.");
        }

        std::cout << "Applying radius-outlier-removal-filter with radius " << radius << " and minNeighborsInRadius " << minNeighborsInRadius << std::endl;

        pcl::PointCloud<pcl::PointXYZRGBL>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZRGBL>);

        // Create the filtering object
        pcl::RadiusOutlierRemoval<pcl::PointXYZRGBL> sor;
        sor.setInputCloud (pointcloud);
        sor.setRadiusSearch(radius);
        sor.setMinNeighborsInRadius (minNeighborsInRadius);
        sor.setKeepOrganized(true);
        sor.filter (*cloud_filtered);

        pointcloud = cloud_filtered;

        header.push_back(fmt::format("radius-outlier-removal-filter {}\n", vector_to_string(rorf)));
    }

    // smoothing
    if (search_radius != 0) {
        std::cout << "Smoothing pointcloud with search radius " << search_radius << std::endl;

        // Create a KD-Tree
        pcl::search::KdTree<pcl::PointXYZRGBL>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGBL>);

        // Output has the PointXYZRGBL type in order to store the normals calculated by MLS
        auto mls_points = std::shared_ptr<pcl::PointCloud<pcl::PointXYZRGBL>>(new pcl::PointCloud<pcl::PointXYZRGBL>());

        // Init object (second point type is for the normals, even if unused)
        pcl::MovingLeastSquares<pcl::PointXYZRGBL, pcl::PointXYZRGBL> mls;

        mls.setComputeNormals(true);

        // Set parameters
        mls.setInputCloud(pointcloud);
        mls.setPolynomialOrder(2);
        mls.setSearchMethod(tree);
        mls.setSearchRadius(search_radius);

        // Reconstruct
        mls.process(*mls_points);
        pointcloud = mls_points;

        header.push_back(fmt::format("smoothing {}\n", search_radius));
    }

    // subsampling
    if (subsample_val != 0) {
        if (subsample_val < 0 or std::floor(subsample_val) != subsample_val)
            throw std::invalid_argument("The subsampling size must be a positive integer");
        if (pointcloud->size() <= subsample_val)
            std::cout << "WARNING: subsampling not applied, because pointcloud size (" << pointcloud->size() << ") is "
                      << "smaller or equal to the subsampling size provided (" << subsample_val << ")" << std::endl;
        else
            std::cout << "Applying subsampling with subsampling size " << subsample_val << std::endl;
            subsample(pointcloud, subsample_val);

        header.push_back(fmt::format("subsample {}\n", subsample_val));
    }

    // translation
    if (!torf.empty()) {
        if (torf.size() != 3) {
            throw std::invalid_argument("Invalid number of arguments for translation. Please refer to the usage.");
        }

        Eigen::Affine3f transform_translation = Eigen::Affine3f::Identity();

        transform_translation.translation() << torf.at(0), torf.at(1), torf.at(2);

        // Print the transformation
        printf ("\n Transforming the point cloud with the following matrix:\n");
        std::cout << transform_translation.matrix() << std::endl;

        pcl::PointCloud<pcl::PointXYZRGBL>::Ptr tmp_pc (new pcl::PointCloud<pcl::PointXYZRGBL>);

        pcl::transformPointCloud (*pointcloud, *tmp_pc, transform_translation);

        pointcloud = tmp_pc;

        header.push_back(fmt::format("translation {}, {}, {}\n", torf.at(0), torf.at(1), torf.at(2)));
    }

    // rotation
    if (!rotorf.empty()) {
        if (rotorf.size() != 3) {
            throw std::invalid_argument("Invalid number of arguments for rotation. Please refer to the usage.");
        }

        Eigen::Affine3f transform_trotation = Eigen::Affine3f::Identity();

        
        transform_trotation.rotate (Eigen::AngleAxisf (rotorf.at(0) * (M_PI / 180), Eigen::Vector3f::UnitX()));
        transform_trotation.rotate (Eigen::AngleAxisf (rotorf.at(1) * (M_PI / 180), Eigen::Vector3f::UnitY()));
        transform_trotation.rotate (Eigen::AngleAxisf (rotorf.at(2) * (M_PI / 180), Eigen::Vector3f::UnitZ()));

        // Print the transformation
        printf ("\n Transforming the point cloud with the following matrix:\n");
        std::cout << transform_trotation.matrix() << std::endl;

        pcl::PointCloud<pcl::PointXYZRGBL>::Ptr tmp_pc (new pcl::PointCloud<pcl::PointXYZRGBL>);

        pcl::transformPointCloud (*pointcloud, *tmp_pc, transform_trotation);

        pointcloud = tmp_pc;

        header.push_back(fmt::format("rotation {}, {}, {}\n", rotorf.at(0), rotorf.at(1), rotorf.at(2)));
    }

    // random transformation
    if (!trorf.empty()) {
        if (trorf.size() != 2) {
            throw std::invalid_argument("Invalid number of arguments for random transformation. Please refer to the usage.");
        }

        std::cout << "Applying random transformation with range (" << trorf.at(0) << ", " << trorf.at(1) << ")" << std::endl;

        std::random_device rd; // obtain a random number from hardware
        std::mt19937 gen(rd()); // seed the generator
        std::uniform_real_distribution<> distr(trorf.at(0), trorf.at(1)); // define the range

        pcl::PointCloud<pcl::PointXYZRGBL>::Ptr tmp_pc (new pcl::PointCloud<pcl::PointXYZRGBL>);
            
        for (auto &point: *pointcloud) {
            point.x += distr(gen);
            point.y += distr(gen);
            point.z += distr(gen);

            tmp_pc->push_back(point);
        }

        pointcloud = tmp_pc;

        header.push_back(fmt::format("random transformation ({},{})\n", trorf.at(0), trorf.at(1)));
    }

    // random sampling (shuffle)
    if (vm.count("random-sample")) {
        std::random_device rd; // obtain a random number from hardware
        std::mt19937 gen(rd()); // seed the generator
        std::uniform_real_distribution<> distr(std::numeric_limits<float>::min(), std::numeric_limits<float>::max()); // define the range
        float seed = distr(gen);

        std::cout << "Applying random sampling with seed " << seed << std::endl;

        pcl::RandomSample <pcl::PointXYZRGBL> random;
        pcl::PointCloud<pcl::PointXYZRGBL>::Ptr tmp_pc (new pcl::PointCloud<pcl::PointXYZRGBL>);


        random.setInputCloud(pointcloud);
        random.setSeed(seed);
        random.setSample((unsigned int)(pointcloud->size()));
        random.filter(*tmp_pc);

        pointcloud = tmp_pc;

        header.push_back(fmt::format("random sample (shuffle) with seed {}\n", seed));
    }

    if (vm.count("hue")) {
        std::cout << "Applying hue shift" << std::endl;

        if (hue_val == -1) {
            std::random_device rd; // obtain a random number from hardware
            std::mt19937 gen(rd()); // seed the generator // time(nullptr)
            std::uniform_int_distribution<> distr(0, 359); // define the range
            hue_val = distr(gen);
        }

        if (hue_val < 0 or hue_val > 359)
            throw std::invalid_argument("Invalid hue value, it mus be from range [0, 359]");

        std::cout << "New hue: " << hue_val << std::endl;

        pcl::PointCloud<pcl::PointXYZRGBL>::Ptr tmp_pc (new pcl::PointCloud<pcl::PointXYZRGBL>);

        struct rgb rgb_values;
        struct hsv hsv_values;

        for (auto &point: *pointcloud) {
            rgb_values.r = point.r;
            rgb_values.g = point.g;
            rgb_values.b = point.b;

            hsv_values = rgb2hsv(rgb_values);
            hsv_values.h = hue_val;
            rgb_values = hsv2rgb(hsv_values);

            point.r = rgb_values.r;
            point.g = rgb_values.g;
            point.b = rgb_values.b;

            tmp_pc->push_back(point);
        }

        pointcloud = tmp_pc;

        header.push_back(fmt::format("hue shifted to {}\n", hue_val));
    }

    if (vm.count("greyscale")) {
        std::cout << "Applying greyscale transformation" << std::endl;

        pcl::PointCloud<pcl::PointXYZRGBL>::Ptr tmp_pc (new pcl::PointCloud<pcl::PointXYZRGBL>);

        struct rgb rgb_values;

        for (auto &point: *pointcloud) {
            rgb_values.r = point.r;
            rgb_values.g = point.g;
            rgb_values.b = point.b;

            rgb_values = rgb2greyscale_rgb(rgb_values);

            point.r = rgb_values.r;
            point.g = rgb_values.g;
            point.b = rgb_values.b;

            tmp_pc->push_back(point);
        }

        pointcloud = tmp_pc;

        header.push_back("greyscale\n");
    }

    if (vm.count("black-white")) {
        std::cout << "Applying black-white transformation" << std::endl;

        pcl::PointCloud<pcl::PointXYZRGBL>::Ptr tmp_pc (new pcl::PointCloud<pcl::PointXYZRGBL>);

        struct rgb rgb_values;

        for (auto &point: *pointcloud) {
            rgb_values.r = point.r;
            rgb_values.g = point.g;
            rgb_values.b = point.b;

            rgb_values = rgb2black_white_rgb(rgb_values);

            point.r = rgb_values.r;
            point.g = rgb_values.g;
            point.b = rgb_values.b;

            tmp_pc->push_back(point);
        }

        pointcloud = tmp_pc;

        header.push_back("black-white\n");
    }

    // write to file
    if (vm.count("outfile") || (vm.count("outfile-named") && !outfile.empty())) {
        // Save output
        if (outfile.empty()) { // we name the outfile automatically
            size_t dot_idx = infile.find_last_of('.');
            if (dot_idx == std::string::npos)
                throw std::invalid_argument("Filename does not have a infile extension!");
            outfile = fmt::format("{}_df{:.2f}_ff{:.2f}_sr{:.2f}_c{}_s{}{}", infile.substr(0, dot_idx),
                                  df_threshold, ff_threshold, search_radius, confidence, subsample_val,
                                  infile.substr(dot_idx, infile.length()));
        }
        std::cout << "Writing pointcloud: " << outfile << std::endl;
        write_point_cloud(pointcloud, outfile);
    }

    // visualise
    if (!vm.count("headless")) {
        visualise(pointcloud, infile, false);
    }

    return 0;
}
