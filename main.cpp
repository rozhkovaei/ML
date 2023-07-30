#include <iostream>
#include <fstream>

#include <Eigen/Dense>

#include <mnist/mlp_classifier.h>
#include <helpers.h>

const size_t input_dim = 784;
const size_t hidden_dim = 128;
const size_t output_dim = 10;

using namespace mnist;

int main(int argc, char* argv[])
{
    if(argc < 4)
    {
        std::cout << "not enough arguments.\nUsage: fashion_mnist [file_name]\t[w1 file]\t[w2 file] " << std::endl;
        return 0;
    }

    std::string data_file = argv[1];
    std::string w1_file = argv[2];
    std::string w2_file = argv[3];

    std::cout << "processing file: " << data_file << std::endl;

    std::ifstream test_data{data_file};
    if(!test_data.is_open())
    {
        std::cout << "error opening file " << std::endl;
        return 0;
    }

    auto w1 = read_mat_from_file(input_dim, hidden_dim, w1_file);
    auto w2 = read_mat_from_file(hidden_dim, output_dim, w2_file);

    auto clf = MlpClassifier{w1.transpose(), w2.transpose()};

    auto features = MlpClassifier::features_t{};

    int correct = 0;
    int all = 0;

    for (std::string line; std::getline( test_data, line );)
    {
        std::istringstream linestream{line};

        size_t y_true;
        std::string fake;

        std::getline(linestream, fake, ',');

        std::stringstream sstream(fake);
        sstream >> y_true;

        read_features_csv(linestream, features);

        auto y_pred = clf.predict(features);

        all++;
        if( y_pred == y_true )
            correct++;
    }

    if(!all)
        std::cout << " no data !" << std::endl;

    std::cout << "accuracy: " << (correct * 1.0)/ (all * 1.0 ) << std::endl;

    return 0;
}