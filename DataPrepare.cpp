#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

#include <blitz/array.h>
#include <SFML/Audio.hpp>

#include "BOB/include/bob.ap/Ceps.h"

enum FeaturesType { MFCC, IMFCC, LFCC, RFCC };
void prepare_data(std::vector<std::string>& v, int type);

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cout << "Need the path to the database" << std::endl;
        return -1;
    }

    const int features{ 4 }; // number of tested features

   // Vector of all database files
    std::vector<std::string> files;
    for (auto& path : std::filesystem::recursive_directory_iterator(argv[1],
        std::filesystem::directory_options::skip_permission_denied)) {
        if (path.is_regular_file()) {
            files.push_back(path.path().string());
        }
    }

    // Creating training bases with feature vectors
    for (int i = 0; i < features; i++)
        prepare_data(files, i);

}

void prepare_data(std::vector<std::string>& v, int type)
{
    std::ofstream os(std::to_string(type) + ".csv");
    std::vector<std::string> speakers;
    int new_label = 0;
    for (int f = 0; f < v.size(); f++)
    {
        // Speaker definition by file path
        int label;
        std::vector<std::string> strings;
        std::istringstream filename(v[f]);
        std::string s;
        while (std::getline(filename, s, '\\')) {
            strings.push_back(s);
        }
        std::string sp = strings[strings.size() - 2];
        auto ex = std::find(speakers.begin(), speakers.end(), sp);
        if (ex == speakers.end()) {
            speakers.push_back(sp);
            label = new_label;
            new_label++;
        }
        else {
            label = std::distance(speakers.begin(), ex);
        }

        // Read file
        sf::SoundBuffer file;
        if (!file.loadFromFile(v[f])) {
            std::cout << "File " << v[f] << " error" << std::endl;
            break;
        }
        auto file_buf = file.getSamples();
        int samples_count = file.getSampleCount();
        blitz::Array <double, 1> input(samples_count);
        for (unsigned long i = 0; i < samples_count; i++)
            input(i) = file_buf[i] / 32768.0;

        // File data processing
        switch (type)
        {
        case MFCC: {
            bob::ap::Ceps mfcc(16000, 30.0, 10.0, 26, 13, 0.0, 8000.0, 2, 0.97);
            mfcc.setWithDelta(true);
            mfcc.setWithDeltaDelta(true);

            auto output_size = mfcc.getShape(input);
            blitz::Array <double, 2> output(output_size[0], output_size[1]);
            mfcc(input, output);

            for (int i = 0; i < output_size[0]; i++) {
                os << label << ",";
                for (int k = 0; k < output_size[1]; k++) {
                    if (k != output_size[1] - 1)
                        os << output(i, k) << ",";
                    else os << output(i, k) << std::endl;
                }
            }
            break;
        }

        case IMFCC: {
            bob::ap::Ceps imfcc(16000, 30.0, 10.0, 26, 13, 0.0, 8000.0, 2, 0.97, true, false, false, false, true);
            imfcc.setWithDelta(true);
            imfcc.setWithDeltaDelta(true);

            auto output_size = imfcc.getShape(input);
            blitz::Array <double, 2> output(output_size[0], output_size[1]);
            imfcc(input, output);

            for (int i = 0; i < output_size[0]; i++) {
                os << label << ",";
                for (int k = 0; k < output_size[1]; k++) {
                    if (k != output_size[1] - 1)
                        os << output(i, k) << ",";
                    else os << output(i, k) << "\n";
                }
            }
            break;
        }

        case LFCC: {
            bob::ap::Ceps lfcc(16000, 30.0, 10.0, 26, 13, 0.0, 8000.0, 2, 0.97, false, false, false);
            lfcc.setWithDelta(true);
            lfcc.setWithDeltaDelta(true);

            auto output_size = lfcc.getShape(input);
            blitz::Array <double, 2> output(output_size[0], output_size[1]);
            lfcc(input, output);

            for (int i = 0; i < output_size[0]; i++) {
                os << label << ",";
                for (int k = 0; k < output_size[1]; k++) {
                    if (k != output_size[1] - 1)
                        os << output(i, k) << ",";
                    else os << output(i, k) << "\n";
                }
            }
            break;
        }

        case RFCC: {
            bob::ap::Ceps rfcc(16000, 30.0, 10.0, 26, 13, 0.0, 8000.0, 2, 0.97, false, false, false, true);
            rfcc.setWithDelta(true);
            rfcc.setWithDeltaDelta(true);

            auto output_size = rfcc.getShape(input);
            blitz::Array <double, 2> output(output_size[0], output_size[1]);
            rfcc(input, output);
            for (int i = 0; i < output_size[0]; i++) {
                os << label << ",";
                for (int k = 0; k < output_size[1]; k++) {
                    if (k != output_size[1] - 1)
                        os << output(i, k) << ",";
                    else os << output(i, k) << "\n";
                }
            }
            break;
        }
        }
    }
    os.close();
}

