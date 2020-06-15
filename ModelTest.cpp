#include <fstream>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <opencv2/ml.hpp>
#include <opencv2/core.hpp>

int main()
{
    const int features{ 4 };

    // Loading datasets
    std::vector<cv::Ptr<cv::ml::TrainData>> data(features);
    // if name of dataset file is "0.csv" , "1.csv" etc
    for (int k = 0; k < features; k++)
    {
        data[k] = cv::ml::TrainData::loadFromCSV(std::to_string(k) + ".csv", 0, 0, 1);
        data[k]->setTrainTestSplitRatio(0.8);
    }
    std::cout << "Data is ready" << std::endl << std::endl;
    std::ofstream os("report.txt");
    bool save_flag = 1;
    for (int k = 0; k < features; k++)
    {
        // Model creation
        cv::Ptr<cv::ml::RTrees> forest = cv::ml::RTrees::create();
        forest->setMaxCategories(15);
        forest->setMaxDepth(25);
        forest->setMinSampleCount(2);
      
        // Model training
        forest->train(data[k]);
        if (save_flag) forest->save("forest" + std::to_string(k) + ".yml");
        std::cout << "Forest #" << k << " is ready" << std::endl;

        // Testing
        cv::Mat labels = data[k]->getTestResponses();
        cv::Mat samples = data[k]->getTestSamples();
        int n_samples = data[k]->getNTestSamples();
        double accuracy = 0;
        std::unordered_map<int, int> true_predict;
        std::unordered_map<int, int> speakers_seg;
        os << "Test #" << k << ":" << std::endl << std::endl;
        for (long int i = 0; i < n_samples; i++)
        {
            int speaker = labels.at<float>(i, 0);
            speakers_seg[speaker]++;
            float predicted_speaker = forest->predict(samples.row(i));
            if (predicted_speaker == speaker) {
                accuracy++;
                true_predict[predicted_speaker]++;
            }
        }
        
        // Results
        int speakers_num = speakers_seg.size();
        for (int i = 0; i < speakers_num; i++)
        {
            os << "Speaker #" << i << " accuracy: "
                << true_predict[i] / (double)speakers_seg[i] * 100 << " %" << std::endl << std::endl;
        }
        os << "Verified segments: " << n_samples << std::endl;
        std::cout << "Test #" << k << " is over" << std::endl;
        os << "Total accuracy: " << accuracy / n_samples * 100 << std::endl << std::endl << std::endl;
    }
}
