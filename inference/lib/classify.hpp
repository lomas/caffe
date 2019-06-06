#pragma once
#include "opencv2/opencv.hpp"
#include <string>
#include <vector>

namespace inference
{
    namespace classify
    {
        class PARAM
        {
            public:
            int class_num;
            cv::Scalar mean;
            cv::Size input_size;
            PARAM(){
                mean = cv::Scalar(0,0,0);
                class_num = 0;
            }
            ~PARAM(){

            }
        };

        class CLASSIFY
        {
            public:
            CLASSIFY( std::string model_file,  std::string weights_file, PARAM& param);
            ~CLASSIFY();
            public:
            int forward(std::vector<cv::Mat>& imgs,  cv::Mat& mat_score, int batch_size_max);
            private:
            void* _kernel;
        };
    }
}