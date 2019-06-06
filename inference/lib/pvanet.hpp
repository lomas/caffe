#pragma once
#include "opencv2/opencv.hpp"
#include <string>
#include <vector>

namespace inference
{
    namespace pvanet
    {
        class TARGET
        {
            cv::Rect m_location;
            float m_score;
            int m_class_id;
            public:
            TARGET();
            ~TARGET();
            TARGET(const TARGET& target);
            public:
            TARGET& operator=(const TARGET& target);
            cv::Rect& location();
            int& class_id();
            float& score();
        };
        class PARAM
        {
            public:
            int class_num;
            int size_min, size_max;
            cv::Scalar mean;
            int scale_depth;
            float score_nms, score_obj;
            PARAM(){
                scale_depth = 32;
                score_nms = 0.3;
                score_obj = 0.6;
            }
            ~PARAM(){

            }
        };
        class PVANET
        {
        public:
            PVANET( std::string& model_file,  std::string& weights_file, PARAM& param);
            ~PVANET();
        public:
            int forward(cv::Mat image, std::vector< TARGET >& targets);
        private:
            void* _kernel;
        };
        
    }
}

