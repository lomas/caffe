#include <stdio.h>  // for snprintf
#include <string>
#include <vector>
#include <math.h>
#include <fstream>
#include "caffe/caffe.hpp"
#include "classify.hpp"



namespace inference
{
    namespace classify
    {
        struct KERNEL
        {
         
            caffe::shared_ptr< caffe::Net<float> > net;
            PARAM param;

        };


        CLASSIFY::CLASSIFY(std::string model_file, std::string weights_file, PARAM& param)
        {
            KERNEL* kernel = new KERNEL();
            if (kernel == NULL) return;

            kernel->net = caffe::shared_ptr< caffe::Net<float> > (new caffe::Net<float>(model_file, caffe::TEST));
            kernel->net->CopyTrainedLayersFrom(weights_file);

            memcpy(&(kernel->param), &param, sizeof(PARAM));

            _kernel = (void*)kernel;
            return;
        }
        CLASSIFY::~CLASSIFY()
        {
            KERNEL* kernel = (KERNEL*)_kernel;
            if(kernel == NULL) return;
            delete kernel;
            _kernel = NULL;
            return;
        }
        int CLASSIFY::forward(std::vector<cv::Mat>& imgs,  cv::Mat& mat_score, int batch_size_max)
        {
            if (imgs.size() < 1) return -1;
            KERNEL* kernel = (KERNEL*)_kernel;
            if(kernel == NULL) return -1;

            int width = kernel->param.input_size.width;
            int height = kernel->param.input_size.height;

            cv::Mat mat_mean(height,width,CV_32FC3, kernel->param.mean);
            cv::Mat mat_float(height,width,CV_32FC3);

            mat_score = cv::Mat::zeros(imgs.size(),kernel->param.class_num,CV_32FC1);
            

            for(int ind = 0; ind < imgs.size(); ind += batch_size_max)
            {
                int ind_begin = ind, ind_end = ind + batch_size_max;
                if(ind_end > imgs.size()) ind_end = imgs.size();
                int batch_size = ind_end - ind_begin;
                caffe::shared_ptr< caffe::Blob<float> > input_layer = kernel->net->blob_by_name("data");
                input_layer->Reshape(batch_size, 3, height, width);
                float* input_data = input_layer->mutable_cpu_data();
                for(int k = ind_begin; k < ind_end; k++)
                {
                    cv::Mat img;
                    if(img.cols != kernel->param.input_size.width || img.rows != kernel->param.input_size.height)
                    {
                        cv::resize(imgs[k], img, kernel->param.input_size);
                    }
                    else
                    {
                        img = imgs[k].clone();
                    }
                    
                    img.convertTo(mat_float,CV_32F);
                    cv::subtract(mat_float,mat_mean, mat_float);
                    
                    std::vector< cv::Mat > imgs_float;
                    for(int ch = 0; ch < 3; ch++, input_data += width * height)
                    {
                        cv::Mat mat_ch(height,width,CV_32FC1, input_data);
                        imgs_float.push_back(mat_ch);
                    }


                    cv::split(mat_float, imgs_float);

                    
                }
                

                kernel->net->Reshape();
                kernel->net->Forward();
                

                caffe::Blob<float>* output_layer = kernel->net->output_blobs()[0];
                
                const float* output_data = output_layer->cpu_data();

                for(int k = ind_begin; k < ind_end; k++)
                {
                    for(int c = 0; c < kernel->param.class_num; c++)
                    {
                        mat_score.at<float>(k,c) = *(output_data++);
                    }
                }                
            }
            return 0;
        }
    }
}