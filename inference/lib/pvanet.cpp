#include <stdio.h>  // for snprintf
#include <string>
#include <vector>
#include <math.h>
#include <fstream>
#include "caffe/caffe.hpp"
#include "caffe/util/nms.hpp"
#include "pvanet.hpp"

using namespace caffe;
using namespace std;

//#define max(a, b) (((a)>(b)) ? (a) :(b))
//#define min(a, b) (((a)<(b)) ? (a) :(b))

namespace inference
{
    namespace pvanet
    
    {

        /*
        * ===  FUNCTION  ======================================================================
        *         Name:  bbox_transform_inv
        *  Description:  Compute bounding box regression value
        * =====================================================================================
        */
        void bbox_transform_inv(int class_num,int num, const float* box_deltas, const float* pred_cls, float* boxes, float* pred, int img_height, int img_width)
        {
            float width, height, ctr_x, ctr_y, dx, dy, dw, dh, pred_ctr_x, pred_ctr_y, pred_w, pred_h;
            for(int i=0; i< num; i++)
            {
                width = boxes[i*4+2] - boxes[i*4+0] + 1.0;
                height = boxes[i*4+3] - boxes[i*4+1] + 1.0;
                ctr_x = boxes[i*4+0] + 0.5 * width;		// (ctr_x, ctr_y) is center coordinater of bounding box 
                ctr_y = boxes[i*4+1] + 0.5 * height;
                for (int j=0; j< class_num; j++)
                {

                    dx = box_deltas[(i*class_num+j)*4+0];
                    dy = box_deltas[(i*class_num+j)*4+1];
                    dw = box_deltas[(i*class_num+j)*4+2];
                    dh = box_deltas[(i*class_num+j)*4+3];
                    pred_ctr_x = ctr_x + width*dx;
                    pred_ctr_y = ctr_y + height*dy;
                    pred_w = width * exp(dw);
                    pred_h = height * exp(dh);
                    pred[(j*num+i)*5+0] = std::max<float>(std::min<float>(pred_ctr_x - 0.5* pred_w, img_width -1), 0);	// avoid over boundary
                    pred[(j*num+i)*5+1] = std::max<float>(std::min<float>(pred_ctr_y - 0.5* pred_h, img_height -1), 0);
                    pred[(j*num+i)*5+2] = std::max<float>(std::min<float>(pred_ctr_x + 0.5* pred_w, img_width -1), 0);
                    pred[(j*num+i)*5+3] = std::max<float>(std::min<float>(pred_ctr_y + 0.5* pred_h, img_height -1), 0);
                    pred[(j*num+i)*5+4] = pred_cls[i*class_num+j];
                }
            }

        }
        struct Info
        {
            float score;
            const float* head;
        };
        bool compare(const Info& Info1, const Info& Info2)
        {
            return Info1.score > Info2.score;
        }

        void boxes_sort(const int num, const float* pred, float* sorted_pred)
        {
            vector<Info> my;
            Info tmp;
            for (int i = 0; i< num; i++)
            {
                tmp.score = pred[i*5 + 4];
                tmp.head = pred + i*5;
                my.push_back(tmp);
            }
            std::sort(my.begin(), my.end(), compare);
            for (int i=0; i<num; i++)
            {
                for (int j=0; j<5; j++)
                    sorted_pred[i*5+j] = my[i].head[j];	// sequence data
            }
        }        

        struct KERNEL
        {
         
            shared_ptr< Net<float> > net;

            PARAM param;
        };

        TARGET::TARGET()
        {
            m_location = cv::Rect(0,0,0,0);
            m_score = 0;
            m_class_id = -1;
        }

        TARGET::TARGET(const TARGET& target)
        {
            *this = target;
            return;
        }
        
        TARGET::~TARGET()
        {
            m_location = cv::Rect(0,0,0,0);
            m_score = 0;
            m_class_id = -1;
            return;
        }

        TARGET& TARGET::operator= (const TARGET& target)
        {
            this->location() = target.m_location;
            this->score() = target.m_score;
            this->class_id() = target.m_class_id;
            return *this;
        }

        cv::Rect& TARGET::location() {
            return m_location;
        }

        int& TARGET::class_id() {
            return m_class_id;
        }

        float& TARGET::score() {
            return m_score;
        }

        PVANET::PVANET(string& model_file,string& weights_file, PARAM& param)
        {
            KERNEL* kernel = new KERNEL;
            if(kernel == NULL) return;


            memcpy(&(kernel->param), &param, sizeof(PARAM));
            
            kernel->net = shared_ptr<Net<float> >(new Net<float>(model_file, caffe::TEST));

            kernel->net->CopyTrainedLayersFrom(weights_file);

            _kernel = kernel;

        }

        PVANET::~PVANET()
        {
            KERNEL* kernel = (KERNEL*)_kernel;
            if(kernel == NULL) return;
            delete kernel;
            _kernel = NULL;
            return;
        }




        int PVANET::forward(cv::Mat image, std::vector< TARGET >& targets)
        {    
            
            targets.clear();
            KERNEL* kernel = (KERNEL*)_kernel;
            if(kernel == NULL) return -1;

            //float CONF_THRESH = 0.6; //0.8;	// confidence thresh
            //float NMS_THRESH = 0.4;

            
            float *boxes = NULL;
            float *pred = NULL;
            float *pred_per_class = NULL;
            float *sorted_pred_cls = NULL;
            int *keep = NULL;


            
        
            cv::Mat bgr_img;

            if (image.channels() == 1) 
            {
                cv::cvtColor(image, bgr_img, cv::COLOR_GRAY2BGR);
            }
            else
            {
                bgr_img = image;
            }
            
            
            
            int im_size_min = std::min(bgr_img.rows, bgr_img.cols);
            int im_size_max = std::max(bgr_img.rows, bgr_img.cols);
            float im_scale = float(kernel->param.size_min) / im_size_min;// if im_scale have many value, it will support muti-scale detection
            if (round(im_scale * im_size_max) > kernel->param.size_max)
                im_scale = float(kernel->param.size_max) / im_size_max;

            // Make width and height be multiple of a specified number
            float im_scale_x = std::floor(bgr_img.cols * im_scale / kernel->param.scale_depth) * kernel->param.scale_depth / bgr_img.cols;
            float im_scale_y = std::floor(bgr_img.rows * im_scale / kernel->param.scale_depth) * kernel->param.scale_depth / bgr_img.rows;

            // keep image size less than 2000 * 640
            int height = int(bgr_img.rows * im_scale_y);
            int width = int(bgr_img.cols * im_scale_x);
            int num_out;
            cv::Mat cv_resized;

            
            float im_info[6];	// (cv_resized)image's height, width, scale(equal 1 or 1/max_scale) only first two used
            float data_buf[height*width*3];	// each pixel value in cv_resized
    
            const float* bbox_delt;
            const float* rois;
            const float* pred_cls;
            int num;

            cv::resize(bgr_img, bgr_img, cv::Size(width,height));



            cv::Mat mean_img(bgr_img.size(), CV_32FC3, kernel->param.mean);
            //std::cout<<kernel->param.mean[0]<<","<<kernel->param.mean[1]<<","<<kernel->param.mean[2]<<std::endl;
            cv::Mat float_img;
            bgr_img.convertTo(float_img, CV_32FC3); //4-bytes aligned!
            cv::subtract(float_img, mean_img, float_img);
            
            
            im_info[0] = float_img.rows;
            im_info[1] = float_img.cols;
            //im_info[2] = im_scale_x;
            //im_info[3] = im_scale_y;
            //im_info[4] = im_scale_x;
            //im_info[5] = im_scale_y;
            

            shared_ptr< caffe::Blob<float> > input_layer = kernel->net->blob_by_name("data");
            input_layer->Reshape(1, 3, float_img.rows, float_img.cols);
            float *input_data = input_layer->mutable_cpu_data();
            std::vector< cv::Mat > input_channels;
            for (int i = 0; i < input_layer->channels(); i++) {
                cv::Mat channel(float_img.rows, float_img.cols, CV_32FC1, input_data);
                input_channels.push_back(channel);
                input_data += float_img.rows * float_img.cols;
            }
            cv::split(float_img, input_channels);
            
            
            memcpy( kernel->net->blob_by_name("im_info")->mutable_cpu_data(), im_info, sizeof(float) * 2);

            #if 0
            std::cout<<"forward data: "<<float_img.cols<<"x"<<float_img.rows<<std::endl;
            std::cout<<im_scale_x <<","<<im_scale_y<<std::endl;
            #endif

            kernel->net->ForwardFrom(0);

            bbox_delt = kernel->net->blob_by_name("bbox_pred")->cpu_data();	// bbox_delt is offset ratio of bounding box, get by bounding box regression
            num = kernel->net->blob_by_name("rois")->num();	// number of region proposals
            rois = kernel->net->blob_by_name("rois")->cpu_data();	// scores and bounding boxes coordinate
            pred_cls = kernel->net->blob_by_name("cls_prob")->cpu_data();

            if(boxes == NULL)
            {
                boxes = new float[num*4];
                pred = new float[num*5*kernel->param.class_num];
                pred_per_class = new float[num*5];
                sorted_pred_cls = new float[num*5];
                keep = new int[num];	// index of bounding box?
            }
            for (int n = 0; n < num; n++)
            {
                boxes[n*4+0] = rois[n*5+0+1] / im_scale_x;
                boxes[n*4+1] = rois[n*5+1+1] / im_scale_y;
                boxes[n*4+2] = rois[n*5+2+1] / im_scale_x;
                boxes[n*4+3] = rois[n*5+3+1] / im_scale_y;
            }

            

            bbox_transform_inv(kernel->param.class_num,num, bbox_delt, pred_cls, boxes, pred, image.rows, image.cols);
            for (int i = 1; i < kernel->param.class_num; i ++)		// i = 0, means background
            {
                for (int j = 0; j< num; j++)
                {
                    memcpy(pred_per_class + j*5 , pred + (i * num + j) * 5 , sizeof(float) * 5);
                }
                boxes_sort(num, pred_per_class, sorted_pred_cls);
                std::vector<int> pred_index(num);
                int num_after_nms;
                caffe::nms_cpu(num, pred_per_class, pred_index.data(), &num_after_nms, 0, kernel->param.score_nms, num);

             
                for(int k = 0; k < num_after_nms; k++)
                {
                    int ind = pred_index[k];
                    TARGET target;

                    if(pred_per_class[ind * 5 + 4] < kernel->param.score_obj)
                        continue;
                    #if 0
                        std::cout<<"nms score : "<<k<<","<<ind<<","<<pred_per_class[ind * 5 + 4]<<std::endl;
                    #endif                        
                    target.class_id() = i;
                    target.score() = pred_per_class[ind * 5 + 4];
                    int x0 = pred_per_class[ind*5 + 0];
                    int y0 = pred_per_class[ind*5 + 1];
                    int x1 = pred_per_class[ind*5 + 2];
                    int y1 = pred_per_class[ind*5 + 3];
                    #if 0
                        std::cout<<"target : "<<x0<<","<<x1<<std::endl;
                    #endif
                    target.location() = cv::Rect( x0, y0, x1 - x0, y1 - y0   );
                    targets.push_back(target);
                }
            }
            delete []boxes;
            delete []pred;
            delete []pred_per_class;
            delete []keep;
            delete []sorted_pred_cls;           
            return targets.size();
        }
    
    }
}