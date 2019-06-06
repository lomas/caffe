#include "inference.hpp"
#include "caffe/caffe.hpp"

namespace inference
{

    ENV::ENV(int device)
    {
        setup(device);
    }


    ENV::~ENV()
    {

    }

    void ENV::setup(int device)
    {
        caffe::Caffe::SetDevice(device);
        if(device >= 0)
            caffe::Caffe::set_mode(caffe::Caffe::GPU);
        else
        {
            caffe::Caffe::set_mode(caffe::Caffe::CPU);
        }
        return;
    }
        
}