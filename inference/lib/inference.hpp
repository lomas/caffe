#pragma once
#include "pvanet.hpp"
#include "classify.hpp"




namespace inference
{
    class ENV
    {
        public:
        ENV(int device);
        ~ENV();
        private:
        void setup(int device);
    };
}