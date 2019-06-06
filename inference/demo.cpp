#include <stdio.h>  // for snprintf
#include <string>
#include <vector>
#include <math.h>
#include <fstream>
#include <iterator>
#include "inference.hpp"
#include "sys/times.h"
#include "unistd.h"

class TIMER
{
    double m_t0;
    double m_hz;
    public:
    TIMER()
    {
        m_hz = sysconf( _SC_CLK_TCK ); 
        m_t0 = times(NULL);
    }
    ~TIMER()
    {

    }
    public:
    double cost()
    {
        double t = (times(NULL) - m_t0) / m_hz;
        m_t0 = times(NULL);
        return t;
    }
};



int run_pvanet(std::string config_file,std::vector< std::string >& path_list, std::string outdir)
{
    inference::pvanet::PARAM param;


    std::string model_file, weight_file;
    cv::FileNode mean;
    
    cv::FileStorage fs(config_file, cv::FileStorage::READ);
    fs["num_classes"] >> param.class_num;
    fs["model"] >> model_file;
    fs["weight"] >> weight_file;
    mean = fs["mean"];
    
    fs["scale"] >> param.scale_depth;
    fs["size_min"] >> param.size_min;
    fs["size_max"] >> param.size_max;
    fs["score_nms"] >> param.score_nms;
    fs["score_obj"] >> param.score_obj;
    fs.release();

    {
        cv::FileNodeIterator itr = mean.begin();
        std::vector<float> tmp;
        for(itr = mean.begin(); itr != mean.end(); itr++)
        {
            tmp.push_back(*itr);
            
        }
        param.mean[0] = tmp[0];
        param.mean[1] = tmp[1];
        param.mean[2] = tmp[2];
    }


    inference::pvanet::PVANET net(model_file, weight_file,param);
    //std::cout<<"ok"<<std::endl;

    double time_cost = 0;
    double pixel_num = 0;
    TIMER timer;
    for(int ind_path = 0; ind_path < path_list.size(); ind_path++)
    {
        std::cout<<path_list[ind_path]<<std::endl;
        cv::Mat img = cv::imread(path_list[ind_path],1);

        if (img.empty()) continue;
        //cv::resize(img,img,cv::Size(224 * 5,224 * 5));        


        std::vector< inference::pvanet::TARGET > targets;

        timer.cost();
        net.forward(img, targets);
        pixel_num += (img.cols * img.rows) / (float)1000000;
        time_cost += timer.cost();

        //if(targets.empty()) continue;

        //std::cout<<"demo targets num = "<<targets.size()<<std::endl;
        for(int k = 0; k < targets.size(); k++)
        {
          //  std::cout<<targets[k].location().x<<","<<targets[k].location().width<<std::endl;
            cv::rectangle(img, targets[k].location(), CV_RGB(255,0,0),2,8,0);
        }

        int pos = path_list[ind_path].rfind('/') + 1;
        std::string filename = path_list[ind_path].substr(pos);
        std::ostringstream oss;
        oss<<outdir<<"/"<<filename;
        cv::imwrite(oss.str(),img);
    }
    std::cout<<"Second/MPixel= "<<time_cost / pixel_num<<std::endl;
    return 0;
}


int run_classify(std::string config_file,std::vector< std::string >& path_list, std::string outdir)
{
    inference::classify::PARAM param;


    std::string model_file, weight_file;
    cv::Size input_size;
    
    cv::FileStorage fs(config_file, cv::FileStorage::READ);
    fs["num_classes"] >> param.class_num;
    fs["model"] >> model_file;
    fs["weight"] >> weight_file;
    fs["input_width"] >> param.input_size.width;
    fs["input_height"] >> param.input_size.height;
  
    std::vector<std::string> provinces, city_codes;
    std::vector<std::string> serial_no, serial_no_last;
    
    {
        cv::FileNode fn = fs["mean"];
        cv::FileNodeIterator itr;
        std::vector<float> tmp;
        for(itr = fn.begin(); itr != fn.end(); itr++)
        {
            tmp.push_back(*itr);
            
        }
        param.mean[0] = tmp[0];
        param.mean[1] = tmp[1];
        param.mean[2] = tmp[2];
    }
    
    {
        cv::FileNode fn = fs["province"];
        cv::FileNodeIterator itr;
        for(itr = fn.begin(); itr != fn.end(); itr++)
        {
            provinces.push_back(*itr);   
        }   
    }

    {
        cv::FileNode fn = fs["city_code"];
        cv::FileNodeIterator itr;
        //std::cout<<"city_code"<<std::endl;
        for(itr = fn.begin(); itr != fn.end(); itr++)
        {
            city_codes.push_back(*itr);  
            //std::cout<<city_codes.back(); 
        }   
        //std::cout<<std::endl;
    }

    {
        cv::FileNode fn = fs["serial_no"];
        cv::FileNodeIterator itr;
        //std::cout<<"serial_no"<<std::endl;
        for(itr = fn.begin(); itr != fn.end(); itr++)
        {
            serial_no.push_back(*itr);   
          //  std::cout<<serial_no.back();
        }   
        //std::cout<<std::endl;
    }
    {
        cv::FileNode fn = fs["serial_no_last"];
        cv::FileNodeIterator itr;
        for(itr = fn.begin(); itr != fn.end(); itr++)
        {
            serial_no_last.push_back(*itr);   
        }   
    }

    fs.release();

    inference::classify::CLASSIFY net(model_file, weight_file,param);
    //std::cout<<"ok"<<std::endl;

    const int field_num = 7;
    int seperator_width[field_num] = {
        29,25,34,34,34,34,37
    };

    double time_cost = 0;
    double pixel_num = 0;
    TIMER timer;
    for(int ind_path = 0; ind_path < path_list.size(); ind_path++)
    {
        std::cout<<path_list[ind_path]<<std::endl;
        cv::Mat img = cv::imread(path_list[ind_path],1);

        if (img.empty()) continue;
       // cv::resize(img,img,cv::Size(256,64));        


        cv::Mat mat_score;
        std::vector< cv::Mat > imgs;
        imgs.push_back(img);

        
        
        timer.cost();
        net.forward(imgs, mat_score,1);
            
        std::vector< cv::Mat > results_split;
        {
            int begin = 0;
            
            for(int k = 0; k < field_num; k++)
            {
                cv::Rect roi(begin, 0, seperator_width[k], mat_score.rows  );
               // std::cout<<seperator_width[k]<<","<<mat_score.cols<<","<<roi.x<<","<<roi.x + roi.width<<","<<roi.y<<","<<roi.y + roi.height<<std::endl;
                results_split.push_back(mat_score(roi).clone());
                begin += seperator_width[k];
            }
        }
        std::vector< int > inds_class;
        std::vector< float> scores_class;
        std::vector< std::string > str_class;
        {
            for(int ind_field = 0; ind_field < results_split.size(); ind_field++)
            {
                cv::Mat mat = results_split[ind_field];
                double score_max;
                cv::Point loc_max;
                cv::minMaxLoc(mat, NULL, &score_max, NULL, &loc_max);
                inds_class.push_back(loc_max.x);
                if(ind_field == 0)
                    str_class.push_back(provinces[loc_max.x]);
                else if(ind_field == 1)
                    str_class.push_back(city_codes[loc_max.x]);
                else if(ind_field + 1 == results_split.size())
                    str_class.push_back(serial_no_last[loc_max.x]);
                else
                {
                    str_class.push_back(serial_no[loc_max.x]);
                }
                
                scores_class.push_back(mat.at<float>(loc_max));
                //std::cout<<ind_field<<",("<<loc_max.x<<","<<loc_max.y<<")"<<str_class.back()<<std::endl;
            }
        }

        pixel_num += (img.cols * img.rows) / (float)1000000;
        time_cost += timer.cost();

        std::ostringstream oss;
        oss<<outdir<<"/"<<ind_path<<",";
        float score = 0;
        for(int k = 0; k < str_class.size(); k++)
        {
            oss<<str_class[k];
            score += scores_class[k];
        }
        score /= str_class.size();
        oss<<"," << score <<".jpg";
        //std::cout<<oss.str()<<std::endl;
        cv::imwrite(oss.str(),img);
  

    }
    std::cout<<"Second/MPixel= "<<time_cost / pixel_num<<std::endl;
    return 0;
}

const char* keys = {

	"{ input    |               | input file list               }"
	"{ config   | config.yml    | config file               }"
	"{ outdir   |  ""           | input dir                   }"
};	


struct Line
{
	std::string data;
	friend std::istream& operator >> (std::istream& is, Line& l)
	{
		std::getline(is, l.data);
		return is;
	}
	operator std::string() const { return data; }
};


int main(int argc, const char* argv[])
{
    cv::CommandLineParser parser(argc, argv, keys);


	std::string input = parser.get<std::string>("input");
	std::string outdir = parser.get<std::string>("outdir");
    std::string config_path = parser.get<std::string>("config");



    std::vector<std::string> paths;
	if ((input.rfind(".txt") != std::string::npos) || (input.rfind(".") == std::string::npos))
	{
		std::ifstream file(input.c_str());
		std::vector<std::string> temp;
		std::copy(std::istream_iterator<Line>(file), std::istream_iterator<Line>(), std::back_inserter(temp));
		for (int k = 0; k < temp.size(); k++)
		{
            std::string name = temp[k];
            if(name == "" || name == "\r" || name == "\n" || name == "#") break;
			if (name.rfind(".jpg") == std::string::npos) continue;
			paths.push_back(name);
		}

	}
	else
	{
		paths.push_back(input);
	}



    inference::ENV env(0);

//    run_pvanet(config_path,paths,outdir);
    run_classify(config_path, paths, outdir);



}
