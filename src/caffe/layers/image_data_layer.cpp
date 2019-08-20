#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/image_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {



template <typename Dtype>
ImageDataLayer<Dtype>::~ImageDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void ImageDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int new_height = this->layer_param_.image_data_param().new_height();
  const int new_width  = this->layer_param_.image_data_param().new_width();
  const bool is_color  = this->layer_param_.image_data_param().is_color();
  string root_folder = this->layer_param_.image_data_param().root_folder();

  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";
  // Read the file with filenames and labels
  const string& source = this->layer_param_.image_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string line;
  size_t pos;
  int label;


  if(top.size() == 2)
  { //path label 
    output_weights_ = false;
    while (std::getline(infile, line)) {
      pos = line.find_last_of(' ');
      label = atoi(line.substr(pos + 1).c_str());
      lines_.push_back(std::make_pair(line.substr(0, pos), label));
    }
#if 0
    Dtype weight = (Dtype)(1.0);
    for(int k = 0; k < lines_.size(); k++)
    {
      std::string path = lines_[k].first;
      typename std::map<std::string, Dtype>::iterator iter = path2weight_.find(path);
      if(iter == path2weight_.end())
        path2weight_[path]= weight;
      else
      {
        LOG(WARNING)<<"duplicated image found";
        iter->second += weight;
      }
    }
  #endif

  }
  else if (top.size() == 3)
  { // path label weight
      output_weights_ = true;

      float weight;
      while (std::getline(infile, line)) {
      pos = line.find_last_of(' ');
      weight = atof(line.substr(pos + 1).c_str());
      line[pos] = '\0';

      pos = line.find_last_of(' ');
      label = atoi(line.substr(pos + 1).c_str());

      std::string path = line.substr(0, pos);
      lines_.push_back(  std::make_pair(path, label));

      typename std::map<std::string, Dtype>::iterator iter = path2weight_.find(path);
      if(iter == path2weight_.end())
        path2weight_[path]= weight;
      else
      {
        LOG(WARNING)<<"duplicated image found";
        iter->second += weight;
      }
    }
    CHECK(lines_.size() == path2weight_.size())<<"different count between file and weight";
  }
  else
  {
    LOG(FATAL)<<"only 2 (data,label) or 3 (data,label,weight) top blob supported";
  }

  CHECK(!lines_.empty()) << "File is empty";
  
  lines_org_.clear();
  for(int k = 0; k < lines_.size(); k++)
  {
    lines_org_.push_back(lines_[k]);
  }


  if (this->layer_param_.image_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleImages();
  } 
  if( this->layer_param_.image_data_param().label_shuffle() ) 
  {
   label_shuffling();
  }
  else 
  {
    if (this->phase_ == TRAIN && Caffe::solver_rank() > 0 &&
        this->layer_param_.image_data_param().rand_skip() == 0) {
      LOG(WARNING) << "Shuffling or skipping recommended for multi-GPU";
    }
  }
  LOG(INFO) << "A total of " << lines_.size() << " images.";

  lines_id_ = 0;
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.image_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
        this->layer_param_.image_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }
  // Read an image, and use it to initialize the top blob.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
                                    new_height, new_width, is_color);
  CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
  // Use data_transformer to infer the expected blob shape from a cv_image.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  // Reshape prefetch_data and top[0] according to the batch_size.
  const int batch_size = this->layer_param_.image_data_param().batch_size();
  CHECK_GT(batch_size, 0) << "Positive batch size required";
  top_shape[0] = batch_size;
  for (int i = 0; i < this->prefetch_.size(); ++i) {
    this->prefetch_[i]->data_.Reshape(top_shape);
  }
  top[0]->Reshape(top_shape);

  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  vector<int> label_shape(1, batch_size);
  top[1]->Reshape(label_shape);
  for (int i = 0; i < this->prefetch_.size(); ++i) {
    this->prefetch_[i]->label_.Reshape(label_shape);
  }

  if(top.size() == 3)
  {
    top[2]->Reshape(label_shape);
    for (int i = 0; i < this->prefetch_.size(); ++i) {
      this->prefetch_[i]->weight_.Reshape(label_shape);
    }
  }
}

template <typename Dtype>
void ImageDataLayer<Dtype>::ShuffleImages() 
{

  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}



template <typename Dtype>
void ImageDataLayer<Dtype>::label_shuffling() 
{
  const unsigned int prefetch_rng_seed = caffe_rng_rand();
  prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));

  lines_.clear();
  std::map< int, std::vector<int> > labels_count;
  for(int k = 0; k < lines_org_.size(); k++)
  {
    int label = lines_org_[k].second;
    std::map<int,std::vector<int> >::iterator iter = labels_count.find(label);
    if(iter == labels_count.end())
    {
      labels_count[label].clear();
      labels_count[label].push_back(k);
    }
    else
    {
      labels_count[label].push_back(k);
    }
  }

  int max_num = 0;
  for(int k = 0; k < labels_count.size(); k++)
  {
    if(max_num < labels_count[k].size())
      max_num = labels_count[k].size();
  }


  int total_req = max_num;
  float* inds = new float[total_req];
  for(int k = 0; k < labels_count.size(); k++)
  {
    //LOG(INFO)<<"label_shuffle for class:"<<k <<" "<<total_req;
    caffe_rng_uniform(total_req, 0.0f, (INT_MAX - 1) * 1.0f, inds);
    for(int j = 0; j < total_req; j++)
    {
      int tmp = (int) inds[j];
      int ind = labels_count[k][  tmp % labels_count[k].size()  ];
      //LOG(INFO)<<"label_shuffle: "<<k<<","<<tmp<<","<<labels_count[k].size() <<","<<ind << "/" << lines_org_.size() <<"," << lines_.size();
      //LOG(INFO)<<"\t"<<lines_org_[ind].first<<","<<lines_org_[ind].second;
      lines_.push_back( lines_org_[ind]   );
      if(lines_org_[ind].second != k)
      {
        LOG(INFO)<<"label shuffle error: "<< k <<" -> " << lines_org_[ind].second ;
      }
    }

  }
  delete[] inds;

  //LOG(INFO)<<"label shuffle result: "<< lines_.size();
  ShuffleImages();
  return;
}

// This function is called on prefetch thread
template <typename Dtype>
void ImageDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());
  ImageDataParameter image_data_param = this->layer_param_.image_data_param();
  const int batch_size = image_data_param.batch_size();
  const int new_height = image_data_param.new_height();
  const int new_width = image_data_param.new_width();
  const bool is_color = image_data_param.is_color();
  string root_folder = image_data_param.root_folder();

  // Reshape according to the first image of each batch
  // on single input batches allows for inputs of varying dimension.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
      new_height, new_width, is_color);
  CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
  // Use data_transformer to infer the expected blob shape from a cv_img.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);

 // LOG(INFO)<<"batch->data_.shape:"<<batch->data_.shape_string();
  //LOG(INFO)<<"batch->weight_.shape:"<<batch->weight_.shape_string();

  Dtype* prefetch_label = batch->label_.mutable_cpu_data();
  Dtype* prefetch_data = batch->data_.mutable_cpu_data();
  Dtype* prefetch_weight;
  if(this->output_weights_)
    prefetch_weight = batch->weight_.mutable_cpu_data();

  // datum scales
  const int lines_size = lines_.size();
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    timer.Start();
    CHECK_GT(lines_size, lines_id_);
    cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
        new_height, new_width, is_color);
    CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply transformations (mirror, crop...) to the image
    int offset = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(prefetch_data + offset);
    this->data_transformer_->Transform(cv_img, &(this->transformed_data_));
    trans_time += timer.MicroSeconds();

    prefetch_label[item_id] = lines_[lines_id_].second;
    if(this->output_weights_)
    {
      prefetch_weight[item_id] = path2weight_[ lines_[lines_id_].first ];
      //LOG(INFO)<<"prefetch weight: "<<prefetch_weight[item_id];
    }

    // go to the next iter
    lines_id_++;
    if (lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (this->layer_param_.image_data_param().shuffle()) {
        ShuffleImages();
      }

      if(this->layer_param_.image_data_param().label_shuffle() ) 
      {
        label_shuffling();
      }

    }
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(ImageDataLayer);
REGISTER_LAYER_CLASS(ImageData);

}  // namespace caffe
#endif  // USE_OPENCV
