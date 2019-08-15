#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/softmax_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void SoftmaxLossForwardGPU(const int nthreads,
          const Dtype* prob_data, const Dtype* label, Dtype* loss,
          const Dtype* class_weight_vec, const Dtype* sample_weight_vec,
          const int num, const int dim, const int spatial_dim,
          const bool has_ignore_label_, const int ignore_label_,
          Dtype* counts,
          bool use_label_smooth, float label_smooth, int class_num) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int label_value = static_cast<int>(label[n * spatial_dim + s]);
    Dtype weight_value = class_weight_vec[label_value];
    if(sample_weight_vec != NULL)
        weight_value *= static_cast<Dtype>(sample_weight_vec[n*spatial_dim + s]);
    if (has_ignore_label_ && label_value == ignore_label_) {
      loss[index] = 0;
      counts[index] = 0;
    } else {

      if(use_label_smooth)
      {
        loss[index] = 0;
        for(int c = 0; c < class_num; c++)
        {
          float w = weight_value * ( (c == label_value)?(1.F - label_smooth):(label_smooth / float(class_num - 1)) );
          loss[index] += -w * log(max(prob_data[n * dim + c * spatial_dim + s],
            Dtype(FLT_MIN)));
          
        }
        counts[index] = weight_value;
      }
      else
      {
        loss[index] = -weight_value * log(max(prob_data[n * dim + label_value * spatial_dim + s],
                        Dtype(FLT_MIN)));
        counts[index] = weight_value;
      }
    }
  }
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.gpu_data();
  const Dtype* label = bottom[1]->gpu_data();
  const Dtype* class_weight_vec = class_weights_.gpu_data();
  const int dim = prob_.count() / outer_num_;
  const int nthreads = outer_num_ * inner_num_;
  // Since this memory is not used for anything, we use it here to avoid having
  // to allocate new GPU memory to accumulate intermediate results.
  Dtype* loss_data = bottom[0]->mutable_gpu_diff();
  // Similarly, this memory is never used elsewhere, and thus we can use it
  // to avoid having to allocate additional GPU memory.
  Dtype* counts = prob_.mutable_gpu_diff();
  int num_class = bottom[0]->shape(softmax_axis_);
  // NOLINT_NEXT_LINE(whitespace/operators)
  //LOG(INFO)<<"class num :"<<num_class <<" softmax_axis:" << softmax_axis_;
  if(bottom.size() == 2)
  {
    SoftmaxLossForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, prob_data, label, loss_data, class_weight_vec, NULL,
        outer_num_, dim, inner_num_, has_ignore_label_, ignore_label_, counts, use_label_smooth_, label_smooth_, num_class);
  }
  else if(bottom.size() == 3)
  {
    const Dtype* sample_weight = bottom[2]->gpu_data();
    //const Dtype* sample_weight_cpu = bottom[2]->cpu_data();
    //const Dtype* label_cpu = bottom[1]->cpu_data();
    //LOG(INFO)<<"softmax weigth:"<<sample_weight_cpu[0]<<","<<sample_weight_cpu[20];
    
    SoftmaxLossForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, prob_data, label, loss_data, class_weight_vec, sample_weight,
        outer_num_, dim, inner_num_, has_ignore_label_, ignore_label_, counts,use_label_smooth_, label_smooth_, num_class);
  }
  Dtype loss;
  caffe_gpu_asum(nthreads, loss_data, &loss);
  Dtype valid_count = -1;
  // Only launch another CUDA kernel if we actually need the count of valid
  // outputs.
  if (normalization_ == LossParameter_NormalizationMode_VALID &&
      has_ignore_label_) {
    caffe_gpu_asum(nthreads, counts, &valid_count);
  }
  top[0]->mutable_cpu_data()[0] = loss / get_normalizer(normalization_,
                                                        valid_count);
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }

  // Clear scratch memory to prevent interfering with backward (see #6202).
  caffe_gpu_set(bottom[0]->count(), Dtype(0), bottom[0]->mutable_gpu_diff());
}

template <typename Dtype>
__global__ void SoftmaxLossBackwardGPU(const int nthreads, const Dtype* top,
          const Dtype* label, Dtype* bottom_diff, 
          const Dtype* class_weight_vec, const Dtype* sample_weight_vec,
          const int num, const int dim,
          const int spatial_dim, const bool has_ignore_label_,
          const int ignore_label_, Dtype* counts,
          bool use_label_smooth, float label_smooth, int class_num) {
  const int channels = dim / spatial_dim;

  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int label_value = static_cast<int>(label[n * spatial_dim + s]);
    Dtype weight_value = class_weight_vec[label_value];
    if(sample_weight_vec != NULL)
        weight_value *= static_cast<Dtype>(sample_weight_vec[n*spatial_dim + s]);   
    if (has_ignore_label_ && label_value == ignore_label_) {
      for (int c = 0; c < channels; ++c) {
        bottom_diff[n * dim + c * spatial_dim + s] = 0;
      }
      counts[index] = 0;
    } else {
      
      if(use_label_smooth)
      {
        //bottom_diff[n * dim + label_value * spatial_dim + s] -= 1;
        for(int c = 0; c < channels; c++)
        {
          float w = weight_value * ( (c == label_value)?(1.F - label_smooth):(label_smooth / float(class_num - 1)) );
          bottom_diff[n * dim + c * spatial_dim + s] -= w;
        }
        counts[index] = 1;
      }
      else
      {
        //bottom_diff[n * dim + label_value * spatial_dim + s] -= 1;
        for(int c = 0; c < channels; c++)
          bottom_diff[n * dim + c * spatial_dim + s] -= weight_value;
        counts[index] = 1;
      }
    }
  }
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* prob_data = prob_.gpu_data();
    const Dtype* top_data = top[0]->gpu_data();
    const Dtype* class_weight_vec = class_weights_.gpu_data();
    caffe_gpu_memcpy(prob_.count() * sizeof(Dtype), prob_data, bottom_diff);
    const Dtype* label = bottom[1]->gpu_data();
    const int dim = prob_.count() / outer_num_;
    const int nthreads = outer_num_ * inner_num_;
    // Since this memory is never used for anything else,
    // we use to to avoid allocating new GPU memory.
    Dtype* counts = prob_.mutable_gpu_diff();
    int num_class = bottom[0]->shape(softmax_axis_);
    // NOLINT_NEXT_LINE(whitespace/operators)
    if(bottom.size() == 2)
    {
      SoftmaxLossBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
                CAFFE_CUDA_NUM_THREADS>>>(nthreads, top_data, label, bottom_diff, class_weight_vec, NULL,
                outer_num_, dim, inner_num_, has_ignore_label_, ignore_label_, counts,use_label_smooth_, label_smooth_, num_class);
    }
    else if(bottom.size() == 3)
    {

      const Dtype* sample_weight = bottom[2]->gpu_data();
      SoftmaxLossBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
          CAFFE_CUDA_NUM_THREADS>>>(nthreads, top_data, label, bottom_diff, class_weight_vec, sample_weight,
          outer_num_, dim, inner_num_, has_ignore_label_, ignore_label_, counts,use_label_smooth_, label_smooth_, num_class);
    }
    Dtype valid_count = -1;
    // Only launch another CUDA kernel if we actually need the count of valid
    // outputs.
    if (normalization_ == LossParameter_NormalizationMode_VALID &&
        has_ignore_label_) {
      caffe_gpu_asum(nthreads, counts, &valid_count);
    }
    const Dtype loss_weight = top[0]->cpu_diff()[0] /
                              get_normalizer(normalization_, valid_count);
    caffe_gpu_scal(prob_.count(), loss_weight , bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SoftmaxWithLossLayer);

}  // namespace caffe
