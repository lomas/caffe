// TODO (sergeyk): effect should not be dependent on phase. wasted memcpy.


#include <vector>

#include "caffe/layers/spatial_dropout_layer.hpp"
#include "caffe/util/math_functions.hpp"


namespace caffe {

template <typename Dtype>
void SpatialDropoutLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
  //CHECK(this->layer_param_.has_spatial_dropout_param());
  threshold_ = this->layer_param_.spatial_dropout_param().dropout_ratio();
  DCHECK(threshold_ > 0.);
  DCHECK(threshold_ < 1.);
  scale_ = 1. / (1. - threshold_);
  uint_thres_ = static_cast<unsigned int>(UINT_MAX * threshold_);
}

template <typename Dtype>
void SpatialDropoutLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::Reshape(bottom, top);
  // Set up the cache for random number generation
  // ReshapeLike does not work because rand_vec_ is of Dtype uint
  
  rand_vec_.Reshape(bottom[0]->shape(0), bottom[0]->shape(1), 1, 1);
  top[0]->Reshape(bottom[0]->shape()); 
}

template <typename Dtype>
void SpatialDropoutLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	const int count = bottom[0]->count();//����*ͨ����*�߶�*����
  //��ֻ������������Blob��dataָ�룬data��ǰ�򴫲��������� 
  const Dtype* bottom_data = bottom[0]->cpu_data();
  //����д����������Blob��dataָ�� 
  Dtype* top_data = top[0]->mutable_cpu_data();
  //����д�����rand_vec��Blob��dataָ�� 
  unsigned int* mask = rand_vec_.mutable_cpu_data();
  //��������BlobԪ�ظ���  
  const int num = bottom[0]->num();
  //��������BlobԪ�ص�ͨ����
  const int channel = bottom[0]->channels();
  //����ͼ��ͨ��ռ�Ŀռ��С
  const int img_size = bottom[0]->height() * bottom[0]->width();
  size_t spatial_dim = bottom[0]->count() / (num * channel);
  //�����trainģʽ����ʼ����
  if (this->phase_ == TRAIN) {
    // Create random numbers
    caffe_rng_bernoulli(num * channel, 1. - threshold_, mask);
#ifdef _OPENMP
    #pragma omp parallel for collapse(2)
#endif
    for (int i = 0; i < num; ++i) {
      for (int j = 0; j < channel; j++) {
        int offset = (i * channel + j) * img_size;
        if (mask[i * channel + j] == 1) {
          caffe_cpu_axpby(img_size, this->scale_ , bottom_data + offset, (Dtype)0, top_data + offset);
        } else {
          caffe_set(img_size, (Dtype)0, top_data + offset);
        }
      }
    }
  } else {
	  DLOG(INFO) << "12332432323123213123123123123123123123123 ";
      caffe_copy(count, bottom_data, top_data);
	 //caffe_cpu_axpby(count, this->scale_ , bottom_data + count, (Dtype)0, top[0]->data + count);
  }
}

template <typename Dtype>
void SpatialDropoutLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
    //������´���
  if (propagate_down[0]) {
    //��ֻ�������������diffָ�룬diff�����򴫲���������
    const Dtype* top_diff = top[0]->cpu_diff(); 
	//����д�����������diffָ�룬diff�����򴫲���������
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
	//��������BlobԪ�ظ���  
    const int num = bottom[0]->num();
	//��������BlobԪ�ص�ͨ���ĸ���  
    const int channel = bottom[0]->channels();
	////����ͼ��ռ�Ŀռ��С
    const int img_size = bottom[0]->height() * bottom[0]->width();
    if (this->phase_ == TRAIN) { //�����ѵ��ģʽ
      const unsigned int* mask = rand_vec_.cpu_data(); //ָ������
#ifdef _OPENMP
    #pragma omp parallel for collapse(2)
#endif
      for (int i = 0; i < num; ++i) {
        for (int j = 0; j < channel; j++) {
          int offset = (i * channel + j) * img_size;
          if (mask[i * channel + j] == 1) {
            caffe_cpu_axpby(img_size, scale_, top_diff + offset, (Dtype)0, bottom_diff + offset);
          } else {
            caffe_set(img_size, (Dtype)0, bottom_diff + offset);
          }
        }
      }
    } else {
      caffe_copy(bottom[0]->count(), top_diff, bottom_diff);
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(SpatialDropoutLayer);
#endif

INSTANTIATE_CLASS(SpatialDropoutLayer);
REGISTER_LAYER_CLASS(SpatialDropout);

}  // namespace caffe