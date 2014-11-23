// Copyright 2014 BVLC and contributors.

#include <cstring>
#include <vector>

#include "cuda_runtime.h"
#include "gtest/gtest.h"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;

template <typename Dtype>
class MaskWeightLayerTest : public ::testing::Test {
 protected:
  MaskWeightLayerTest()
      : blob_bottom_(new Blob<Dtype>()),
		blob_bottom_mask_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()){}
  virtual void SetUp() {
    blob_bottom_->Reshape(1, 3, 6, 4);
	blob_bottom_mask_->Reshape(1, 1, 6, 4);
    // fill the values
    FillerParameter filler_param;
    filler_param.set_value(1.);
    
	GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
	filler.Fill(this->blob_bottom_mask_);

	
	const Dtype* bottom_data = this->blob_bottom_->cpu_data();
	const Dtype* mask_data = this->blob_bottom_mask_->cpu_data();
	for (int i = 0; i < this->blob_bottom_->count(); ++i) {
		cout<<bottom_data[i]<<'\t';
	}
	cout<<endl;
	for (int i = 0; i < this->blob_bottom_mask_->count(); ++i) {
		cout<<mask_data[i]<<'\t';
	}
	cout<<endl;

    blob_bottom_vec_.push_back(blob_bottom_);
	blob_bottom_vec_.push_back(blob_bottom_mask_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~MaskWeightLayerTest() { delete blob_bottom_; delete blob_top_; }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_bottom_mask_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

typedef ::testing::Types<float, double> Dtypes;
TYPED_TEST_CASE(MaskWeightLayerTest, Dtypes);

TYPED_TEST(MaskWeightLayerTest, TestSetup) {
  LayerParameter layer_param;
  MaskWeightParameter* mask_weight_param =
	  layer_param.mutable_mask_weight_param();
  mask_weight_param->mutable_weight_filler()->set_type("constant");
  shared_ptr<Layer<TypeParam> > layer(
      new MaskWeightLayer<TypeParam>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 1);
  EXPECT_EQ(this->blob_top_->height(), 6);
  EXPECT_EQ(this->blob_top_->width(), 4);
}

TYPED_TEST(MaskWeightLayerTest, TestCPUForward) {
  // We will simply see if the convolution layer carries out averaging well.
  FillerParameter filler_param;
  filler_param.set_value(1.);
  ConstantFiller<TypeParam> filler(filler_param);
  filler.Fill(this->blob_bottom_);
  LayerParameter layer_param;
  MaskWeightParameter* mask_weight_param =
      layer_param.mutable_mask_weight_param();
  mask_weight_param->mutable_weight_filler()->set_type("constant");
  mask_weight_param->mutable_weight_filler()->set_value(0.1);

  shared_ptr<Layer<TypeParam> > layer(
      new MaskWeightLayer<TypeParam>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  Caffe::set_mode(Caffe::CPU);
  layer->Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  
  const TypeParam* top_data = this->blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], 0.3, 1e-5);
  }
}

TYPED_TEST(MaskWeightLayerTest, TestCPUGradient) {
  LayerParameter layer_param;
  MaskWeightParameter* mask_weight_param =
      layer_param.mutable_mask_weight_param();
  mask_weight_param->mutable_weight_filler()->set_type("gaussian");
  mask_weight_param->mutable_weight_filler()->set_value(0.1);

  Caffe::set_mode(Caffe::CPU);
  MaskWeightLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_));
}

}  // namespace caffe