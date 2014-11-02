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
class ColinearConvolutionLayerTest : public ::testing::Test {
 protected:
  ColinearConvolutionLayerTest()
      : blob_bottom_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    blob_bottom_->Reshape(2, 3, 6, 4);
    // fill the values
    FillerParameter filler_param;
    filler_param.set_value(1.);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~ColinearConvolutionLayerTest() { delete blob_bottom_; delete blob_top_; }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

typedef ::testing::Types<float, double> Dtypes;
TYPED_TEST_CASE(ColinearConvolutionLayerTest, Dtypes);

TYPED_TEST(ColinearConvolutionLayerTest, TestSetup) {
  LayerParameter layer_param;
  ColinearConvolutionParameter* convolution_param =
	  layer_param.mutable_colinear_convolution_param();
  convolution_param->set_kernel_size(3);
  convolution_param->set_stride(2);
  convolution_param->set_num_output(4);
  shared_ptr<Layer<TypeParam> > layer(
      new ColinearConvolutionLayer<TypeParam>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 4);
  EXPECT_EQ(this->blob_top_->height(), 2);
  EXPECT_EQ(this->blob_top_->width(), 1);
  // setting group should not change the shape
  convolution_param->set_num_output(3);
  convolution_param->set_group(3);
  layer.reset(new ColinearConvolutionLayer<TypeParam>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 3);
  EXPECT_EQ(this->blob_top_->height(), 2);
  EXPECT_EQ(this->blob_top_->width(), 1);
}

//************ Factorize ************
TYPED_TEST(ColinearConvolutionLayerTest, TestCPUSimpleConvolutionFactor) {
  // We will simply see if the convolution layer carries out averaging well.
  FillerParameter filler_param;
  filler_param.set_value(1.);
  ConstantFiller<TypeParam> filler(filler_param);
  filler.Fill(this->blob_bottom_);
  LayerParameter layer_param;
  ColinearConvolutionParameter* convolution_param =
      layer_param.mutable_colinear_convolution_param();
  convolution_param->set_kernel_size(3);
  convolution_param->set_stride(2);
  convolution_param->set_num_output(4);
  convolution_param->mutable_quadratic_weight_filler()->set_type("constant");
  convolution_param->mutable_quadratic_weight_filler()->set_value(1);
  convolution_param->mutable_linear_weight_filler()->set_type("constant");
  convolution_param->mutable_linear_weight_filler()->set_value(1);
  convolution_param->mutable_bias_filler()->set_type("constant");
  convolution_param->mutable_bias_filler()->set_value(0.1);
  convolution_param->set_symmetric(true);
  convolution_param->set_factor(2);
  shared_ptr<Layer<TypeParam> > layer(
      new ColinearConvolutionLayer<TypeParam>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  Caffe::set_mode(Caffe::CPU);
  layer->Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  // After the convolution, the output should all have output values 27.1
  const TypeParam* top_data = this->blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], 27*27*2+27+0.1, 1e-4);
  }
}

#if 0
TYPED_TEST(ColinearConvolutionLayerTest, TestGPUSimpleConvolutionFactor) {
  // We will simply see if the convolution layer carries out averaging well.
  FillerParameter filler_param;
  filler_param.set_value(1.);
  ConstantFiller<TypeParam> filler(filler_param);
  filler.Fill(this->blob_bottom_);
  LayerParameter layer_param;
  ColinearConvolutionParameter* convolution_param =
      layer_param.mutable_colinear_convolution_param();
  convolution_param->set_kernel_size(3);
  convolution_param->set_stride(2);
  convolution_param->set_num_output(4);
  convolution_param->mutable_quadratic_weight_filler()->set_type("constant");
  convolution_param->mutable_quadratic_weight_filler()->set_value(1);
  convolution_param->mutable_linear_weight_filler()->set_type("constant");
  convolution_param->mutable_linear_weight_filler()->set_value(1);
  convolution_param->mutable_bias_filler()->set_type("constant");
  convolution_param->mutable_bias_filler()->set_value(0.1);
  convolution_param->set_symmetric(true);
  convolution_param->set_factor(2);
  shared_ptr<Layer<TypeParam> > layer(
      new ColinearConvolutionLayer<TypeParam>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  Caffe::set_mode(Caffe::GPU);
  layer->Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  // After the convolution, the output should all have output values 27.1
  const TypeParam* top_data = this->blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], 27*27*2+27+0.1, 1e-4);
  }
}
#endif

TYPED_TEST(ColinearConvolutionLayerTest, TestCPUGradientFactor) {
  LayerParameter layer_param;
  ColinearConvolutionParameter* convolution_param =
      layer_param.mutable_colinear_convolution_param();
  convolution_param->set_kernel_size(3);
  convolution_param->set_stride(2);
  convolution_param->set_num_output(2);
  convolution_param->mutable_quadratic_weight_filler()->set_type("gaussian");
  convolution_param->mutable_linear_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("gaussian");
  convolution_param->set_symmetric(true);
  convolution_param->set_factor(2);
  Caffe::set_mode(Caffe::CPU);
  ColinearConvolutionLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_));
}

#if 0
TYPED_TEST(ColinearConvolutionLayerTest, TestGPUGradientFactor) {
  LayerParameter layer_param;
  ColinearConvolutionParameter* convolution_param =
      layer_param.mutable_colinear_convolution_param();
  convolution_param->set_kernel_size(3);
  convolution_param->set_stride(2);
  convolution_param->set_num_output(2);
  convolution_param->mutable_quadratic_weight_filler()->set_type("gaussian");
  convolution_param->mutable_linear_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("gaussian");
  convolution_param->set_symmetric(true);
  convolution_param->set_factor(2);
  Caffe::set_mode(Caffe::GPU);
  ColinearConvolutionLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_));
}
#endif

TYPED_TEST(ColinearConvolutionLayerTest, TestCPUGradientOnlyQFactor) {
  LayerParameter layer_param;
  ColinearConvolutionParameter* convolution_param =
      layer_param.mutable_colinear_convolution_param();
  convolution_param->set_kernel_size(3);
  convolution_param->set_stride(2);
  convolution_param->set_num_output(2);
  convolution_param->set_linear_term(false);
  convolution_param->mutable_quadratic_weight_filler()->set_type("gaussian");
  convolution_param->mutable_linear_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("gaussian");
  convolution_param->set_symmetric(true);
  convolution_param->set_factor(2);
  Caffe::set_mode(Caffe::CPU);
  ColinearConvolutionLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_));
}

#if 0
TYPED_TEST(ColinearConvolutionLayerTest, TestGPUGradientOnlyQFactor) {
  LayerParameter layer_param;
  ColinearConvolutionParameter* convolution_param =
      layer_param.mutable_colinear_convolution_param();
  convolution_param->set_kernel_size(3);
  convolution_param->set_stride(2);
  convolution_param->set_num_output(2);
  convolution_param->set_linear_term(false);
  convolution_param->mutable_quadratic_weight_filler()->set_type("gaussian");
  convolution_param->mutable_linear_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("gaussian");
  convolution_param->set_symmetric(true);
  convolution_param->set_factor(2);
  Caffe::set_mode(Caffe::GPU);
  ColinearConvolutionLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_));
}
#endif

TYPED_TEST(ColinearConvolutionLayerTest, TestCPUGradientOnlyLFactor) {
  LayerParameter layer_param;
  ColinearConvolutionParameter* convolution_param =
      layer_param.mutable_colinear_convolution_param();
  convolution_param->set_kernel_size(3);
  convolution_param->set_stride(2);
  convolution_param->set_num_output(2);
  convolution_param->set_quadratic_term(false);
  convolution_param->mutable_quadratic_weight_filler()->set_type("gaussian");
  convolution_param->mutable_linear_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("gaussian");
  convolution_param->set_symmetric(true);
  convolution_param->set_factor(2);
  Caffe::set_mode(Caffe::CPU);
  ColinearConvolutionLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_));
}

#if 0
TYPED_TEST(ColinearConvolutionLayerTest, TestGPUGradientOnlyLFactor) {
  LayerParameter layer_param;
  ColinearConvolutionParameter* convolution_param =
      layer_param.mutable_colinear_convolution_param();
  convolution_param->set_kernel_size(3);
  convolution_param->set_stride(2);
  convolution_param->set_num_output(2);
  convolution_param->set_quadratic_term(false);
  convolution_param->mutable_quadratic_weight_filler()->set_type("gaussian");
  convolution_param->mutable_linear_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("gaussian");
  convolution_param->set_symmetric(true);
  convolution_param->set_factor(2);
  Caffe::set_mode(Caffe::GPU);
  ColinearConvolutionLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_));
}
#endif
//************ Symmetric ************
TYPED_TEST(ColinearConvolutionLayerTest, TestCPUSimpleConvolutionSymm) {
  // We will simply see if the convolution layer carries out averaging well.
  FillerParameter filler_param;
  filler_param.set_value(1.);
  ConstantFiller<TypeParam> filler(filler_param);
  filler.Fill(this->blob_bottom_);
  LayerParameter layer_param;
  ColinearConvolutionParameter* convolution_param =
      layer_param.mutable_colinear_convolution_param();
  convolution_param->set_kernel_size(3);
  convolution_param->set_stride(2);
  convolution_param->set_num_output(4);
  convolution_param->mutable_quadratic_weight_filler()->set_type("constant");
  convolution_param->mutable_quadratic_weight_filler()->set_value(1);
  convolution_param->mutable_linear_weight_filler()->set_type("constant");
  convolution_param->mutable_linear_weight_filler()->set_value(1);
  convolution_param->mutable_bias_filler()->set_type("constant");
  convolution_param->mutable_bias_filler()->set_value(0.1);
  convolution_param->set_symmetric(true);
  shared_ptr<Layer<TypeParam> > layer(
      new ColinearConvolutionLayer<TypeParam>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  Caffe::set_mode(Caffe::CPU);
  layer->Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  // After the convolution, the output should all have output values 27.1
  const TypeParam* top_data = this->blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], 27*27+27+0.1, 1e-4);
  }
}

TYPED_TEST(ColinearConvolutionLayerTest, TestGPUSimpleConvolutionSymm) {
  // We will simply see if the convolution layer carries out averaging well.
  FillerParameter filler_param;
  filler_param.set_value(1.);
  ConstantFiller<TypeParam> filler(filler_param);
  filler.Fill(this->blob_bottom_);
  LayerParameter layer_param;
  ColinearConvolutionParameter* convolution_param =
      layer_param.mutable_colinear_convolution_param();
  convolution_param->set_kernel_size(3);
  convolution_param->set_stride(2);
  convolution_param->set_num_output(4);
  convolution_param->mutable_quadratic_weight_filler()->set_type("constant");
  convolution_param->mutable_quadratic_weight_filler()->set_value(1);
  convolution_param->mutable_linear_weight_filler()->set_type("constant");
  convolution_param->mutable_linear_weight_filler()->set_value(1);
  convolution_param->mutable_bias_filler()->set_type("constant");
  convolution_param->mutable_bias_filler()->set_value(0.1);
  convolution_param->set_symmetric(true);
  shared_ptr<Layer<TypeParam> > layer(
      new ColinearConvolutionLayer<TypeParam>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  Caffe::set_mode(Caffe::GPU);
  layer->Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  // After the convolution, the output should all have output values 27.1
  const TypeParam* top_data = this->blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], 27*27+27+0.1, 1e-4);
  }
}


TYPED_TEST(ColinearConvolutionLayerTest, TestCPUGradientSymm) {
  LayerParameter layer_param;
  ColinearConvolutionParameter* convolution_param =
      layer_param.mutable_colinear_convolution_param();
  convolution_param->set_kernel_size(3);
  convolution_param->set_stride(2);
  convolution_param->set_num_output(2);
  convolution_param->mutable_quadratic_weight_filler()->set_type("gaussian");
  convolution_param->mutable_linear_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("gaussian");
  convolution_param->set_symmetric(true);
  Caffe::set_mode(Caffe::CPU);
  ColinearConvolutionLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_));
}

TYPED_TEST(ColinearConvolutionLayerTest, TestGPUGradientSymm) {
  LayerParameter layer_param;
  ColinearConvolutionParameter* convolution_param =
      layer_param.mutable_colinear_convolution_param();
  convolution_param->set_kernel_size(3);
  convolution_param->set_stride(2);
  convolution_param->set_num_output(2);
  convolution_param->mutable_quadratic_weight_filler()->set_type("gaussian");
  convolution_param->mutable_linear_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("gaussian");
  convolution_param->set_symmetric(true);
  Caffe::set_mode(Caffe::GPU);
  ColinearConvolutionLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_));
}

TYPED_TEST(ColinearConvolutionLayerTest, TestCPUGradientOnlyQSymm) {
  LayerParameter layer_param;
  ColinearConvolutionParameter* convolution_param =
      layer_param.mutable_colinear_convolution_param();
  convolution_param->set_kernel_size(3);
  convolution_param->set_stride(2);
  convolution_param->set_num_output(2);
  convolution_param->set_linear_term(false);
  convolution_param->mutable_quadratic_weight_filler()->set_type("gaussian");
  convolution_param->mutable_linear_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("gaussian");
  convolution_param->set_symmetric(true);
  Caffe::set_mode(Caffe::CPU);
  ColinearConvolutionLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_));
}

TYPED_TEST(ColinearConvolutionLayerTest, TestGPUGradientOnlyQSymm) {
  LayerParameter layer_param;
  ColinearConvolutionParameter* convolution_param =
      layer_param.mutable_colinear_convolution_param();
  convolution_param->set_kernel_size(3);
  convolution_param->set_stride(2);
  convolution_param->set_num_output(2);
  convolution_param->set_linear_term(false);
  convolution_param->mutable_quadratic_weight_filler()->set_type("gaussian");
  convolution_param->mutable_linear_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("gaussian");
  convolution_param->set_symmetric(true);
  Caffe::set_mode(Caffe::GPU);
  ColinearConvolutionLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_));
}

TYPED_TEST(ColinearConvolutionLayerTest, TestCPUGradientOnlyLSymm) {
  LayerParameter layer_param;
  ColinearConvolutionParameter* convolution_param =
      layer_param.mutable_colinear_convolution_param();
  convolution_param->set_kernel_size(3);
  convolution_param->set_stride(2);
  convolution_param->set_num_output(2);
  convolution_param->set_quadratic_term(false);
  convolution_param->mutable_quadratic_weight_filler()->set_type("gaussian");
  convolution_param->mutable_linear_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("gaussian");
  convolution_param->set_symmetric(true);
  Caffe::set_mode(Caffe::CPU);
  ColinearConvolutionLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_));
}

TYPED_TEST(ColinearConvolutionLayerTest, TestGPUGradientOnlyLSymm) {
  LayerParameter layer_param;
  ColinearConvolutionParameter* convolution_param =
      layer_param.mutable_colinear_convolution_param();
  convolution_param->set_kernel_size(3);
  convolution_param->set_stride(2);
  convolution_param->set_num_output(2);
  convolution_param->set_quadratic_term(false);
  convolution_param->mutable_quadratic_weight_filler()->set_type("gaussian");
  convolution_param->mutable_linear_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("gaussian");
  convolution_param->set_symmetric(true);
  Caffe::set_mode(Caffe::GPU);
  ColinearConvolutionLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_));
}


//************ Asymmetric ************
TYPED_TEST(ColinearConvolutionLayerTest, TestCPUSimpleConvolution) {
  // We will simply see if the convolution layer carries out averaging well.
  FillerParameter filler_param;
  filler_param.set_value(1.);
  ConstantFiller<TypeParam> filler(filler_param);
  filler.Fill(this->blob_bottom_);
  LayerParameter layer_param;
  ColinearConvolutionParameter* convolution_param =
      layer_param.mutable_colinear_convolution_param();
  convolution_param->set_kernel_size(3);
  convolution_param->set_stride(2);
  convolution_param->set_num_output(4);
  convolution_param->mutable_quadratic_weight_filler()->set_type("constant");
  convolution_param->mutable_quadratic_weight_filler()->set_value(1);
  convolution_param->mutable_linear_weight_filler()->set_type("constant");
  convolution_param->mutable_linear_weight_filler()->set_value(1);
  convolution_param->mutable_bias_filler()->set_type("constant");
  convolution_param->mutable_bias_filler()->set_value(0.1);
  shared_ptr<Layer<TypeParam> > layer(
      new ColinearConvolutionLayer<TypeParam>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  Caffe::set_mode(Caffe::CPU);
  layer->Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  // After the convolution, the output should all have output values 27.1
  const TypeParam* top_data = this->blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], 27*27+27+0.1, 1e-4);
  }
}

TYPED_TEST(ColinearConvolutionLayerTest, TestGPUSimpleConvolution) {
  // We will simply see if the convolution layer carries out averaging well.
  FillerParameter filler_param;
  filler_param.set_value(1.);
  ConstantFiller<TypeParam> filler(filler_param);
  filler.Fill(this->blob_bottom_);
  LayerParameter layer_param;
  ColinearConvolutionParameter* convolution_param =
      layer_param.mutable_colinear_convolution_param();
  convolution_param->set_kernel_size(3);
  convolution_param->set_stride(2);
  convolution_param->set_num_output(4);
  convolution_param->mutable_quadratic_weight_filler()->set_type("constant");
  convolution_param->mutable_quadratic_weight_filler()->set_value(1);
  convolution_param->mutable_linear_weight_filler()->set_type("constant");
  convolution_param->mutable_linear_weight_filler()->set_value(1);
  convolution_param->mutable_bias_filler()->set_type("constant");
  convolution_param->mutable_bias_filler()->set_value(0.1);
  shared_ptr<Layer<TypeParam> > layer(
      new ColinearConvolutionLayer<TypeParam>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  Caffe::set_mode(Caffe::GPU);
  layer->Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  // After the convolution, the output should all have output values 27.1
  const TypeParam* top_data = this->blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], 27*27+27+0.1, 1e-4);
  }
}

TYPED_TEST(ColinearConvolutionLayerTest, TestCPUGradient) {
  LayerParameter layer_param;
  ColinearConvolutionParameter* convolution_param =
      layer_param.mutable_colinear_convolution_param();
  convolution_param->set_kernel_size(3);
  convolution_param->set_stride(2);
  convolution_param->set_num_output(2);
  convolution_param->mutable_quadratic_weight_filler()->set_type("gaussian");
  convolution_param->mutable_linear_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("gaussian");
  Caffe::set_mode(Caffe::CPU);
  ColinearConvolutionLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_));
}

TYPED_TEST(ColinearConvolutionLayerTest, TestGPUGradient) {
  LayerParameter layer_param;
  ColinearConvolutionParameter* convolution_param =
      layer_param.mutable_colinear_convolution_param();
  convolution_param->set_kernel_size(3);
  convolution_param->set_stride(2);
  convolution_param->set_num_output(2);
  convolution_param->mutable_quadratic_weight_filler()->set_type("gaussian");
  convolution_param->mutable_linear_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("gaussian");
  Caffe::set_mode(Caffe::GPU);
  ColinearConvolutionLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_));
}

TYPED_TEST(ColinearConvolutionLayerTest, TestCPUGradientOnlyQ) {
  LayerParameter layer_param;
  ColinearConvolutionParameter* convolution_param =
      layer_param.mutable_colinear_convolution_param();
  convolution_param->set_kernel_size(3);
  convolution_param->set_stride(2);
  convolution_param->set_num_output(2);
  convolution_param->set_linear_term(false);
  convolution_param->mutable_quadratic_weight_filler()->set_type("gaussian");
  convolution_param->mutable_linear_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("gaussian");
  Caffe::set_mode(Caffe::CPU);
  ColinearConvolutionLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_));
}

TYPED_TEST(ColinearConvolutionLayerTest, TestGPUGradientOnlyQ) {
  LayerParameter layer_param;
  ColinearConvolutionParameter* convolution_param =
      layer_param.mutable_colinear_convolution_param();
  convolution_param->set_kernel_size(3);
  convolution_param->set_stride(2);
  convolution_param->set_num_output(2);
  convolution_param->set_linear_term(false);
  convolution_param->mutable_quadratic_weight_filler()->set_type("gaussian");
  convolution_param->mutable_linear_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("gaussian");
  Caffe::set_mode(Caffe::GPU);
  ColinearConvolutionLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_));
}

TYPED_TEST(ColinearConvolutionLayerTest, TestCPUGradientOnlyL) {
  LayerParameter layer_param;
  ColinearConvolutionParameter* convolution_param =
      layer_param.mutable_colinear_convolution_param();
  convolution_param->set_kernel_size(3);
  convolution_param->set_stride(2);
  convolution_param->set_num_output(2);
  convolution_param->set_quadratic_term(false);
  convolution_param->mutable_quadratic_weight_filler()->set_type("gaussian");
  convolution_param->mutable_linear_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("gaussian");
  Caffe::set_mode(Caffe::CPU);
  ColinearConvolutionLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_));
}

TYPED_TEST(ColinearConvolutionLayerTest, TestGPUGradientOnlyL) {
  LayerParameter layer_param;
  ColinearConvolutionParameter* convolution_param =
      layer_param.mutable_colinear_convolution_param();
  convolution_param->set_kernel_size(3);
  convolution_param->set_stride(2);
  convolution_param->set_num_output(2);
  convolution_param->set_quadratic_term(false);
  convolution_param->mutable_quadratic_weight_filler()->set_type("gaussian");
  convolution_param->mutable_linear_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("gaussian");
  Caffe::set_mode(Caffe::GPU);
  ColinearConvolutionLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_));
}

}  // namespace caffe