// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <cfloat>
#include <vector>
#include <unordered_map>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/io.hpp"

#include "opencvlib.h"

using std::max;
using namespace cv;

namespace caffe {

	template <typename Dtype>
	void MaskWeightLayer<Dtype>::SetUp(
		const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
			CHECK_EQ(bottom.size(), 2) << "Loss Layer takes two blobs as input.";
			CHECK_EQ(top->size(), 1) << "Loss Layer takes one blob as output.";

			CHECK_EQ(bottom[0]->height(), bottom[1]->height());
			CHECK_EQ(bottom[0]->width(), bottom[1]->width());

			this->width_ = bottom[0]->width();
			this->height_ = bottom[0]->height();
			this->channel_ = bottom[0]->channels();

			(*top)[0]->Reshape(1, 1, height_, width_);

			if (this->blobs_.size() > 0) {
				LOG(INFO) << "Skipping parameter initialization";
			} else {
				int blob_size = 1;
				this->blobs_.resize(blob_size);

				this->blobs_[0].reset(new Blob<Dtype>(
					1, channel_, height_, width_));
				// fill the weights
				shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
					this->layer_param_.mask_weight_param().weight_filler()));
				weight_filler->Fill(this->blobs_[0].get());
				
			}

			need_bottom_ = this->layer_param_.mask_weight_param().need_bottom();
			need_mask_ = this->layer_param_.mask_weight_param().need_mask();

			if (!need_bottom_)
				LOG(INFO) << "Input of bottom[0] will be ignored.";
			if (!need_mask_)
				LOG(INFO) << "Input of bottom[1] will be ignored.";

			mask_.Reshape(1, 1,	height_, width_);
			has_mask_ = false;
	}

	template <typename Dtype>
	Dtype MaskWeightLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) {
			int count = bottom[0]->count();
			int num = bottom[0]->num();

			if (!has_mask_) {
				caffe_cpu_zeros(bottom[1]->count(), bottom[1]->cpu_data(), mask_.mutable_cpu_data());
				has_mask_ = true;
			}

			Dtype *top_data = (*top)[0]->mutable_cpu_data();
			const Dtype *bottom_data = bottom[0]->cpu_data();
			const Dtype *mask_data = mask_.cpu_data();
			const Dtype *weight_data = this->blobs_[0]->cpu_data();

			memset(top_data, 0, sizeof(Dtype) * (*top)[0]->count());

			for (int c = 0; c <channel_; ++c) {
				for (int i = 0; i < height_*width_; ++i) {
					if (!need_mask_ || *mask_data != 0)
					{
						if (need_bottom_) {
							*top_data += (*bottom_data) * (*weight_data);
						} else {
							*top_data += *weight_data;
						}
					}
					bottom_data++;
					weight_data++;
					top_data++;
					mask_data++;
				}
				top_data = (*top)[0]->mutable_cpu_data();
				mask_data = mask_.cpu_data();
			}

			return Dtype(0.);
	}

	template <typename Dtype>
	void MaskWeightLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
			int count = (*bottom)[0]->count();
			int num = (*bottom)[0]->num();

			const Dtype* top_diff = top[0]->cpu_diff();
			const Dtype* weight_data = this->blobs_[0]->cpu_data();
			Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
			const Dtype* bottom_data = (*bottom)[0]->cpu_data();
			Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
			const Dtype *mask_data = mask_.cpu_data();

			memset(weight_diff, 0, sizeof(Dtype) * this->blobs_[0]->count());

			for (int c = 0; c <channel_; ++c) {
				for (int i = 0; i < height_*width_; ++i) {
					if (!need_mask_ || *mask_data != 0)
					{
						if (need_bottom_) {
							*weight_diff = (*bottom_data) * (*top_diff);
						} else {
							*weight_diff = *top_diff;
						}
					}
					bottom_data++;
					weight_diff++;
					top_diff++;
					mask_data++;
				}
				top_diff = top[0]->cpu_diff();
				mask_data = mask_.cpu_data();
			}

			if (propagate_down) {
				memset(bottom_diff, 0, sizeof(Dtype) * (*bottom)[0]->count());

				for (int c = 0; c <channel_; ++c) {
					for (int i = 0; i < height_*width_; ++i) {
						if (!need_mask_ || *mask_data != 0)
						{
							*bottom_diff = (*weight_data) * (*top_diff);
						}
						bottom_diff++;
						weight_data++;
						top_diff++;
						mask_data++;
					}
					top_diff = top[0]->cpu_diff();
					mask_data = mask_.cpu_data();
				}
			}
	}

	INSTANTIATE_CLASS(MaskWeightLayer);


}  // namespace caffe