// Copyright 2014 BVLC and contributors.

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

	template <typename Dtype>
	Dtype ColinearConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) {
			const Dtype* bottom_data = bottom[0]->gpu_data();
			Dtype* top_data = (*top)[0]->mutable_gpu_data();
			Dtype* col_data = col_buffer_.mutable_gpu_data();
			const Dtype* weight = this->blobs_[1]->gpu_data();
			const Dtype* q_weight = this->blobs_[2]->gpu_data();
			int out_height = col_buffer_.height();
			int out_width = col_buffer_.width();
			int weight_offset = M_ * K_;
			int col_offset = K_ * N_;
			int top_offset = M_ * N_;

			int img_offset = out_height*out_width;
			int cor_size_offset = cor_size_*cor_size_;

			for (int n = 0; n < num_; ++n) {
				// First, im2col
				im2col_gpu(bottom_data + bottom[0]->offset(n), channels_, height_,
					width_, kernel_size_, pad_, stride_, col_data);
				// Second, add quadratic term
				if (quadratic_term_) {
					CUDA_POST_KERNEL_CHECK;
					QuadraticActivation<<<CAFFE_GET_BLOCKS(n_threads), CAFFE_CUDA_NUM_THREADS>>>
						(num_output_*out_height*out_width, col_data, q_weight, n, num_output_, out_height, out_width, cor_size_, top_data);
					CUDA_POST_KERNEL_CHECK;

				}
				// Third, add linear term
				if (linear_term_) {
					for (int g = 0; g < group_; ++g) {
						caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, K_,
							(Dtype)1., weight + weight_offset * g, col_data + col_offset * g,
							(Dtype)0., top_data + (*top)[0]->offset(n) + top_offset * g);
					}
				}
				// Fourth, add bias
				if (bias_term_) {
					caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
						N_, 1, (Dtype)1., this->blobs_[0]->gpu_data(),
						reinterpret_cast<const Dtype*>(bias_multiplier_->gpu_data()),
						(Dtype)1., top_data + (*top)[0]->offset(n));
				}
			}
			return Dtype(0.);
	}

	template <typename Dtype>
	void ColinearConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
			const Dtype* top_diff = top[0]->gpu_diff();
			const Dtype* weight = this->blobs_[1]->gpu_data();
			Dtype* weight_diff = this->blobs_[1]->mutable_gpu_diff();
			const Dtype* q_weight = this->blobs_[2]->gpu_data();
			Dtype* q_weight_diff = this->blobs_[2]->mutable_gpu_diff();

			const Dtype* bottom_data = (*bottom)[0]->gpu_data();
			Dtype* bottom_diff = (*bottom)[0]->mutable_gpu_diff();
			Dtype* col_data = col_buffer_.mutable_gpu_data();
			Dtype* col_diff = col_buffer_.mutable_gpu_diff();
			int out_height = col_buffer_.height();
			int out_width = col_buffer_.width();
			// bias gradient if necessary
			Dtype* bias_diff = NULL;

			if (bias_term_) {
				bias_diff = this->blobs_[1]->mutable_gpu_diff();
				CUDA_CHECK(cudaMemset(bias_diff, 0,
					sizeof(Dtype) * this->blobs_[0]->count()));
				for (int n = 0; n < num_; ++n) {
					caffe_gpu_gemv<Dtype>(CblasNoTrans, num_output_, N_,
						1., top_diff + top[0]->offset(n),
						reinterpret_cast<const Dtype*>(bias_multiplier_->gpu_data()),
						1., bias_diff);
				}
			}

			int weight_offset = M_ * K_;
			int col_offset = K_ * N_;
			int top_offset = M_ * N_;

			int img_offset = out_height*out_width;
			int cor_size_offset = cor_size_*cor_size_;

			CUDA_CHECK(cudaMemset(weight_diff, 0,
				sizeof(Dtype) * this->blobs_[1]->count()));
			CUDA_CHECK(cudaMemset(q_weight_diff, 0, 
				sizeof(Dtype) * this->blobs_[2]->count()));
			for (int n = 0; n < num_; ++n) {
				// since we saved memory in the forward pass by not storing all col data,
				// we will need to recompute them.
				im2col_gpu(bottom_data + (*bottom)[0]->offset(n), channels_, height_,
					width_, kernel_size_, pad_, stride_, col_data);

				if (quadratic_term_) {

				}

				// gradient w.r.t. weight. Note that we will accumulate diffs.
				if (linear_term_) {
					for (int g = 0; g < group_; ++g) {
						caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, K_, N_,
							(Dtype)1., top_diff + top[0]->offset(n) + top_offset * g,
							col_data + col_offset * g, (Dtype)1.,
							weight_diff + weight_offset * g);
					}
				}


				// gradient w.r.t. bottom data, if necessary
				if (propagate_down) {
					if (quadratic_term_) {

					}

					if (linear_term_) {
						for (int g = 0; g < group_; ++g) {
							caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, N_, M_,
								(Dtype)1., weight + weight_offset * g,
								top_diff + top[0]->offset(n) + top_offset * g,
								(Dtype)0., col_diff + col_offset * g);
						}
					}

					// col2im back to the data
					col2im_gpu(col_diff, channels_, height_, width_, kernel_size_, pad_,
						stride_, bottom_diff + (*bottom)[0]->offset(n));
				}
			}
	}

	template <typename Dtype>
	__global__ void QuadraticActivation(
		const int nthreads, const Dtype * col_data, const Dtype * q_weight, const int n,
		const int num_output_, const int out_height, const int out_width, const int cor_size_, 
		Dtype * top_data) {
			CUDA_KERNEL_LOOP(index, nthreads) {
				int m = index / out_height / out_width;
				int h = (index / out_width) % out_height;
				int w = index % out_width;
				Dtype val = 0.0;
				for (int x = 0; x < cor_size_; ++x) {
					for (int y = 0; y < cor_size_; ++y) {

					}
				}
			}

	}

	template <typename Dtype>
	__global__ void QuadraticError() {

	}

	template <typename Dtype>
	__global__ void QuadraticDWeight() {

	}

	INSTANTIATE_CLASS(ColinearConvolutionLayer);

}  // namespace caffe
