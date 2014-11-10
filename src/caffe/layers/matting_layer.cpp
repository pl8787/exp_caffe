// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <cfloat>
#include <vector>
#include <unordered_map>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/io.hpp"

#include "opencvlib.h"

using std::max;
using namespace cv;

namespace caffe {

	template <typename Dtype>
	void MattingLossLayer<Dtype>::SetUp(
		const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
			CHECK_EQ(bottom.size(), 3) << "Loss Layer takes three blobs as input.";
			CHECK_EQ(top->size(), 0) << "Loss Layer takes no as output.";
			CHECK_EQ(bottom[0]->num(), bottom[1]->num())
				<< "The data and label should have the same number.";
			CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
			CHECK_EQ(bottom[0]->height(), bottom[1]->height());
			CHECK_EQ(bottom[0]->width(), bottom[1]->width());
			difference_.Reshape(bottom[0]->num(), bottom[0]->channels(),
				bottom[0]->height(), bottom[0]->width());
			mask_.Reshape(bottom[0]->num(), bottom[0]->channels(),
				bottom[0]->height(), bottom[0]->width());
			epsilon = this->layer_param_.matting_loss_param().epsilon();
			beta = this->layer_param_.matting_loss_param().beta();
			has_mask_ = false;
			has_L_ = false;
	}

	template <typename Dtype>
	Dtype MattingLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) {
			int count = bottom[0]->count();
			int num = bottom[0]->num();

			caffe_cpu_sign(count, bottom[1]->cpu_data(), bottom[1]->mutable_cpu_data());

			if (!has_mask_) {
				caffe_cpu_fabs(count, bottom[1]->cpu_data(), mask_.mutable_cpu_data());
				caffe_cpu_sign(count, mask_.cpu_data(), mask_.mutable_cpu_data());
				has_mask_ = true;
			}

			if (!has_L_) {
				GetLaplacianMatrix(bottom[2], bottom[1]);
				has_L_ = true;
			}

			// caffe_set(count, (Dtype)0, difference_.mutable_cpu_data());

			// data term
			caffe_sub(count, bottom[0]->cpu_data(), bottom[1]->cpu_data(),
				difference_.mutable_cpu_data());
			caffe_mul(count, mask_.cpu_data(), difference_.cpu_data(), difference_.mutable_cpu_data());

			// smooth term
			Dtype * diff = difference_.mutable_cpu_data();
			const Dtype * alpha = bottom[0]->cpu_data();
			for (int i = 0; i < count; ++i) {
				if (mask_.cpu_data()[i] == 0) {
					for (auto p = L_[i].begin(); p != L_[i].end(); ++p) {
						diff[i] += 2 * p->second * alpha[i] * beta;
					}
				}
			}

			Dtype loss = caffe_cpu_dot(
				count, difference_.cpu_data(), difference_.cpu_data()) / num / Dtype(2);
			return loss;
	}

	template <typename Dtype>
	void MattingLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
			int count = (*bottom)[0]->count();
			int num = (*bottom)[0]->num();
			// Compute the gradient
			caffe_cpu_axpby(count, Dtype(1) / num, difference_.cpu_data(), Dtype(0),
				(*bottom)[0]->mutable_cpu_diff());
	}

	template <typename Dtype>
	bool MattingLossLayer<Dtype>::CheckKnown(Mat& trimap, Mat& w_ind) {
		for(int i=0;i<9;i++)
		{
			int idx = w_ind.at<int>(0,i);
			int h = idx / trimap.cols;
			int w = idx % trimap.cols;

			if(fabs(trimap.at<Dtype>(h, w)-0) < 0.01)
			{
				return false;
			}
		}
		return true;
	}

	template <typename Dtype>
	void MattingLossLayer<Dtype>::Locate(int l1, int l2, Dtype l3) {
		if (!L_.count(l1)) {
			L_[l1] = std::unordered_map<int, Dtype>();
			L_[l1][l2] = 0.0;
		}
		L_[l1][l2] += l3;
	}

	template <typename Dtype>
	void MattingLossLayer<Dtype>::CheckL() {
		for (auto q = L_.begin(); q != L_.end(); ++q) {
			float sum = 0.0;
			int i = q->first;
			for (auto p = L_[i].begin(); p != L_[i].end(); ++p) {
				sum += p->second;
			}
			if (fabs(sum) > 1e-5) {
				cout<<"error: "<<sum<<endl;
			}
		}
	}

	template <typename Dtype>
	void MattingLossLayer<Dtype>::GetLaplacianMatrix(Blob<Dtype> *blob_img, Blob<Dtype> *blob_trimap) {

		Mat img(blob_img->height(), blob_img->width(), CV_32FC3);
		ChangeBlobToImage(blob_img, img);

		Mat trimap(blob_trimap->height(), blob_trimap->width(), CV_32FC1);
		ChangeBlobToImage(blob_trimap, trimap);

		int SizeH_W = img.rows * img.cols;
		int count = 0;
		Mat k = Mat::ones(img.rows, img.cols, CV_32SC1);
		Mat U = Mat::eye(3, 3, CV_32FC1);
		Mat D = Mat::eye(9, 9, CV_32FC1);

		for(int i=0;i<SizeH_W;i++) {
			((int *)k.data)[i]=i;
		}

		for(int x = 1; x < img.cols - 1; ++x)
		{
			cout<<"\r"<<x;
			for(int y = 1; y < img.rows - 1; ++y)
			{
				Mat wk(3, 3, CV_32FC3);
				img.rowRange(y-1,y+2).colRange(x-1,x+2).copyTo(wk);

				Scalar win_mus=mean(wk);
				wk=wk.reshape(1,9);

				Mat w_ind(3,3,CV_32SC1);
				k.rowRange(y-1,y+2).colRange(x-1,x+2).copyTo(w_ind);
				w_ind=w_ind.reshape(1,9);

				if(CheckKnown(trimap, w_ind))
				{
					continue;
				}

				Mat win_mu(3,1,CV_32FC1);
				win_mu.at<Dtype>(0)=win_mus[0];
				win_mu.at<Dtype>(1)=win_mus[1];
				win_mu.at<Dtype>(2)=win_mus[2];

				Mat win_cov=(wk.t()*wk/9.0)-win_mu*win_mu.t();

				Mat dif(wk);
				for(Dtype * p=(Dtype *)dif.datastart; p<(Dtype *)dif.dataend;)
				{
					*p-=win_mus[0];
					p++;
					*p-=win_mus[1];
					p++;
					*p-=win_mus[2];
					p++;
				}

				Mat elements=D-(1.0+dif*(win_cov+U*epsilon/9.0).inv(CV_SVD_SYM)*dif.t())/9.0;

				int *l1=(int *)w_ind.datastart;
				int *l2=(int *)w_ind.datastart;
				Dtype *l3=(Dtype *)elements.datastart;
				count=1;

				for(;l3<(Dtype *)elements.dataend;l3++,count++)
				{
					Locate(*l1,*l2,*l3);
					l2++;
					if(count%9==0)
					{
						count=0;
						l1++;
						l2=(int *)w_ind.datastart;
					}
				}
			}
		}

		cout<<endl<<"Complete calculate Sparse Matrix L."<<endl;
		cout<<"L"<<endl<<"\trows="<<SizeH_W<<endl<<"\tcols="<<SizeH_W<<endl<<"\tElements="<<SizeH_W*25<<endl;

		CheckL();
	}

	template <typename Dtype>
	void MattingLayer<Dtype>::SetUp(
		const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
			CHECK_EQ(bottom.size(), 2) << "Loss Layer takes two blobs as input.";
			CHECK_EQ(top->size(), 1) << "Loss Layer takes one as output.";
			CHECK_EQ(bottom[0]->num(), bottom[1]->num())
				<< "The data and label should have the same number.";
			CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
			CHECK_EQ(bottom[0]->height(), bottom[1]->height());
			CHECK_EQ(bottom[0]->width(), bottom[1]->width());

			(*top)[0]->Reshape(1, 2, 1, 1);

			difference_.Reshape(bottom[0]->num(), bottom[0]->channels(),
				bottom[0]->height(), bottom[0]->width());
			mask_.Reshape(bottom[0]->num(), bottom[0]->channels(),
				bottom[0]->height(), bottom[0]->width());
			has_mask_ = false;
	}

	template <typename Dtype>
	Dtype MattingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) {
			int count = bottom[0]->count();
			int num = bottom[0]->num();

			if (!has_mask_) {
				caffe_cpu_fabs(count, bottom[1]->cpu_data(), mask_.mutable_cpu_data());
				caffe_cpu_sign(count, mask_.cpu_data(), mask_.mutable_cpu_data());
				has_mask_ = true;
			}

			caffe_sub(count, bottom[0]->cpu_data(), bottom[1]->cpu_data(),
				difference_.mutable_cpu_data());
			caffe_mul(count, mask_.cpu_data(), difference_.cpu_data(), difference_.mutable_cpu_data());
			Dtype loss = caffe_cpu_dot(
				count, difference_.cpu_data(), difference_.cpu_data()) / num / Dtype(2);
			
			(*top)[0]->mutable_cpu_data()[0] = loss;

			return Dtype(0);
	}

	INSTANTIATE_CLASS(MattingLayer);
	INSTANTIATE_CLASS(MattingLossLayer);


}  // namespace caffe