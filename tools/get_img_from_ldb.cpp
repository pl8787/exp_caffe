#include <iostream>
#include <cstring>

#include "caffe/caffe.hpp"

#include <stdint.h>
#include <leveldb/db.h>
#include "opencvlib.h"


using namespace caffe;  // NOLINT(build/namespaces)

bool ReadImageToDatum(const string& filename, const int label,
	const int height, const int width, Datum* datum) {
		cv::Mat cv_img;
		if (height > 0 && width > 0) {
			cv::Mat cv_img_origin = cv::imread(filename, CV_LOAD_IMAGE_COLOR);
			cv::resize(cv_img_origin, cv_img, cv::Size(height, width));
		} else {
			cv_img = cv::imread(filename, CV_LOAD_IMAGE_COLOR);
		}
		if (!cv_img.data) {
			LOG(ERROR) << "Could not open or find file " << filename;
			return false;
		}
		datum->set_channels(3);
		datum->set_height(cv_img.rows);
		datum->set_width(cv_img.cols);
		datum->set_label(label);
		datum->clear_data();
		datum->clear_float_data();
		string* datum_string = datum->mutable_data();
		for (int c = 0; c < 3; ++c) {
			for (int h = 0; h < cv_img.rows; ++h) {
				for (int w = 0; w < cv_img.cols; ++w) {
					datum_string->push_back(
						static_cast<char>(cv_img.at<cv::Vec3b>(h, w)[c]));
				}
			}
		}
		return true;
}

bool WriteDatumToImage(const string& filename, Datum* datum) {
		
	int channels = 3;
	int height = datum->height();
	int width = datum->width();
	cv::Mat cv_img(height, width, CV_8UC3);

	memcpy(cv_img.data, datum->data().c_str(), 3*channels*height*width*sizeof(char));

	cv::imwrite(filename, cv_img);
	return true;
}

int main(int argc, char** argv) {
	char filename[1000];

	// Initialize the leveldb
	leveldb::DB* db_;
	leveldb::Iterator* iter_;
	leveldb::Options options;
	options.create_if_missing = false;
	//options.max_open_files = 100;
	string source(argv[1]);

	LOG(INFO) << "Opening leveldb " << source;
	leveldb::Status status = leveldb::DB::Open(
		options, source, &db_);
	CHECK(status.ok()) << "Failed to open leveldb "
		<< source << std::endl
		<< status.ToString();

	

	iter_ = db_->NewIterator(leveldb::ReadOptions());
	printf(iter_->status().ToString().c_str());
	iter_->SeekToFirst();
	
	Datum datum;
	int idx = 0;
	CHECK(iter_);
	CHECK(iter_->Valid());

	while (iter_->Valid()) {
		
		datum.ParseFromString(iter_->value().ToString());
		const string& data = datum.data();
		CHECK(data.size()) << "Image cropping only support uint8 data";

		sprintf(filename, "image/img%05d.bmp", idx);
		printf("Saving img%d\n",idx);
		WriteDatumToImage(filename, &datum);
		iter_->Next();
		idx++;
	}
}