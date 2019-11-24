#include <iostream>
#include<sstream>
#include <utility>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

/*
 * DETECT
 */
pair<Mat, float> scaleDown(Mat src, int max_height, float scale) {
	Mat dst;

	if (src.rows > max_height) {
		scale = src.rows / float(max_height);
		float new_w = (float(max_height) / src.rows) * src.cols;
		resize(src, dst, Size(new_w, max_height));
	}

	return make_pair(dst, scale);
}

Mat grayscaleCLAHE(Mat src) {
	Mat dst_1, dst_2;

	cvtColor(src, dst_1, CV_BGR2GRAY);

	Ptr<CLAHE> clahe = createCLAHE();
	clahe->setClipLimit(2);
	clahe->setTilesGridSize(Size(8, 8));
	clahe->apply(dst_1, dst_2);

	return dst_2;
}

Mat sobel(Mat src) {
	Mat dst_1, dst_2;

	Sobel(src, dst_1, CV_32F, 1, 0, -1);
	convertScaleAbs(dst_1, dst_2);

	return dst_2;
}

Mat blurThreshold(Mat src) {
	Mat dst_1, dst_2;

	GaussianBlur(src, dst_1, Size(9, 9), 20, 20);
	threshold(dst_1, dst_2, 225, 255, THRESH_BINARY);

	return dst_2;
}

Mat fill(Mat src, Size s) {
	Mat dst;

	auto rect = getStructuringElement(MORPH_RECT, s);
	morphologyEx(src, dst, MORPH_CLOSE, rect);

	return dst;
}

Mat erodeDilate(Mat src, int e_iter, int d_iter) {
	Mat dst_1, dst_2;

	erode(src, dst_1, Mat(), Point(-1, -1), e_iter);
	dilate(dst_1, dst_2, Mat(), Point(-1, -1), d_iter);

	return dst_2;
}

vector<vector<Point>> getContours(Mat src) {
	vector<vector<Point>> contours;

	findContours(src, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	sort(contours.begin(), contours.end(),
			[](vector<Point>& c1, vector<Point>& c2) {
				return contourArea(c1) > contourArea(c2);
			});

	return contours;
}

pair<pair<vector<Point2f>, vector<Point2f>>, int> getAreaZone(Mat image,
		vector<vector<Point>> contours) {
	vector<Point2f> area, zone;
	RotatedRect rr_b = minAreaRect(contours[0]);
	Point2f rr_p2f[4];
	rr_b.points(rr_p2f);
	float rr_b_angle = rr_b.angle;
	Size rr_b_size = rr_b.size;

	if (rr_b_angle < -45.) {
		rr_b_angle += 90.0;
		swap(rr_b_size.width, rr_b_size.height);
	}

	Point2f a_p0(rr_p2f[0].x, rr_p2f[0].y);
	Point2f a_p1(rr_p2f[1].x, rr_p2f[1].y);
	Point2f a_p2(rr_p2f[2].x, rr_p2f[2].y);
	Point2f a_p3(rr_p2f[3].x, rr_p2f[3].y);

	float ext_a_t, ext_b_t, ext_a_b, ext_b_b;
	float ext_x = image.cols - 1;
	int margin = rr_b_size.height / 6;
	if (rr_b_angle < 0) {
		ext_a_t = (a_p2.y - a_p1.y) / (a_p2.x - a_p1.x);
		ext_b_t = a_p1.y - ext_a_t * a_p1.x;
		ext_a_b = (a_p0.y - a_p3.y) / (a_p0.x - a_p3.x);
		ext_b_b = a_p3.y - ext_a_b * a_p3.x;
	} else {
		margin = -margin;
		ext_a_t = (a_p1.y - a_p0.y) / (a_p1.x - a_p0.x);
		ext_b_t = a_p0.y - ext_a_t * a_p0.x;
		ext_a_b = (a_p3.y - a_p2.y) / (a_p3.x - a_p2.x);
		ext_b_b = a_p2.y - ext_a_b * a_p2.x;
	}

	Point2f z_p0(0, ext_b_t - margin);
	Point2f z_p1(ext_x, (ext_a_t * ext_x + ext_b_t) - margin);
	Point2f z_p2(0, ext_b_b + margin);
	Point2f z_p3(ext_x, (ext_a_b * ext_x + ext_b_b) + margin);

	area = {a_p0, a_p1, a_p2, a_p3};
	zone = {z_p0, z_p1, z_p2, z_p3};

	pair<vector<Point2f>, vector<Point2f>> area_zone = make_pair(area, zone);

	return make_pair(area_zone, margin);
}

vector<vector<Point>> removeUninterestingContours(vector<Point2f> zone,
		vector<vector<Point>> contours, int margin) {
	for (unsigned int i = 1; i < contours.size(); ++i) {
		RotatedRect rr_box = minAreaRect(contours[i]);
		Point2f rr_points2f[4];
		rr_box.points(rr_points2f);

		if (pointPolygonTest(zone, rr_points2f[0], true) < -abs(margin)
				|| pointPolygonTest(zone, rr_points2f[1], true) < -abs(margin)
				|| pointPolygonTest(zone, rr_points2f[2], true) < -abs(margin)
				|| pointPolygonTest(zone, rr_points2f[3], true)
						< -abs(margin)) {
			contours.erase(contours.begin() + i);
			--i;
		}
	}
	return contours;
}

pair<Mat, vector<Vec4i>> getLines(Mat src, Mat img, RotatedRect rr,
		int max_h_crop, int i) {
	vector<Vec4i> rr_lines;

	Mat rr_rm, rr_crop;
	rr_rm = getRotationMatrix2D(rr.center, rr.angle, 1.0);
	warpAffine(src, rr_crop, rr_rm, img.size(), INTER_CUBIC);
	getRectSubPix(rr_crop, rr.size, rr.center, rr_crop);

	cvtColor(rr_crop, rr_crop, CV_BGR2GRAY);
	Canny(rr_crop, rr_crop, 50, 200, 3);

	if (rr_crop.rows < 55) {
		resize(rr_crop, rr_crop, Size(rr_crop.cols, max_h_crop));
	}

	HoughLinesP(rr_crop, rr_lines, 1, CV_PI / 180, 50, 50, 10);

	return make_pair(rr_crop, rr_lines);
}

int getLinesAngle(vector<Vec4i> rr_lines) {
	int rr_lines_angle = 0;

	for (size_t i = 0; i < rr_lines.size(); i++) {
		Vec4i l = rr_lines[i];
		int a = (int) abs(atan2(l[1] - l[3], l[0] - l[2]) * 180.0 / CV_PI);
		rr_lines_angle += a;
	}

	return rr_lines_angle /= rr_lines.size();
}

pair<vector<Point2f>, pair<float, int>> getNewArea(vector<Point2f> area,
		RotatedRect rr, Mat rr_crop, float rr_angle, float base_rr_angle,
		int max_h_crop) {
	Point2f rr_points2f[4];
	rr.points(rr_points2f);

	// south
	if (rr_points2f[0].y > area[0].y) {
		area[0] = rr_points2f[0];
	}
	// west
	if (rr_points2f[1].x < area[1].x) {
		area[1] = rr_points2f[1];
	}
	// north
	if (rr_points2f[2].y < area[2].y) {
		area[2] = rr_points2f[2];
	}
	// east
	if (rr_points2f[3].x > area[3].x) {
		area[3] = rr_points2f[3];
	}

	// Base angle of biggest element
	base_rr_angle = (base_rr_angle == 200) ? rr_angle : base_rr_angle;

	// Max height for each crop with vertical lines
	max_h_crop = (rr_crop.rows > max_h_crop) ? rr_crop.rows : max_h_crop;

	pair<float, int> angle_h = make_pair(base_rr_angle, max_h_crop);
	return make_pair(area, angle_h);
}

vector<Point2f> checkContours(Mat src, Mat image,
		vector<vector<Point>> contours, vector<Point2f> area) {
	int max_h_crop = 55;
	float base_rr_angle = 200.0;

	for (unsigned int i = 0; i < contours.size(); ++i) {
		RotatedRect rr = minAreaRect(contours[i]);
		float rr_angle = rr.angle;
		Size rr_size = rr.size;

		if (rr_angle < -45.) {
			rr_angle += 90.0;
			swap(rr_size.width, rr_size.height);
		}

		if (rr_angle == 0
				&& ((base_rr_angle <= -2.) || (2. <= base_rr_angle))) {
			RotatedRect rr_new(rr.center, rr.size, base_rr_angle);
			rr = rr_new;
			rr_angle = rr.angle;
			rr_size = rr.size;
		}

		if (base_rr_angle == 200.
				|| ((base_rr_angle - 2. <= rr_angle)
						&& (rr_angle <= base_rr_angle + 2.))) {
			Mat rr_crop;
			vector<Vec4i> rr_lines;

			// Detect the lines in the area using the Hough transform
			tie(rr_crop, rr_lines) = getLines(src, image, rr, max_h_crop, i);

			if (rr_lines.size() > 1) {
				// Get mean lines angle
				int rr_lines_angle = getLinesAngle(rr_lines);

				if (rr_lines_angle >= 88 && rr_lines_angle <= 92) {
					pair<float, int> angle_h;

					// Calculation of the new area taking into account the contour
					tie(area, angle_h) = getNewArea(area, rr, rr_crop, rr_angle,
							base_rr_angle, max_h_crop);
					tie(base_rr_angle, max_h_crop) = angle_h;
				}
			}
		}
	}

	return area;
}

Mat getBarcode(Mat original, vector<Point2f> area, float scale) {
	vector<Point2f> bc_points { area[0] * scale, area[1] * scale, area[2]
			* scale, area[3] * scale };
	RotatedRect bc_box = minAreaRect(bc_points);
	float box_angle = bc_box.angle;
	Size box_size = bc_box.size;

	if (box_angle < -45.) {
		box_angle += 90.0;
		swap(box_size.width, box_size.height);
	}

	Mat bc_rm, bc_crop;
	bc_rm = getRotationMatrix2D(bc_box.center, box_angle, 1.0);
	warpAffine(original, bc_crop, bc_rm, original.size(), INTER_CUBIC);
	getRectSubPix(bc_crop, box_size, bc_box.center, bc_crop);

	return bc_crop;
}

/*
 * DECODE
 */
Mat grayscale(Mat src) {
	Mat dst = Mat(src.rows, src.cols, CV_8U);

	for (int y = 0; y < src.rows; y++) {
		for (int x = 0; x < src.cols; x++) {
			dst.at<uchar>(y, x) = (src.at<Vec3b>(y, x)[0]
					+ src.at<Vec3b>(y, x)[1] + src.at<Vec3b>(y, x)[2]) / 3;
		}
	}

	return dst;
}

Mat projectionV(Mat src) {
	Mat image = Mat(10, src.cols, CV_8U);

	for (int x = 0; x < src.cols; x++) {
		int total = 0;
		for (int y = 0; y < src.rows; y++) {
			total += src.at<uchar>(y, x);
			//cout << src.at<uchar>(y, x) << endl;
		}
		int res = total / src.rows;
		for (int i = 0; i < 10; ++i)
			image.at<uchar>(i, x) = res;
	}

	return image;
}

Mat projectionLigne(Mat src) {
	Mat image = Mat(10, src.cols, CV_8U);

	for (int x = 0; x < src.cols; x++) {
		for (int i = 0; i < 10; ++i)
			image.at<uchar>(i, x) = src.at<uchar>(0, x);
	}

	return image;
}

Mat clear(Mat src) {
	int min = 0;
	int max = src.cols - 1;

	while ((int) src.at<uchar>(min) == 255) {
		++min;
	}

	while ((int) src.at<uchar>(max) == 255) {
		--max;
	}

	Mat dst = src(Rect(min, 0, max - min + 1, 10));
	return dst;
}

vector<int> readCode(Mat src) {

	vector<int> vec = vector<int>(95);

	double inter = src.cols / (double) 95;
	double deb = inter / (double) 2;

	for (int i = 0; i < 95; ++i) {
		int pos = (int) (deb + (i * inter) + 0.5);
		int val = (int) src.at<uchar>(1, pos);
		vec[i] = val;
		val == 0 ? src.at<uchar>(5, pos) = 255 : src.at<uchar>(5, pos) = 0;
	}

	return vec;
}

vector<int> readCodeV2(Mat src) {
	vector<int> vec = vector<int>(95);

	double inter = src.cols / (double) 95;
	double deb = inter / (double) 2;
	double inter4 = inter / (double) 4;

	for (int i = 0; i < 95; ++i) {
		int pos = (int) (deb + (i * inter) + 0.5);
		int val = (int) src.at<uchar>(1, pos);

		int pos2;

		i < 47 ? pos2 = pos - inter4 : pos2 = pos + inter4;

		int val2 = (int) src.at<uchar>(1, pos2);

		if (val2 != val) {
			val = val2;
			pos = pos2;
			i < 47 ? deb -= inter4 : deb += inter4;
		}

		vec[i] = val;
		val == 0 ? src.at<uchar>(5, pos) = 255 : src.at<uchar>(5, pos) = 0;
	}
	return vec;
}

vector<vector<int>> splitCode(vector<int> src) {
	vector<vector<int>> vect = vector<vector<int>>(15);

	vect[0].push_back(src[0]);
	vect[0].push_back(src[1]);
	vect[0].push_back(src[2]);

	vect[14].push_back(src[92]);
	vect[14].push_back(src[93]);
	vect[14].push_back(src[94]);

	int index = 2;
	for (int i = 1; i < 14; ++i) {
		vect[i].push_back(src[++index]);
		vect[i].push_back(src[++index]);
		vect[i].push_back(src[++index]);
		vect[i].push_back(src[++index]);
		vect[i].push_back(src[++index]);
		if (i != 7) {
			vect[i].push_back(src[++index]);
			vect[i].push_back(src[++index]);
		}
	}

	for (unsigned int i = 0; i < vect.size(); ++i) {
		for (unsigned int j = 0; j < vect[i].size(); ++j) {
		}
	}

	return vect;
}

string decodeBarre(vector<vector<int>> vec) {
	string code = "";
	string motif = "";

	for (unsigned int i = 0; i < vec.size(); ++i) {
		if (vec[i].size() == 3) {
			if (vec[i][0] == 0 && vec[i][1] == 255 && vec[i][2] == 0)
				code += '"';
		}

		if (vec[i].size() == 5) {
			if (vec[i][0] == 255 && vec[i][1] == 0 && vec[i][2] == 255
					&& vec[i][3] == 0 && vec[i][4] == 255)
				code += "-";
		}

		if (vec[i].size() == 7) {
			// 0
			if ((vec[i][0] == 255 && vec[i][1] == 255 && vec[i][2] == 255
					&& vec[i][3] == 0 && vec[i][4] == 0 && vec[i][5] == 255
					&& vec[i][6] == 0)) {
				code += "0";
				if (i < 7)
					motif += "A";
			}
			if ((vec[i][0] == 255 && vec[i][1] == 0 && vec[i][2] == 255
					&& vec[i][3] == 255 && vec[i][4] == 0 && vec[i][5] == 0
					&& vec[i][6] == 0)
					|| (vec[i][0] == 0 && vec[i][1] == 0 && vec[i][2] == 0
							&& vec[i][3] == 255 && vec[i][4] == 255
							&& vec[i][5] == 0 && vec[i][6] == 255)) {
				code += "0";
				if (i < 7)
					motif += "B";
			}

			// 1
			if ((vec[i][0] == 255 && vec[i][1] == 255 && vec[i][2] == 0
					&& vec[i][3] == 0 && vec[i][4] == 255 && vec[i][5] == 255
					&& vec[i][6] == 0)) {
				code += "1";
				if (i < 7)
					motif += "A";
			}
			if ((vec[i][0] == 255 && vec[i][1] == 0 && vec[i][2] == 0
					&& vec[i][3] == 255 && vec[i][4] == 255 && vec[i][5] == 0
					&& vec[i][6] == 0)
					|| (vec[i][0] == 0 && vec[i][1] == 0 && vec[i][2] == 255
							&& vec[i][3] == 255 && vec[i][4] == 0
							&& vec[i][5] == 0 && vec[i][6] == 255)) {
				code += "1";
				if (i < 7)
					motif += "B";
			}

			// 2
			if ((vec[i][0] == 255 && vec[i][1] == 255 && vec[i][2] == 0
					&& vec[i][3] == 255 && vec[i][4] == 255 && vec[i][5] == 0
					&& vec[i][6] == 0)) {
				code += "2";
				if (i < 7)
					motif += "A";
			}
			if ((vec[i][0] == 255 && vec[i][1] == 255 && vec[i][2] == 0
					&& vec[i][3] == 0 && vec[i][4] == 255 && vec[i][5] == 0
					&& vec[i][6] == 0)
					|| (vec[i][0] == 0 && vec[i][1] == 0 && vec[i][2] == 255
							&& vec[i][3] == 0 && vec[i][4] == 0
							&& vec[i][5] == 255 && vec[i][6] == 255)) {
				code += "2";
				if (i < 7)
					motif += "B";
			}

			// 3
			if ((vec[i][0] == 255 && vec[i][1] == 0 && vec[i][2] == 0
					&& vec[i][3] == 0 && vec[i][4] == 0 && vec[i][5] == 255
					&& vec[i][6] == 0)) {
				code += "3";
				if (i < 7)
					motif += "A";
			}
			if ((vec[i][0] == 255 && vec[i][1] == 0 && vec[i][2] == 255
					&& vec[i][3] == 255 && vec[i][4] == 255 && vec[i][5] == 255
					&& vec[i][6] == 0)
					|| (vec[i][0] == 0 && vec[i][1] == 255 && vec[i][2] == 255
							&& vec[i][3] == 255 && vec[i][4] == 255
							&& vec[i][5] == 0 && vec[i][6] == 255)) {
				code += "3";
				if (i < 7)
					motif += "B";
			}

			// 4
			if ((vec[i][0] == 255 && vec[i][1] == 0 && vec[i][2] == 255
					&& vec[i][3] == 255 && vec[i][4] == 255 && vec[i][5] == 0
					&& vec[i][6] == 0)) {
				code += "4";
				if (i < 7)
					motif += "A";
			}
			if ((vec[i][0] == 255 && vec[i][1] == 255 && vec[i][2] == 0
					&& vec[i][3] == 0 && vec[i][4] == 0 && vec[i][5] == 255
					&& vec[i][6] == 0)
					|| (vec[i][0] == 0 && vec[i][1] == 255 && vec[i][2] == 0
							&& vec[i][3] == 0 && vec[i][4] == 0
							&& vec[i][5] == 255 && vec[i][6] == 255)) {
				code += "4";
				if (i < 7)
					motif += "B";
			}

			// 5
			if ((vec[i][0] == 255 && vec[i][1] == 0 && vec[i][2] == 0
					&& vec[i][3] == 255 && vec[i][4] == 255 && vec[i][5] == 255
					&& vec[i][6] == 0)) {
				code += "5";
				if (i < 7)
					motif += "A";
			}
			if ((vec[i][0] == 255 && vec[i][1] == 0 && vec[i][2] == 0
					&& vec[i][3] == 0 && vec[i][4] == 255 && vec[i][5] == 255
					&& vec[i][6] == 0)
					|| (vec[i][0] == 0 && vec[i][1] == 255 && vec[i][2] == 255
							&& vec[i][3] == 0 && vec[i][4] == 0
							&& vec[i][5] == 0 && vec[i][6] == 255)) {
				code += "5";
				if (i < 7)
					motif += "B";
			}

			// 6
			if ((vec[i][0] == 255 && vec[i][1] == 0 && vec[i][2] == 255
					&& vec[i][3] == 0 && vec[i][4] == 0 && vec[i][5] == 0
					&& vec[i][6] == 0)) {
				code += "6";
				if (i < 7)
					motif += "A";
			}
			if ((vec[i][0] == 255 && vec[i][1] == 255 && vec[i][2] == 255
					&& vec[i][3] == 255 && vec[i][4] == 0 && vec[i][5] == 255
					&& vec[i][6] == 0)
					|| (vec[i][0] == 0 && vec[i][1] == 255 && vec[i][2] == 0
							&& vec[i][3] == 255 && vec[i][4] == 255
							&& vec[i][5] == 255 && vec[i][6] == 255)) {
				code += "6";
				if (i < 7)
					motif += "B";
			}

			// 7
			if ((vec[i][0] == 255 && vec[i][1] == 0 && vec[i][2] == 0
					&& vec[i][3] == 0 && vec[i][4] == 255 && vec[i][5] == 0
					&& vec[i][6] == 0)) {
				code += "7";
				if (i < 7)
					motif += "A";
			}
			if ((vec[i][0] == 255 && vec[i][1] == 255 && vec[i][2] == 0
					&& vec[i][3] == 255 && vec[i][4] == 255 && vec[i][5] == 255
					&& vec[i][6] == 0)
					|| (vec[i][0] == 0 && vec[i][1] == 255 && vec[i][2] == 255
							&& vec[i][3] == 255 && vec[i][4] == 0
							&& vec[i][5] == 255 && vec[i][6] == 255)) {
				code += "7";
				if (i < 7)
					motif += "B";
			}

			// 8
			if ((vec[i][0] == 255 && vec[i][1] == 0 && vec[i][2] == 0
					&& vec[i][3] == 255 && vec[i][4] == 0 && vec[i][5] == 0
					&& vec[i][6] == 0)) {
				code += "8";
				if (i < 7)
					motif += "A";
			}
			if ((vec[i][0] == 255 && vec[i][1] == 255 && vec[i][2] == 255
					&& vec[i][3] == 0 && vec[i][4] == 255 && vec[i][5] == 255
					&& vec[i][6] == 0)
					|| (vec[i][0] == 0 && vec[i][1] == 255 && vec[i][2] == 255
							&& vec[i][3] == 0 && vec[i][4] == 255
							&& vec[i][5] == 255 && vec[i][6] == 255)) {
				code += "8";
				if (i < 7)
					motif += "B";
			}

			// 9
			if ((vec[i][0] == 255 && vec[i][1] == 255 && vec[i][2] == 255
					&& vec[i][3] == 0 && vec[i][4] == 255 && vec[i][5] == 0
					&& vec[i][6] == 0)) {
				code += "9";
				if (i < 7)
					motif += "A";
			}
			if ((vec[i][0] == 255 && vec[i][1] == 255 && vec[i][2] == 0
					&& vec[i][3] == 255 && vec[i][4] == 0 && vec[i][5] == 0
					&& vec[i][6] == 0)
					|| (vec[i][0] == 0 && vec[i][1] == 0 && vec[i][2] == 0
							&& vec[i][3] == 255 && vec[i][4] == 0
							&& vec[i][5] == 255 && vec[i][6] == 255)) {
				code += "9";
				if (i < 7)
					motif += "B";
			}
		}
	}

	// First digit
	if (motif == "AAAAAA") {
		code = "0-" + code;
	}
	if (motif == "AABABB") {
		code = "1-" + code;
	}
	if (motif == "AABBAB") {
		code = "2-" + code;
	}
	if (motif == "AABBBA") {
		code = "3-" + code;
	}
	if (motif == "ABAABB") {
		code = "4-" + code;
	}
	if (motif == "ABBAAB") {
		code = "5-" + code;
	}
	if (motif == "ABBBAA") {
		code = "6-" + code;
	}
	if (motif == "ABABAB") {
		code = "7-" + code;
	}
	if (motif == "ABABBA") {
		code = "8-" + code;
	}
	if (motif == "ABBABA") {
		code = "9-" + code;
	}

	return code;
}

string test(Mat source, int version) {
	Mat src = source.clone();
	resize(src, src, Size(), 10.0, 10.0);

	src = grayscale(src);

	threshold(src, src, 128, 255, CV_THRESH_OTSU);

	Mat image = projectionV(src);

	threshold(image, image, 127, 255, CV_THRESH_BINARY);

	Mat dst = clear(image);

	vector<int> vect;

	if (version == 1 || version == 3)
		vect = readCode(dst);
	if (version == 2 || version == 4)
		vect = readCodeV2(dst);

	vector<vector<int>> doubleVect = splitCode(vect);

	string code = decodeBarre(doubleVect);

	cout << version << " -- " << code << endl;

	if (code.length() != 17 && version == 1) {
		code = test(source, 2);
	}
	if (code.length() != 17 && version == 3) {
		code = test(source, 4);
	}

	return code;
}

int main(int argc, char **argv) {
	Mat image = imread("barcode.jpg");
	Mat original = image.clone();

	//	Resize big image for better detection
	int max_height = 400;
	float scale = 1.0;
	if (image.rows > max_height)
		tie(image, scale) = scaleDown(image, max_height, scale);

	Mat src = image.clone();

	//	Grayscale & CLAHE
	image = grayscaleCLAHE(src);

	//	Apply sobel filter on the image
	image = sobel(image);

	//	Apply Gaussian filter & threshold the image
	image = blurThreshold(image);

	//	Fill
	Size s(24, 3);  // (21, 7) || (24, 3)
	image = fill(image, s);

	//	Erode & dilate the image
	image = erodeDilate(image, 4, 8);

	//	Contours (sorted by area in descending order)
	vector<vector<Point>> contours = getContours(image);

	//	Get the biggest area & the zone to focus around this area
	int margin;
	vector<Point2f> area, zone;
	pair<vector<Point2f>, vector<Point2f>> area_zone;

	tie(area_zone, margin) = getAreaZone(image, contours);
	tie(area, zone) = area_zone;

	//	Remove the contours that are not in the zone of interest
	contours = removeUninterestingContours(zone, contours, margin);

	//	Check for each contour if it contains vertical lines to expand the area
	area = checkContours(src, image, contours, area);

	//	Draw the detected barcode's box
//	line(src, area[0], area[1], Scalar(0, 0, 255), 2);
//	line(src, area[1], area[2], Scalar(0, 0, 255), 2);
//	line(src, area[2], area[3], Scalar(0, 0, 255), 2);
//	line(src, area[3], area[0], Scalar(0, 0, 255), 2);
//
//	namedWindow("Box", CV_WINDOW_AUTOSIZE);
//	imshow("Box", src);

	//	Crop the barcode on the original image from the area
	Mat barcode = getBarcode(original, area, scale);

	imwrite("images/bc.jpg", barcode);

//	//	Get the 3 lines (1px height) from the barcode
	Rect l1(0, barcode.rows * 0.25, barcode.cols, 1);
	Rect l2(0, barcode.rows * 0.5, barcode.cols, 1);
	Rect l3(0, barcode.rows * 0.75, barcode.cols, 1);
	Mat l1_crop = barcode(l1);
	Mat l2_crop = barcode(l2);
	Mat l3_crop = barcode(l3);

	string res = test(barcode, 1);

	if (res.length() != 17) {
		res = test(l2_crop, 3);
	}
	if (res.length() != 17) {
		res = test(l1_crop, 3);
	}
	if (res.length() != 17) {
		res = test(l3_crop, 3);
	}

	if (res.length() != 17) {
		cout << "Je ne trouve pas de code-barres ! :(" << endl;
	}

	waitKey(0);
	return 0;
}

