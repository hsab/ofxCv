#include "ofxCv/DNN.h"
#include "ofGraphics.h"

namespace ofxCv {

using namespace cv;
using namespace std;
DNN::DNN()
    : hasNetwork(false) {
}

DNN::~DNN() {
}

void DNN::readNetFromCaffe(const std::string& prototxt, const std::string& caffeModel, bool absolute) {
   if (absolute)
      net = cv::dnn::readNetFromCaffe(prototxt, caffeModel);
   else
      net = cv::dnn::readNetFromCaffe(ofToDataPath(prototxt), ofToDataPath(caffeModel));

   hasNetwork = true;
}

void DNN::forward(cv::Mat mat) {
   cv::cvtColor(mat, mat, COLOR_RGB2BGR);
   cv::resize(mat, mat, cv::Size(300, 300));
   blob = cv::dnn::blobFromImage(mat, 1.0, cv::Size(300, 300), cv::Scalar(104.0, 177.0, 123.0), false, false);

   net.setInput(blob);
   net_out = net.forward();
}

std::vector<Detection> DNN::detect(cv::Mat mat, float thresh) {
   cv::Size s = mat.size();
   boundingRects.clear();

   forward(mat);
   std::vector<Detection> detections;
   for (int i = 0; i < net_out.size[2]; i++) {
      cv::Vec<float, 7> a = net_out.at<cv::Vec<float, 7>>(0, 0, i);
      float cert = a[2];

      Detection d;
      d.certainty = cert;
      d.left = d.x = a[3] * mat.cols;
      d.top = d.y = a[4] * mat.rows;
      d.right = a[5] * mat.cols;
      d.bottom = a[6] * mat.rows;
      d.centerX = (d.left + d.right) / 2;
      d.centerY = (d.top + d.bottom) / 2;
      d.width = d.right - d.left;
      d.height = d.bottom - d.top;

      if (cert >= thresh && d.left + d.width <= s.width && d.top + d.height <= s.height) {
         boundingRects.push_back(cv::Rect(d.x, d.y, d.width, d.height));

         detections.push_back(d);
      }
   }

   tracker.track(boundingRects);

   for (int i = 0; i < detections.size(); i++) {
      detections[i].age = tracker.getAgeNano(getLabel(i));
      cv::Rect smooth = tracker.getSmoothed(getLabel(i));
      detections[i].smoothX = smooth.x;
      detections[i].smoothY = smooth.y;
      detections[i].smoothWidth = smooth.width;
      detections[i].smoothHeight = smooth.height;
      detections[i].smoothCenterX = smooth.x + (smooth.width / 2);
      detections[i].smoothCenterY = smooth.y + (smooth.height / 2);
   }

   std::sort(detections.begin(), detections.end(), greater<Detection>());

   return detections;
}

cv::Vec2f DNN::getVelocity(unsigned int i) const {
   return tracker.getVelocity(i);
}

unsigned int DNN::getLabel(unsigned int i) const {
   return tracker.getCurrentLabels()[i];
}

RectTracker& DNN::getTracker() {
   return tracker;
}

void DNN::resetNetwork() {
   hasNetwork = false;
}

}  // namespace ofxCv
