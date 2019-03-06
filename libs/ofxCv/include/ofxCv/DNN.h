#pragma once

#include "ofxCv.h"
#include "ofxCv/Kalman.h"
#include "ofxCv/Tracker.h"
#include "ofxCv/Utilities.h"

// Based on "Fast Many Face Detection with C++/OpenFrameworks on macOS using Neural Networks"
// by github@LingDong
// https://github.com/LingDong-/fast-many-face-detection-with-cpp-or-openframeworks-on-mac-using-neural-networks

namespace ofxCv {

class Detection {
  public:
   float left;
   float top;
   float right;
   float bottom;
   float certainty;
   float centerX;
   float centerY;
   float width;
   float height;
   float x;
   float y;
   float smoothX;
   float smoothY;
   float smoothWidth;
   float smoothHeight;
   float smoothCenterX;
   float smoothCenterY;
   float age;

   bool operator<(Detection const& comp) const {
      return (age < comp.age);
   }

   bool operator>(Detection const& comp) const {
      return (age > comp.age);
   }
};

class DNN {
  public:
   // should constructor be protected?
   DNN();
   virtual ~DNN();

   void readNetFromCaffe(const std::string& prototxt, const std::string& caffeModel = "", bool absolute = false);

   template <class T>
   void forward(const T& image) {
      forward(toCv(image));
   }

   void forward(cv::Mat mat);

   std::vector<Detection> detect(cv::Mat mat, float thresh = 0.5);

   cv::Vec2f getVelocity(unsigned int i) const;
   RectTracker& getTracker();
   unsigned int getLabel(unsigned int i) const;

   virtual void resetNetwork();

  private:
   cv::dnn::Net net;
   cv::Mat blob;
   cv::Mat net_out;

  protected:
   bool hasNetwork;
   RectTracker tracker;
   std::vector<cv::Rect> boundingRects;
};

}  // namespace ofxCv
