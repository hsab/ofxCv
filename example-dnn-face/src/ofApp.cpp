#include "ofApp.h"

using namespace ofxCv;
using namespace cv;

void ofApp::setup() {
   ofSetVerticalSync(true);

   updateButton.addListener(this, &ofApp::updateButtonPressed);

   gui.setup();  // most of the time you don't need a name
   gui.add(threshold.setup("Detection Threshold", 0.18, 0.05, 1.0));
   gui.add(persistance.setup("Tracker Persistance", 100, 0, 300));
   gui.add(distance.setup("Tracker Maximum Distance", 160, 16, 300));
   gui.add(smoothingRate.setup("Tracker Smoothing Rate", 0.245, 0.05, 2.0));
   gui.add(smoothed.setup("Smooth Tracker", true));
   gui.add(color.setup("Rectangle Color", ofColor(255, 0, 0), ofColor(0, 0), ofColor(255, 255)));
   gui.add(updateButton.setup("Update Values"));

   dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel");
   dnn.getTracker().setPersistence(persistance);
   dnn.getTracker().setMaximumDistance(distance);
   dnn.getTracker().setSmoothingRate(smoothingRate);

   cam.setup(640, 480);
}

void ofApp::exit() {
   updateButton.removeListener(this, &ofApp::updateButtonPressed);
}

void ofApp::updateButtonPressed() {
   dnn.getTracker().setPersistence(persistance);
   dnn.getTracker().setMaximumDistance(distance);
   dnn.getTracker().setSmoothingRate(smoothingRate);
}

void ofApp::update() {
   cam.update();
   if (cam.isFrameNew()) {
      ofPixels pix = cam.getPixels();
      detections = dnn.detect(toCv(pix), threshold);
      // tracker = dnn.getTracker();
   }
}

void ofApp::draw() {
   ofSetColor(255);
   cam.draw(0, 0);

   for (int i = 0; i < detections.size(); i++) {
      ofxCv::Detection d = detections[i];

      // int label = dnn.getLabel(i);
      // float age = tracker.getAge(label);

      if (smoothed) {
         x = d.smoothX;
         y = d.smoothY;
         centerX = d.smoothCenterX;
         w = d.smoothWidth;
         h = d.smoothHeight;
      } else {
         x = d.x;
         y = d.y;
         centerX = d.centerX;
         w = d.width;
         h = d.height;
      }

      ofPushStyle();
      ofSetColor(color);
      ofNoFill();
      ofSetLineWidth(5);
      ofDrawRectangle(x, y, w, h);
      ofPopStyle();

      ofDrawBitmapString(ofToString(d.certainty) + "   " + ofToString(d.age), x, y + h);

      cam.getTexture().drawSubsection((i % 4) * 160, 480 + floor(i / 4) * 160, 160, 160, centerX - (h / 2), y, h, h);
   }
   gui.draw();
}
