#include "ofApp.h"

using namespace ofxCv;
using namespace cv;

void ofApp::setup() {
   ofSetVerticalSync(true);

   kalman.init(1 / 10000., 1 / 10.);  // inverse of (smoothness, rapidness)

   line.setMode(OF_PRIMITIVE_LINE_STRIP);
   predicted.setMode(OF_PRIMITIVE_LINE_STRIP);
   estimated.setMode(OF_PRIMITIVE_LINE_STRIP);

   speed = 0.f;
}

void ofApp::update() {
   ofVec3f curPoint(mouseX, mouseY, 0);
   line.addVertex(ofPoint(curPoint.x, curPoint.y, 0));

   kalman.update(toGlm(curPoint));  // feed measurement

   point = kalman.getPrediction();  // prediction before measurement
   predicted.addVertex(ofPoint(point.x, point.y, 0));
   estimated.addVertex(kalman.getEstimation());  // corrected estimation after measurement

   speed = kalman.getVelocity().length();
   int alpha = ofMap(speed, 0, 20, 50, 255, true);
   line.addColor(ofColor(255, 255, 255, alpha));
   predicted.addColor(ofColor(255, 0, 0, alpha));
   estimated.addColor(ofColor(0, 255, 0, alpha));
}

void ofApp::draw() {
   ofBackground(0);

   line.draw();

   predicted.draw();
   ofPushStyle();
   ofSetColor(ofColor::red, 128);
   ofFill();
   ofDrawCircle(point, speed * 2);
   ofPopStyle();

   estimated.draw();
}
