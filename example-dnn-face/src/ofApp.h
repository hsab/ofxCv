#pragma once

#include "ofMain.h"
#include "ofxCv.h"
#include "ofxGui.h"

class ofApp : public ofBaseApp {
  public:
   void setup();
   void exit();
   void update();
   void draw();
   void updateButtonPressed();

   ofVideoGrabber cam;

   ofxCv::DNN dnn;
   vector<ofxCv::Detection> detections;

   float x, y, centerX, w, h;
   //    RectTracker& tracker;

   ofxPanel gui;

   ofxFloatSlider threshold;
   ofxIntSlider persistance;
   ofxFloatSlider distance;
   ofxFloatSlider smoothingRate;
   ofxToggle smoothed;
   ofxColorSlider color;
   ofxButton updateButton;
};
