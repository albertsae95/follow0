#include <stdio.h>
#include <iostream>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/core.hpp"
#include <opencv2/video/tracking.hpp>
#include "opencv2/opencv.hpp"
#include <math.h>

#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/videoio/videoio_c.h"
#include "opencv2/core/core_c.h"
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/core/types_c.h"

#include "utils.h"

#define USE_VIDEO 1

using namespace cv;
using namespace std;

#define GREEN CV_RGB(0,255,0)
#define RED CV_RGB(255,0,0)
#define BLUE CV_RGB(255,0,255)
#define PURPLE CV_RGB(255,0,255)

struct Lane {
    Lane(){}
    Lane(cv::Point a, cv::Point b, float angle, float kl, float bl): p0(a),p1(b),angle(angle),
        votes(0),visited(false),found(false),k(kl),b(bl) { }

    cv::Point p0, p1;
    float angle;
    int votes;
    bool visited, found;
    float k, b;
};
/*
void crop(IplImage* src,  IplImage* dst, CvRect rect) {
    cvSetImageROI(src, rect);
    cvCopy(src, dst);
    cvSetImageCOI(src, 0);
    //cvResetImageROI(src);
}
*/
enum{
    SCAN_STEP = 5,			  // in pixels
    LINE_REJECT_DEGREES = 10, // in degrees
    BW_TRESHOLD = 250,		  // edge response strength to recognize for 'WHITE'
    BORDERX = 10,			  // px, skip this much from left & right borders
    MAX_RESPONSE_DIST = 5,	  // px

    CANNY_MIN_TRESHOLD = 1,	  // edge detector minimum hysteresis threshold
    CANNY_MAX_TRESHOLD = 100, // edge detector maximum hysteresis threshold

    HOUGH_TRESHOLD = 50,		// line approval vote threshold
    HOUGH_MIN_LINE_LENGTH = 50,	// remove lines shorter than this treshold
    HOUGH_MAX_LINE_GAP = 100  // join lines to one with smaller than this gaps
};

void processLanes(CvSeq* lines, IplImage* edges, IplImage* temp_frame) {

    // classify lines to left/right side
    std::vector<Lane> left, right;

    for(int i = 0; i < lines->total; i++ )
    {
        CvPoint* line = (CvPoint*)cvGetSeqElem(lines,i);
        int dx = line[1].x - line[0].x;
        int dy = line[1].y - line[0].y;
        float angle = atan2f(dy, dx) * 180/CV_PI;

        if (fabs(angle) <= LINE_REJECT_DEGREES) { // reject near horizontal lines
            continue;
        }

        // assume that vanishing point is close to the image horizontal center
        // calculate line parameters: y = kx + b;
        dx = (dx == 0) ? 1 : dx; // prevent DIV/0!
        float k = dy/(float)dx;
        float b = line[0].y - k*line[0].x;

        // assign lane's side based by its midpoint position
        int midx = (line[0].x + line[1].x) / 2;
        if (midx < temp_frame->width/2) {
            left.push_back(Lane(line[0], line[1], angle, k, b));
        } else if (midx > temp_frame->width/2) {
            right.push_back(Lane(line[0], line[1], angle, k, b));
        }
    }

    // show Hough lines
    for	(int i=0; i<right.size(); i++) {
        cvLine(temp_frame, right[i].p0, right[i].p1, CV_RGB(0, 0, 255), 2);
    }

    for	(int i=0; i<left.size(); i++) {
        cvLine(temp_frame, left[i].p0, left[i].p1, CV_RGB(255, 0, 0), 2);
    }

    //processSide(left, edges, false);
   // processSide(right, edges, true);

/*
    // show computed lanes
    int x = temp_frame->width * 0.55f;
    int x2 = temp_frame->width;
    cvLine(temp_frame, cvPoint(x, laneR.k.get()*x + laneR.b.get()),
        cvPoint(x2, laneR.k.get() * x2 + laneR.b.get()), CV_RGB(255, 0, 255), 2);

    x = temp_frame->width * 0;
    x2 = temp_frame->width * 0.45f;
    cvLine(temp_frame, cvPoint(x, laneL.k.get()*x + laneL.b.get()),
        cvPoint(x2, laneL.k.get() * x2 + laneL.b.get()), CV_RGB(255, 0, 255), 2);
}
*/
 }

int main(void)
{
    string path = "~/albertsae95/follow0/road.avi";
    VideoCapture capture(path);


     if (!capture.isOpened() ) {
        fprintf(stderr, "Error: Can't open video\n");
        return -1;
    }


    Size video_size = Size ((int) CAP_PROP_FRAME_WIDTH,
                            (int) CAP_PROP_FRAME_HEIGHT);

    //long current_frame = 0;
    int key_pressed = 0;
    Mat *frame;

    Size frame_size = Size(video_size.width, video_size.height/2);
    IplImage *temp_frame = cvCreateImage(frame_size, IPL_DEPTH_8U, 3);
    IplImage *grey = cvCreateImage(frame_size, IPL_DEPTH_8U, 1);
    IplImage *edges = cvCreateImage(frame_size, IPL_DEPTH_8U, 1);
    IplImage *half_frame = cvCreateImage(cvSize(video_size.width/2, video_size.height/2), IPL_DEPTH_8U, 3);

    CvMemStorage* houghStorage = cvCreateMemStorage(0);

    //cvSetCaptureProperty(input_video, CV_CAP_PROP_POS_FRAMES, current_frame);
    while(key_pressed != 27) {

        capture.read(*frame);
        //if (*frame == NULL){
          //  fprintf(stderr, "Error: null frame received\n");
            //return -1;
        //}
        }

     cvPyrDown(frame, half_frame, CV_GAUSSIAN_5x5); // Reduce the image by 2
     //cvCvtColor(temp_frame, grey, CV_BGR2GRAY); // convert to grayscale

     // we're interested only in road below horizon - so crop top image portion off
     //crop(IplImage* frame, temp_frame, cvRect(0,frame_size.height,frame_size.width,frame_size.height));
     cvCvtColor(temp_frame, grey, CV_BGR2GRAY); // convert to grayscale

     // Perform a Gaussian blur ( Convolving with 5 X 5 Gaussian) & detect edges
     cvSmooth(grey, grey, CV_GAUSSIAN, 5, 5);
     cvCanny(grey, edges, CANNY_MIN_TRESHOLD, CANNY_MAX_TRESHOLD);

     // do Hough transform to find lanes
     double rho = 1;
     double theta = CV_PI/180;
     CvSeq* lines = cvHoughLines2(edges, houghStorage, CV_HOUGH_PROBABILISTIC,
         rho, theta, HOUGH_TRESHOLD, HOUGH_MIN_LINE_LENGTH, HOUGH_MAX_LINE_GAP);

     processLanes(lines, edges, temp_frame);

     // show middle line
     cvLine(temp_frame, cvPoint(frame_size.width/2,0),
     cvPoint(frame_size.width/2,frame_size.height), CV_RGB(255, 255, 0), 1);

     cvShowImage("Grey", grey);
     cvShowImage("Edges", edges);
     cvShowImage("Color", temp_frame);

     cvMoveWindow("Grey", 0, 0);
     cvMoveWindow("Edges", 0, frame_size.height+25);
     cvMoveWindow("Color", 0, 2*(frame_size.height+25));

     key_pressed = cvWaitKey(5);


     cvReleaseMemStorage(&houghStorage);


     cvReleaseImage(&grey);
     cvReleaseImage(&edges);
     cvReleaseImage(&temp_frame);
     cvReleaseImage(&half_frame);

     capture.release();
}
