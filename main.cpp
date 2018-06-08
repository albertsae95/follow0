#include <stdio.h>
#include <iostream>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/core.hpp"
#include <opencv2/video/tracking.hpp>
#include "opencv2/opencv.hpp"
#include <math.h>
#include "opencv2/core/mat.hpp"

#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/videoio/videoio_c.h"
#include "opencv2/core/core_c.h"
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/core/types_c.h"

#include "utils.h"

//#define USE_VIDEO 1

using namespace cv;
using namespace std;

#define GREEN CV_RGB(0,255,0)
#define RED CV_RGB(255,0,0)
#define BLUE CV_RGB(255,0,255)
#define PURPLE CV_RGB(255,0,255)

#define K_VARY_FACTOR 0.2f
#define B_VARY_FACTOR 20
#define MAX_LOST_FRAMES 30

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

struct Status {
    Status():reset(true),lost(0){}
    ExpMovingAverage k, b;
    bool reset;
    int lost;
};

void crop(Mat src,  Mat dst, CvRect rect) {
    //MatOp::roi(src, rect.y, rect.x, src);
    //cvSetImageROI(src, rect);
    dst = src(rect);
    //src.copyTo(dst);
    //cvSetImageCOI(src, 0);
    //cvResetImageROI(src);
}

Status laneR, laneL;

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

void FindResponses(IplImage *img, int startX, int endX, int y, std::vector<int>& list)
{
    // scans for single response: /^\_

    const int row = y * img->width * img->nChannels;
    unsigned char* ptr = (unsigned char*)img->imageData;

    int step = (endX < startX) ? -1: 1;
    int range = (endX > startX) ? endX-startX+1 : startX-endX+1;

    for(int x = startX; range>0; x += step, range--)
    {
        if(ptr[row + x] <= BW_TRESHOLD) continue; // skip black: loop until white pixels show up
        printf("imagedata=%u\n", *ptr);
        // first response found
        int idx = x + step;

        // skip same response(white) pixels
        while(range > 0 && ptr[row+idx] > BW_TRESHOLD){
            idx += step;
            range--;
        }

        // reached black again
        if(ptr[row+idx] <= BW_TRESHOLD) {
            printf("imagedata=%u\n", *ptr);
            list.push_back(x);
        }

        x = idx; // begin from new pos
    }
}

void processSide(std::vector<Lane> lanes, IplImage *edges, bool right) {

    Status* side = right ? &laneR : &laneL;

    // response search
    int w = edges->width;
    int h = edges->height;
    const int BEGINY = 0;
    const int ENDY = h-1;
    const int ENDX = right ? (w-BORDERX) : BORDERX;
    int midx = w/2;
    int midy = edges->height/2;
    unsigned char* ptr = (unsigned char*)edges->imageData;

    // show responses
    int* votes = new int[lanes.size()]; //considering all the lines that we got
    for(int i=0; i<lanes.size(); i++) votes[i++] = 0;

    for(int y=ENDY; y>=BEGINY; y-=SCAN_STEP) {  //from beginy to endy scanning
        std::vector<int> rsp;
        FindResponses(edges, midx, ENDX, y, rsp);

        if (rsp.size() > 0) {
            int response_x = rsp[0]; //use first response (closest to screen center)

            float dmin = 9999999;
            float xmin = 9999999;
            int match = -1;
            for (int j=0; j<lanes.size(); j++) {
                // compute response point distance to current line
                float d = dist2line(
                        cvPoint2D32f(lanes[j].p0.x, lanes[j].p0.y),
                        cvPoint2D32f(lanes[j].p1.x, lanes[j].p1.y),
                        cvPoint2D32f(response_x, y));

                // point on line at current y line
                int xline = (y - lanes[j].b) / lanes[j].k;
                int dist_mid = abs(midx - xline); // distance to midpoint

                // pick the best closest match to line & to screen center
                if (match == -1 || (d <= dmin && dist_mid < xmin)) {
                    dmin = d;
                    match = j;
                    xmin = dist_mid;
                    break;
                }
            }

            // vote for each line
            if (match != -1) {
                votes[match] += 1;
                printf("current_votes=%d\n", *votes);
            }
        }
    }

    int bestMatch = -1;
    int mini = 9999999;
    for (int i=0; i<lanes.size(); i++) {
        int xline = (midy - lanes[i].b) / lanes[i].k;
        int dist = abs(midx - xline); // distance to midpoint

        if (bestMatch == -1 || (votes[i] > votes[bestMatch] && dist < mini)) {
            bestMatch = i;
            mini = dist;
        }
    }

    if (bestMatch != -1) {
        Lane* best = &lanes[bestMatch]; //assign ID of the best line
        float k_diff = fabs(best->k - side->k.get());
        float b_diff = fabs(best->b - side->b.get());

        bool update_ok = (k_diff <= K_VARY_FACTOR && b_diff <= B_VARY_FACTOR) || side->reset;

        printf("side: %s, k vary: %.4f, b vary: %.4f, lost: %s\n",
            (right?"RIGHT":"LEFT"), k_diff, b_diff, (update_ok?"no":"yes"));

        if (update_ok) {
            // update is in valid bounds
            side->k.add(best->k);
            side->b.add(best->b);
            side->reset = false;
            side->lost = 0;
        } else {
            // can't update, lanes flicker periodically, start counter for partial reset!
            side->lost++;
            if (side->lost >= MAX_LOST_FRAMES && !side->reset) {
                side->reset = true;
            }
        }

    } else {
        printf("no lanes detected - lane tracking lost! counter increased\n");
        side->lost++;
        if (side->lost >= MAX_LOST_FRAMES && !side->reset) {
            // do full reset when lost for more than N frames
            side->reset = true;
            side->k.clear();
            side->b.clear();
        }
    }

    delete[] votes;
}

void processLanes(std::vector<Lane> lines, IplImage* edges, IplImage* temp_frame) {

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

    processSide(left, edges, false);
    processSide(right, edges, true);

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

int main(void)
{
    double rho = 1;
    double theta = CV_PI/180;
    string path = "/home/albert/albertsae95/follow0/road.avi";
    VideoCapture capture(path);
    //argv[1];

     if (!capture.isOpened() ) {
        fprintf(stderr, "Error: Can't open video\n");
        return -1;
    }


    Size video_size(capture.get(CAP_PROP_FRAME_WIDTH), capture.get(CAP_PROP_FRAME_HEIGHT));

    //long current_frame = 0;
    int key_pressed = 0;
    Mat frame;

    Size frame_size = Size(video_size.width, video_size.height/2);
    Mat temp_frame(frame_size, IPL_DEPTH_8U, 3);
    Mat grey(frame_size, IPL_DEPTH_8U, 1);
    Mat edges(frame_size, IPL_DEPTH_8U, 1);
    Mat half_frame(video_size.width/2, video_size.height/2, IPL_DEPTH_8U, 3);

    CvMemStorage* houghStorage = cvCreateMemStorage(0);

    //cvSetCaptureProperty(input_video, CV_CAP_PROP_POS_FRAMES, current_frame);
    while(key_pressed != 27) {

    capture.read(frame);
    imshow("display", frame);
    moveWindow("display", frame_size.width+75, 0);
    imshow("edges from cvCanny", edges);
    moveWindow("edges from cvCanny", frame_size.width+75, 75);
    key_pressed = waitKey(20);

    //double fx=0, double fy=0, int interpolation=INTER_LINEAR
    resize(frame, half_frame, half_frame.size()); // Reduce the image by 2

    //cv::pyrDown(frame, half_frame, half_frame.size(), CV_GAUSSIAN_5x5); // Reduce the image by 2

    // we're interested only in road below horizon - so crop top image portion off
    crop(frame, temp_frame, cvRect(0,frame_size.height,frame_size.width,frame_size.height/2));
    cvtColor(temp_frame, grey, CV_BGR2GRAY); // convert to grayscale

    // Perform a Gaussian blur ( Convolving with 5 X 5 Gaussian) & detect edges
    //CV_GAUSSIAN_5x5.size;
    int ksize=3;
    double sigma=0.3*((ksize-1)*0.5 - 1) + 0.8;
    int ktype=CV_32F;
    Mat getGaussianKernel(ksize, sigma, ktype);
    //Mat.getGaussianKernel(int ksize=3, double 0.3*((ksize-1)*0.5 - 1) + 0.8, CV_32F);
    //cvSmooth (CvArr* grey, CvArr* grey, int smoothtype=CV_GAUSSIAN, int size1=3, int size2=0, double sigma1=0, double sigma2=0 );
    Size kernel_size = getGaussianKernel.size();
    GaussianBlur(grey, grey, kernel_size, 0, 0, BORDER_CONSTANT);

    Canny(grey, edges, CANNY_MIN_TRESHOLD, CANNY_MAX_TRESHOLD);
    //std::vector<Lane> lines;

    // do Hough transform to find lanes
    CvSeq* lines = HoughLinesP(edges, houghStorage, rho, theta, HOUGH_TRESHOLD, HOUGH_MIN_LINE_LENGTH, HOUGH_MAX_LINE_GAP);

    processLanes(lines, edges, temp_frame);

    // show middle line
    cvLine(temp_frame, cvPoint(frame_size.width/2,0), cvPoint(frame_size.width/2,frame_size.height), CV_RGB(255, 255, 0), 1);

    cvShowImage("Grey", grey);
    cvShowImage("Edges", edges);
    cvShowImage("Color", temp_frame);

    moveWindow("Grey", 0, 0);
    moveWindow("Edges", 0, frame_size.height+25);
    moveWindow("Color", 0, 2*(frame_size.height+25));
}
     //key_pressed = waitKey(5);

     cvReleaseMemStorage(&houghStorage);

     cvReleaseImage(&grey);
     cvReleaseImage(&edges);
     cvReleaseImage(&temp_frame);
     half_frame.release();
     capture.release();
     frame.release();
     destroyAllWindows();
     return 0;
}
