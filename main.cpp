
#include <iostream>

#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"

int main(int argc, const char * argv[]) {
    std::cout << "Program loaded.\n";
    
//    Load & display image
    IplImage* img = cvLoadImage("/Users/EvergreenFu/GitHub/GrabCut/hyh.jpg");
    cvNamedWindow("Example1",CV_WINDOW_AUTOSIZE);
    cvShowImage("Example1",img);
    std::cout << "Press any key to terminate.\n";
    cvWaitKey(0);
    cvReleaseImage(&img);
    cvDestroyWindow("Example1");
    
    
//    cvNamedWindow("Video", CV_WINDOW_AUTOSIZE);
//    CvCapture* capture = cvCreateFileCapture(argv[1]);

    return 0;
}
