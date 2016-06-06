# GrabCut
A implementation of ["GrabCut": interactive foreground extraction using iterated graph cuts](http://dl.acm.org/citation.cfm?id=1015720).

Under GNU General Public License.

***

## Dependency
- OpenCV
- Python3-cv2
- numpy

***

## Usage
1. Download all the files and extract to someplace;
2. Do in the command line : python3 grabcut.py   (image)   (number of iterations)   (number of components in each GMM);
3. Press N on the input image to start grbcutting;
4. If user interaction is necessary, press each of the buttons listed below, and use left mouse button to draw points on the ORIGINAL image, then press N to continue.


| Button   | Function                                              |
| -------  |:-----------------------------------------------------:|
| '1'      | Mark Background                                       |
| '2'      | Mark Probable Background                              |
| '3'      | Mark Probable Foreground                              |
| '0'      | Mark Foreground                                       |
| 's'      | Save image, which would be saved as 'img_name_gc.jpg' |
| 'r'      | Reset the process                                     |
