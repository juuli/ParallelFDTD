///////////////////////////////////////////////////////////////////////////////
// Simple image saving code for C++ (c) Daniel Beard
// "A major annoyance for me is saving images from c++. I can’t really find any 
// libraries that I like, they are either too complex for the task or they don’t 
// do what I want them to do. So I have created my own, extremely lightweight 
// image library.
//
// All you need to run this is two files, the cpp file and the header. I chose .
// tga because it is one of the easiest, straightforward formats to save to."
//
// http://danielbeard.wordpress.com/2011/06/06/image-saving-code-c/
//
///////////////////////////////////////////////////////////////////////////////

#ifndef __IMAGE_SAVER__
#define __IMAGE_SAVER__

//includes
#include <vector>
#include <string>
#include <fstream>
#include <iostream>

using namespace std;

class TGAImage {

public:

  //Constructor
  TGAImage();

  //Overridden Constructor
  TGAImage(short width, short height);

    //data structures
  struct Colour {
    unsigned char r,g,b,a;
  };

  //Set all pixels at once
  void setAllPixels(Colour *pixels);

  //set individual pixels
  void setPixel(Colour inputcolor, int xposition, int yposition);

  void WriteImage(string filename);

//General getters and setters

  void setWidth(short width);
  void setHeight(short height);

  short getWidth();
  short getHeight();
  
private:

  //store the pixels
  Colour *m_pixels;

  short m_height;
  short m_width;

  //convert 2D to 1D indexing
  int convert2dto1d(int x, int y); 

  

};


#endif
