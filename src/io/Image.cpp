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

#include "Image.h"

//Default Constructor
TGAImage::TGAImage() {

}

//Overridden Constructor
TGAImage::TGAImage(short width, short height) {
  m_width = width;
  m_height = height;
  m_pixels = new Colour[m_width*m_height];
}

//Set all pixels at once
void TGAImage::setAllPixels(Colour *pixels) {
  m_pixels = pixels;
}

//Set indivdual pixels
void TGAImage::setPixel(Colour inputcolor, int x, int y) {
  m_pixels[convert2dto1d(x,y)] = inputcolor;
}

//Convert 2d array indexing to 1d indexing
int TGAImage::convert2dto1d(int x, int y) {
  return m_width * x + y;
}

void TGAImage::WriteImage(string filename) {

  //Error checking
  if (m_width <= 0 || m_height <= 0)
  {
    cout << "Image size is not set properly" << endl;
    return;
  }

  ofstream o(filename.c_str(), ios::out | ios::binary);

  //Write the header
  o.put(0);
  o.put(0);
  o.put(2);                         /* uncompressed RGB */
  o.put(0);     o.put(0);
  o.put(0);   o.put(0);
  o.put(0);
  o.put(0);   o.put(0);           /* X origin */
  o.put(0);   o.put(0);           /* y origin */
  o.put((m_width & 0x00FF));
  o.put((m_width & 0xFF00) / 256);
  o.put((m_height & 0x00FF));
  o.put((m_height & 0xFF00) / 256);
  o.put(32);                        /* 24 bit bitmap */
  o.put(0);
     
  //Write the pixel data
  for (int i=0;i<m_height*m_width;i++) {
    o.put(m_pixels[i].b);
    o.put(m_pixels[i].g);
    o.put(m_pixels[i].r);
    o.put(m_pixels[i].a);
  }   
  
  //close the file
  o.close();
  
}
