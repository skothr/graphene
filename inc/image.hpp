#ifndef IMAGE_HPP
#define IMAGE_HPP

#include <string>
#include "vector.hpp"

typedef unsigned int GLuint;

struct Image
{
  int width  = -1;
  int height = -1;
  unsigned char *data = nullptr;
};

// load an image from a file (host memory)
Image  loadImage    (const std::string &path);
Image* loadImageData(const std::string &path);
void setPngCompression(int compression);
bool writeTexture(const std::string &path, const void *texData, const Vec2i &texSize, bool alpha);

// load into a GPU texture
GLuint createTexture(Image &img);
GLuint createTexture(Image *img);
// clean up texture
void destroyTexture(GLuint &texId);


#endif // IMAGE_HPP
