#ifndef IMAGE_HPP
#define IMAGE_HPP

#include <string>
#include "vector.hpp"

struct IconImage
{
  int width;
  int height;
  unsigned char *pixels;
};

IconImage* loadImageData(const std::string &path);
void setPngCompression(int compression);
bool writeTexture(const std::string &path, const void *texData, const Vec2i &texSize, bool alpha);

#endif // IMAGE_HPP
