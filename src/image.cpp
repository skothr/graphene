#include "image.hpp"

#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

// app icon loading
IconImage* loadImageData(const std::string &path)
{
  IconImage *img = new IconImage();
  int channels;
  img->pixels = stbi_load(path.c_str(), &img->width, &img->height, &channels, STBI_rgb_alpha);
  return img;
}

void setPngCompression(int compression)
{ stbi_write_png_compression_level = compression; }

// output texture data to image file
bool writeTexture(const std::string &path, const void *texData, const Vec2i &texSize, bool alpha)
{
  if(texSize.x > 0 && texSize.y > 0)
    {
      int channels = (alpha ? 4 : 3);
      if(path.find(".png") != std::string::npos && !stbi_write_png(path.c_str(), texSize.x, texSize.y, channels, texData, channels*texSize.x))
        { std::cout << "====> ERROR: Could not write CudaFieldTex data to PNG file '" << path << "'!\n";  return false; }
      else if(path.find(".bmp") != std::string::npos && !stbi_write_bmp(path.c_str(), texSize.x, texSize.y, channels, texData))
        { std::cout << "====> ERROR: Could not write CudaFieldTex data to BMP file '" << path << "'!\n";  return false; }
      else if(path.find(".hdr") != std::string::npos && !stbi_write_hdr(path.c_str(), texSize.x, texSize.y, channels, (const float*)texData))
        { std::cout << "====> ERROR: Could not write CudaFieldTex data to HDR file '" << path << "'!\n";  return false; }
      else { return true; }
    }
  return false;
}

