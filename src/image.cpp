#include "image.hpp"

#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

Image loadImage(const std::string &path)
{
  Image img;
  int channels;
  img.data = stbi_load(path.c_str(), &img.width, &img.height, &channels, STBI_rgb_alpha);
  return img;
}

Image* loadImageData(const std::string &path)
{
  Image *img = new Image();
  int channels;
  img->data = stbi_load(path.c_str(), &img->width, &img->height, &channels, STBI_rgb_alpha);
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


GLuint createTexture(Image &img)
{
  GLuint texId = 0;
  if(img.data)
    {
      glGenTextures(1, &texId); glBindTexture(GL_TEXTURE_2D, texId);
      // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S,     GL_REPEAT);
      // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T,     GL_REPEAT);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, img.width, img.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img.data);
      glBindTexture(GL_TEXTURE_2D, 0);
    }
  return texId;
}
GLuint createTexture(Image *img) { return (img ? createTexture(*img) : 0); }

void destroyTexture(GLuint &texId)
{
  if(texId > 0)
    {
      glBindTexture(GL_TEXTURE_2D, 0);
      glDeleteTextures(1, &texId); texId = 0;
      texId = 0;
    }
}
