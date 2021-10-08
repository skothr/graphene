#version 440

// uniforms
uniform mat4 vp; // proj ^ view
// vertex attributes
layout(location = 0) in vec3 posAttr;
layout(location = 1) in vec4 colAttr;
// outputs
smooth out vec4 color;

void main()
{
  gl_Position = vp * posAttr;
  color = colAttr;
}

