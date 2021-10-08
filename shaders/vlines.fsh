#version 440

// inputs
smooth in vec4 color;
// output
out vec4 fragColor;

void main()
{
  fragColor = color;
  // fragColor = vec4(1,0,0,1); // TEST
}
