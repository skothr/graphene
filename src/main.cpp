#include "version/version.hpp"

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <string>
#include <vector>
#include <chrono>
#include <stdio.h>
#include <csignal>

#include "version.hpp"
#include "simWindow.hpp"
#include "image.hpp"

#define ICON_PATH "res/icons/graphene-icon-v4-64.png"

#define ENABLE_IMGUI_VIEWPORTS 0
#define ENABLE_IMGUI_DOCKING   0
#define START_MAXIMIZED        0
#define WINDOW_W 1920
#define WINDOW_H 1080
#define PADDING Vec2f(10.0f, 10.0f)

#define FLUID_W 512
#define FLUID_H 512

#define TEX_W 512
#define TEX_H 512

#define GL_MAJOR 4
#define GL_MINOR 6
#define GLSL_VERSION "#version 450"

bool initialized = false;
GLFWwindow* window    = nullptr;
GLFWimage  *appIcon   = nullptr;
SimWindow  *simWindow = nullptr;

void cleanup()
{
  if(initialized)
    {
      std::cout << "Cleaning...\n";
      if(simWindow) { delete simWindow; simWindow = nullptr; }
    
      // imgui cleanup
      ImGui_ImplOpenGL3_Shutdown();
      ImGui_ImplGlfw_Shutdown();
      ImGui::DestroyContext();

      if(window)
        {
          glfwDestroyWindow(window);
          glfwTerminate();
        }
      if(appIcon) { delete appIcon; }
      initialized = false;
    }
}

// interrupt signal callback
#define MAX_SIGINT_TRIES 3
static int  g_sigintTries  = 0;
static bool g_sigintTrying = false;
void signal_callback(int signum)
{
  std::string sigName;
  switch(signum)
    {
    case SIGABRT: sigName = "SIGABRT"; break; // 
    case SIGFPE:  sigName = "SIGFPE";  break; // 
    case SIGILL:  sigName = "SIGILL";  break; // 
    case SIGINT:  sigName = "SIGINT";  break; // 
    case SIGSEGV: sigName = "SIGSEGV"; break; // 
    case SIGTERM: sigName = "SIGTERM"; break; //
    default:      sigName = "UNKNOWN";
    }

  std::cout << " ====> Recieved interrupt signal " << signum << " (" << sigName << ")\n";

  switch(signum)
    {
    case SIGABRT: break;
    case SIGFPE:  break;
    case SIGILL:  break;
    case SIGINT:
      if(simWindow)
        {
          simWindow->quit();
          g_sigintTrying = true;
          g_sigintTries++;
          std::cout << " ==> (attempt " << g_sigintTries << " / " << MAX_SIGINT_TRIES << ")  ";
          if(g_sigintTries >= MAX_SIGINT_TRIES) { std::cout << " ==> () --> FORCE QUIT\n"; exit(1); }
          else                                  { std::cout << "\n"; }
        }
      else
        { std::cout << "ERROR: simWindow is null!\n"; exit(signum); }
      break;
    case SIGSEGV: break;
    case SIGTERM: break;
    }
}

// GLFW error callback
void glfw_error_callback(int error, const char* description)
{ std::cerr << "GLFW ERROR (" << error << ") --> " << description << "\n"; }

int main(int argc, char* argv[])
{
  std::cout << "\n" << "graphene Version: v" << GRPH_VERSION_MAJOR << "." << GRPH_VERSION_MINOR << "\n\n";
  signal(SIGINT, signal_callback);
  
  // set up window
  glfwSetErrorCallback(glfw_error_callback);
  if(!glfwInit()) { return 1; }

  // decide GL+GLSL versions
#ifdef __APPLE__
  // GL 3.2 + GLSL 150
  const char* glsl_version = "#version 150";
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // Required on Mac
#else
  // GL 4.4 + GLSL 440
  const char* glsl_version = GLSL_VERSION;
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, GL_MAJOR);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, GL_MINOR);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
  //glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // 3.0+ only
#endif


#if START_MAXIMIZED // start window maximized
  glfwWindowHint(GLFW_MAXIMIZED, GL_TRUE);
#endif // START_MAXIMIZED
  
  glfwWindowHint(GLFW_SAMPLES, 4); // 4x antialiasing
  
  // create window with graphics context
  window = glfwCreateWindow(WINDOW_W, WINDOW_H, "Graphene Sim", NULL, NULL);
  if(window == NULL) { return 1; }

  // get screen size
  GLFWmonitor       *monitor = glfwGetPrimaryMonitor();
  const GLFWvidmode *mode    = glfwGetVideoMode(monitor);
  std::cout << "Screen Size: " << mode->width <<  "x" << mode->height << "\n\n";
  
#if !START_MAXIMIZED // center window on screen
  glfwSetWindowPos(window, (mode->width - WINDOW_W)/2, (mode->height - WINDOW_H)/2);
#endif // START_MAXIMIZED
  
  // set window icon
  appIcon = (GLFWimage*)loadImageData(ICON_PATH);
  if(appIcon->pixels) { glfwSetWindowIcon(window, 1, appIcon); }

  // initialize gl context  
  glfwMakeContextCurrent(window);
  glfwSwapInterval(0); // Enable vsync
  if(glewInit() != GLEW_OK) { std::cout << "Failed to initialize OpenGL loader!\n"; return 1; }

  // create astro window before imgui setup to preserve GLFW callbacks
  simWindow = new SimWindow(window);
  
  // set up imgui context
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGui::StyleColorsDark(); // dark style
  
  ImGui::PushStyleColor(ImGuiCol_NavHighlight, Vec4f(0,0,0,0)); // no keyboard nav highlighting
  ImGui::GetStyle().TouchExtraPadding = Vec2f(2,2);             // padding for interaction
  
  // imgui context config
  ImGuiIO& io = ImGui::GetIO();
  io.IniFilename = nullptr;                              // disable .ini file
#if ENABLE_IMGUI_VIEWPORTS
  io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;
  io.ConfigViewportsNoTaskBarIcon = true;
#endif // ENABLE_IMGUI_VIEWPORTS
#if ENABLE_IMGUI_DOCKING
  io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;      // enable docking
  io.ConfigDockingWithShift = true;                      // docking when shift is held
#endif // ENABLE_IMGUI_DOCKING
  
  // start imgui context
  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init(glsl_version);

  simWindow->init();
  initialized = true;
  
  // main loop
  Vec2i frameSize(WINDOW_W, WINDOW_H); // size of current frame
  while(!glfwWindowShouldClose(window))
    {
      // handle events
      glfwPollEvents();
      if(simWindow->closing()) { glfwSetWindowShouldClose(window, GLFW_TRUE); }
      // start imgui frame
      ImGui_ImplOpenGL3_NewFrame();
      ImGui_ImplGlfw_NewFrame();
      ImGui::NewFrame();
      {
        glfwGetFramebufferSize(window, &frameSize.x, &frameSize.y); // get frame size
        simWindow->update();        // step simulation
        simWindow->draw(frameSize); // draw UI
      }
      ImGui::EndFrame();
      
      //// RENDERING ////
      glUseProgram(0);
      ImGui::Render();
      // Update and Render additional Platform Windows (if viewports enabled)
      if(io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) { ImGui::UpdatePlatformWindows(); ImGui::RenderPlatformWindowsDefault(); }
      // render to screen
      glViewport(0, 0, frameSize.x, frameSize.y);
      glClearColor(0.15f, 0.15f, 0.15f, 1.0f);
      glClear(GL_COLOR_BUFFER_BIT);
      ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
      glfwSwapBuffers(window);

      // render simulation to file separately
      // simWindow->renderToFile();
    }

  cleanup();
  std::cout << "Done\n";  
  return 0;
}



