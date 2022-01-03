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

#define ICON_PATH "res/icons/graphene-icon-v5-64.png"

#define ENABLE_IMGUI_VIEWPORTS 0
#define ENABLE_IMGUI_DOCKING   0
#define START_MAXIMIZED        1
#define WINDOW_W 1920
#define WINDOW_H 1080

#define GL_MAJOR 4
#define GL_MINOR 6
#define GLSL_VERSION "#version 450"

static bool initialized = false;
static GLFWwindow* window    = nullptr;
static GLFWimage  *appIcon   = nullptr;
static SimWindow  *simWindow = nullptr;

static ImGuiContext *mainContext    = ImGui::CreateContext(); // main ImGui context  for live window/interaction
static ImGuiContext *offlineContext = ImGui::CreateContext(); // offline context for rendering to separate framebuffer and saving to image files

void cleanup()
{
  if(initialized)
    {
      std::cout << "Cleanup (main.cpp)...\n";
      if(simWindow) { delete simWindow; simWindow = nullptr; }
    
      // imgui cleanup
      ImGui_ImplOpenGL3_Shutdown();
      ImGui_ImplGlfw_Shutdown();
      ImGui::DestroyContext(mainContext);
      ImGui::DestroyContext(offlineContext);

      if(window)  { glfwDestroyWindow(window); glfwTerminate(); }
      if(appIcon) { delete appIcon; }
      
      std::cout << "(done)\n";
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
          std::cout << " ======> (attempt " << g_sigintTries << " / " << MAX_SIGINT_TRIES << ")  ";
          if(g_sigintTries >= MAX_SIGINT_TRIES) { std::cout << " ======> () --> FORCE QUIT\n"; exit(1); }
          else                                  { std::cout << "\n"; }
        }
      else
        { std::cout << "======> ERROR(main.cpp:signal_callback): simWindow is null!\n"; exit(signum); }
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
  std::cout << "\n" << "graphene v" << GRPH_VERSION_MAJOR << "." << GRPH_VERSION_MINOR << "\n";
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
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
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
  
#if !START_MAXIMIZED // center window on screen
  glfwSetWindowPos(window, (mode->width - WINDOW_W)/2, (mode->height - WINDOW_H)/2);
#endif // START_MAXIMIZED
  
  // set window icon
  appIcon = (GLFWimage*)loadImageData(ICON_PATH);
  if(appIcon->pixels) { glfwSetWindowIcon(window, 1, appIcon); }

  // initialize gl context  
  glfwMakeContextCurrent(window);
  glfwSwapInterval(0); // Enable vsync
  if(glewInit() != GLEW_OK) { std::cout << "====> ERROR(main.cpp): Failed to initialize OpenGL loader!\n"; return 1; }

  // create astro window before imgui setup to preserve GLFW callbacks
  simWindow = new SimWindow(window);
  
  //// set up imgui context
  IMGUI_CHECKVERSION();
  // set up offline context (offline context for rendering to separate framebuffer and saving to image files)
  offlineContext = ImGui::CreateContext();
  ImGui::SetCurrentContext(offlineContext);
  ImGui::StyleColorsDark();      // dark style
  // imgui context config
  ImGuiIO *io = &ImGui::GetIO();
  io->IniFilename = nullptr;     // disable .ini file
  // start offline context backend
  ImGui_ImplOpenGL3_Init(glsl_version);
  /////////////////////////////////////////////
  
  //////////////////////////////////////////////////////////////////////////////////////////
  //// set up main context (main ImGui context  for live window/interaction)
  //////////////////////////////////////////////////////////////////////////////////////////
  mainContext = ImGui::CreateContext();
  ImGui::SetCurrentContext(mainContext);
  
  ImGui::StyleColorsDark(); // dark style
  ImGui::PushStyleColor(ImGuiCol_NavHighlight, Vec4f(0,0,0,0)); // no keyboard nav highlighting
  ImGui::GetStyle().TouchExtraPadding = Vec2f(1.5f,1.5f);       // padding for interaction
  // imgui context config
  io = &ImGui::GetIO(); io->IniFilename = nullptr;        // disable .ini file
#if ENABLE_IMGUI_VIEWPORTS
  io->ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;    // enable viewports (?)
  io->ConfigViewportsNoTaskBarIcon = true;
#endif // ENABLE_IMGUI_VIEWPORTS
#if ENABLE_IMGUI_DOCKING
  io->ConfigFlags |= ImGuiConfigFlags_DockingEnable;      // enable docking
  io->ConfigDockingWithShift = true;                      // docking when shift is held
#endif // ENABLE_IMGUI_DOCKING
  
  /////////////////////////////////////////////
  // start main context backend
  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init(glsl_version);
  /////////////////////////////////////////////
  const GLubyte *glv = glGetString(GL_VERSION);
  std::cout << "Using OpenGL version:   " << glv << "\n";

  // initialize simulation
  std::cout << "\n";
  simWindow->create(); initialized = true;
  
  // main loop
  Vec2i frameSize(WINDOW_W, WINDOW_H); // size of current frame
  while(!glfwWindowShouldClose(window))
    {
      // handle window events
      glfwPollEvents();
      if(simWindow->closing()) { glfwSetWindowShouldClose(window, GLFW_TRUE); }
      
      // re-upload font texture to gpu if fonts have changed
      if(simWindow->preFrame())
        { std::cout << "====> UPDATING FONT ATLAS...\n"; ImGui_ImplOpenGL3_DestroyDeviceObjects(); }
      
      // imgui implementation frame (main context)
      ImGui_ImplOpenGL3_NewFrame();
      ImGui_ImplGlfw_NewFrame();
      // imgui frame (main context)
      ImGui::NewFrame();
      {
        glfwGetFramebufferSize(window, &frameSize.x, &frameSize.y); // get frame size
        simWindow->draw(frameSize); // draw UI
        simWindow->update(); // step simulation

      }
      ImGui::EndFrame();

      //// RENDERING ////
      ImGui::Render();
      // Update and Render additional Platform Windows (if viewports enabled)
      if(ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
        {
          GLFWwindow* contextBackup = glfwGetCurrentContext();
          ImGui::UpdatePlatformWindows();
          ImGui::RenderPlatformWindowsDefault();
          glfwMakeContextCurrent(contextBackup);
        }
      // render to screen
      glViewport(0, 0, frameSize.x, frameSize.y);
      glClearColor(0.15f, 0.15f, 0.15f, 1.0f);
      glClear(GL_COLOR_BUFFER_BIT);
      glUseProgram(0);
      ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

      // render 3d lines
      simWindow->postRender();

      // render simulation to file if actively rendering (offline context)
      if(simWindow->fileRendering())
        {
          ImGui::SetCurrentContext(offlineContext);
          simWindow->renderToFile();
          ImGui::SetCurrentContext(mainContext);
        }
      glfwSwapBuffers(window);
    }
  std::cout << "\n\n\n";
  
  cleanup();
  return 0;
}



