#ifndef GLFW_KEYS_HPP
#define GLFW_KEYS_HPP

#include <string>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

// converts a key sequence string to a GLFW key -- alpha chars should be capitalized.
static int stringToGlfwKey(const std::string &keyStr)
{
  if(keyStr.size() == 0) { return GLFW_KEY_UNKNOWN; } // (empty)
  else if(keyStr == "SPACE"        ) { return GLFW_KEY_SPACE;         }
  else if(keyStr == "'"            ) { return GLFW_KEY_APOSTROPHE;    }
  else if(keyStr == ","            ) { return GLFW_KEY_COMMA;         }
  else if(keyStr == "-"            ) { return GLFW_KEY_MINUS;         }
  else if(keyStr == "."            ) { return GLFW_KEY_PERIOD;        }
  else if(keyStr == "/"            ) { return GLFW_KEY_SLASH;         }
  else if(keyStr == "0"            ) { return GLFW_KEY_0;             }
  else if(keyStr == "1"            ) { return GLFW_KEY_1;             }
  else if(keyStr == "2"            ) { return GLFW_KEY_2;             }
  else if(keyStr == "3"            ) { return GLFW_KEY_3;             }
  else if(keyStr == "4"            ) { return GLFW_KEY_4;             }
  else if(keyStr == "5"            ) { return GLFW_KEY_5;             }
  else if(keyStr == "6"            ) { return GLFW_KEY_6;             }
  else if(keyStr == "7"            ) { return GLFW_KEY_7;             }
  else if(keyStr == "8"            ) { return GLFW_KEY_8;             }
  else if(keyStr == "9"            ) { return GLFW_KEY_9;             }
  else if(keyStr == ";"            ) { return GLFW_KEY_SEMICOLON;     }
  else if(keyStr == "="            ) { return GLFW_KEY_EQUAL;         }
  else if(keyStr == "A"            ) { return GLFW_KEY_A;             }
  else if(keyStr == "B"            ) { return GLFW_KEY_B;             }
  else if(keyStr == "C"            ) { return GLFW_KEY_C;             }
  else if(keyStr == "D"            ) { return GLFW_KEY_D;             }
  else if(keyStr == "E"            ) { return GLFW_KEY_E;             }
  else if(keyStr == "F"            ) { return GLFW_KEY_F;             }
  else if(keyStr == "G"            ) { return GLFW_KEY_G;             }
  else if(keyStr == "H"            ) { return GLFW_KEY_H;             }
  else if(keyStr == "I"            ) { return GLFW_KEY_I;             }
  else if(keyStr == "J"            ) { return GLFW_KEY_J;             }
  else if(keyStr == "K"            ) { return GLFW_KEY_K;             }
  else if(keyStr == "L"            ) { return GLFW_KEY_L;             }
  else if(keyStr == "M"            ) { return GLFW_KEY_M;             }
  else if(keyStr == "N"            ) { return GLFW_KEY_N;             }
  else if(keyStr == "O"            ) { return GLFW_KEY_O;             }
  else if(keyStr == "P"            ) { return GLFW_KEY_P;             }
  else if(keyStr == "Q"            ) { return GLFW_KEY_Q;             }
  else if(keyStr == "R"            ) { return GLFW_KEY_R;             }
  else if(keyStr == "S"            ) { return GLFW_KEY_S;             }
  else if(keyStr == "T"            ) { return GLFW_KEY_T;             }
  else if(keyStr == "U"            ) { return GLFW_KEY_U;             }
  else if(keyStr == "V"            ) { return GLFW_KEY_V;             }
  else if(keyStr == "W"            ) { return GLFW_KEY_W;             }
  else if(keyStr == "X"            ) { return GLFW_KEY_X;             }
  else if(keyStr == "Y"            ) { return GLFW_KEY_Y;             }
  else if(keyStr == "Z"            ) { return GLFW_KEY_Z;             }
  else if(keyStr == "["            ) { return GLFW_KEY_LEFT_BRACKET;  }
  else if(keyStr == "\\"           ) { return GLFW_KEY_BACKSLASH;     }
  else if(keyStr == "]"            ) { return GLFW_KEY_RIGHT_BRACKET; }
  else if(keyStr == "`"            ) { return GLFW_KEY_GRAVE_ACCENT;  }
  else if(keyStr == "ESCAPE" ||
          keyStr == "ESC"          ) { return GLFW_KEY_ESCAPE;        }
  else if(keyStr == "ENTER"        ) { return GLFW_KEY_ENTER;         }
  else if(keyStr == "TAB"          ) { return GLFW_KEY_TAB;           }
  else if(keyStr == "BACKSPACE"    ) { return GLFW_KEY_BACKSPACE;     }
  else if(keyStr == "INSERT"       ) { return GLFW_KEY_INSERT;        }
  else if(keyStr == "DELETE"       ) { return GLFW_KEY_DELETE;        }
  else if(keyStr == "RIGHT"        ) { return GLFW_KEY_RIGHT;         }
  else if(keyStr == "LEFT"         ) { return GLFW_KEY_LEFT;          }
  else if(keyStr == "DOWN"         ) { return GLFW_KEY_DOWN;          }
  else if(keyStr == "UP"           ) { return GLFW_KEY_UP;            }
  else if(keyStr == "PAGE_UP"      ) { return GLFW_KEY_PAGE_UP;       }
  else if(keyStr == "PAGE_DOWN"    ) { return GLFW_KEY_PAGE_DOWN;     }
  else if(keyStr == "HOME"         ) { return GLFW_KEY_HOME;          }
  else if(keyStr == "END"          ) { return GLFW_KEY_END;           }
  else if(keyStr == "CAPSLOCK"     ) { return GLFW_KEY_CAPS_LOCK;     }
  else if(keyStr == "SCROLL_LOCK"  ) { return GLFW_KEY_SCROLL_LOCK;   }
  else if(keyStr == "NUMLOCK"      ) { return GLFW_KEY_NUM_LOCK;      }
  else if(keyStr == "PRINTSCREEN"  ) { return GLFW_KEY_PRINT_SCREEN;  }
  else if(keyStr == "PAUSE"        ) { return GLFW_KEY_PAUSE;         }

  else if(keyStr[0] == 'F' && keyStr.size() > 1) { return (GLFW_KEY_F1 + std::stoi(keyStr.substr(1)) - 1); } // function keys

  else if(keyStr == "MENU"         ) { return GLFW_KEY_MENU;          }

  else if(keyStr == "LALT"         ) { return GLFW_KEY_LEFT_ALT;      } // modifiers
  else if(keyStr == "RALT"         ) { return GLFW_KEY_RIGHT_ALT;     }
  else if(keyStr == "LCTRL"        ) { return GLFW_KEY_LEFT_CONTROL;  }
  else if(keyStr == "RCTRL"        ) { return GLFW_KEY_RIGHT_CONTROL; }
  else if(keyStr == "LSHIFT"       ) { return GLFW_KEY_LEFT_SHIFT;    }
  else if(keyStr == "RSHIFT"       ) { return GLFW_KEY_RIGHT_SHIFT;   }
  
  else                               { return GLFW_KEY_UNKNOWN;       }
}




// converts a key sequence string to a GLFW key -- alpha chars should be capitalized.
static std::string glfwKeyToString(int key)
{
  switch(key)
    {
    case GLFW_KEY_SPACE:          return "SPACE";
    case GLFW_KEY_APOSTROPHE:     return "'";
    case GLFW_KEY_COMMA:          return ",";
    case GLFW_KEY_MINUS:          return "-";
    case GLFW_KEY_PERIOD:         return ".";
    case GLFW_KEY_SLASH:          return "/";
    case GLFW_KEY_0:              return "0";
    case GLFW_KEY_1:              return "1";
    case GLFW_KEY_2:              return "2";
    case GLFW_KEY_3:              return "3";
    case GLFW_KEY_4:              return "4";
    case GLFW_KEY_5:              return "5";
    case GLFW_KEY_6:              return "6";
    case GLFW_KEY_7:              return "7";
    case GLFW_KEY_8:              return "8";
    case GLFW_KEY_9:              return "9";
    case GLFW_KEY_SEMICOLON:      return ";";
    case GLFW_KEY_EQUAL:          return "=";
    case GLFW_KEY_A:              return "A";
    case GLFW_KEY_B:              return "B";
    case GLFW_KEY_C:              return "C";
    case GLFW_KEY_D:              return "D";
    case GLFW_KEY_E:              return "E";
    case GLFW_KEY_F:              return "F";
    case GLFW_KEY_G:              return "G";
    case GLFW_KEY_H:              return "H";
    case GLFW_KEY_I:              return "I";
    case GLFW_KEY_J:              return "J";
    case GLFW_KEY_K:              return "K";
    case GLFW_KEY_L:              return "L";
    case GLFW_KEY_M:              return "M";
    case GLFW_KEY_N:              return "N";
    case GLFW_KEY_O:              return "O";
    case GLFW_KEY_P:              return "P";
    case GLFW_KEY_Q:              return "Q";
    case GLFW_KEY_R:              return "R";
    case GLFW_KEY_S:              return "S";
    case GLFW_KEY_T:              return "T";
    case GLFW_KEY_U:              return "U";
    case GLFW_KEY_V:              return "V";
    case GLFW_KEY_W:              return "W";
    case GLFW_KEY_X:              return "X";
    case GLFW_KEY_Y:              return "Y";
    case GLFW_KEY_Z:              return "Z";
    case GLFW_KEY_LEFT_BRACKET:   return "[";
    case GLFW_KEY_BACKSLASH:      return "\\";
    case GLFW_KEY_RIGHT_BRACKET:  return "]";
    case GLFW_KEY_GRAVE_ACCENT:   return "`";
    case GLFW_KEY_ESCAPE:         return "ESCAPE";
    case GLFW_KEY_ENTER:          return "ENTER";
    case GLFW_KEY_TAB:            return "TAB";
    case GLFW_KEY_BACKSPACE:      return "BACKSPACE";
    case GLFW_KEY_INSERT:         return "INSERT";
    case GLFW_KEY_DELETE:         return "DELETE";
    case GLFW_KEY_RIGHT:          return "RIGHT";
    case GLFW_KEY_LEFT:           return "LEFT";
    case GLFW_KEY_DOWN:           return "DOWN";
    case GLFW_KEY_UP:             return "UP";
    case GLFW_KEY_PAGE_UP:        return "PAGE_UP";
    case GLFW_KEY_PAGE_DOWN:      return "PAGE_DOWN";
    case GLFW_KEY_HOME:           return "HOME";
    case GLFW_KEY_END:            return "END";
    case GLFW_KEY_CAPS_LOCK:      return "CAPSLOCK";
    case GLFW_KEY_SCROLL_LOCK:    return "SCROLL_LOCK";
    case GLFW_KEY_NUM_LOCK:       return "NUMLOCK";
    case GLFW_KEY_PRINT_SCREEN:   return "PRINTSCREEN";
    case GLFW_KEY_PAUSE:          return "PAUSE";
    case GLFW_KEY_F1:             return "F1";
    case GLFW_KEY_F2:             return "F2";
    case GLFW_KEY_F3:             return "F3";
    case GLFW_KEY_F4:             return "F4";
    case GLFW_KEY_F5:             return "F5";
    case GLFW_KEY_F6:             return "F6";
    case GLFW_KEY_F7:             return "F7";
    case GLFW_KEY_F8:             return "F8";
    case GLFW_KEY_F9:             return "F9";
    case GLFW_KEY_F10:            return "F10";
    case GLFW_KEY_F11:            return "F11";
    case GLFW_KEY_F12:            return "F12";
    case GLFW_KEY_F13:            return "F13";
    case GLFW_KEY_F14:            return "F14";
    case GLFW_KEY_F15:            return "F15";
    case GLFW_KEY_F16:            return "F16";
    case GLFW_KEY_F17:            return "F17";
    case GLFW_KEY_F18:            return "F18";
    case GLFW_KEY_F19:            return "F19";
    case GLFW_KEY_F20:            return "F20";
    case GLFW_KEY_F21:            return "F21";
    case GLFW_KEY_F22:            return "F22";
    case GLFW_KEY_F23:            return "F23";
    case GLFW_KEY_F24:            return "F24";
    case GLFW_KEY_F25:            return "F25";
    case GLFW_KEY_MENU:           return "MENU";
    case GLFW_KEY_LEFT_SHIFT:     return "LSHIFT";
    case GLFW_KEY_RIGHT_SHIFT:    return "RSHIFT";
    case GLFW_KEY_LEFT_CONTROL:   return "LCTRL";
    case GLFW_KEY_RIGHT_CONTROL:  return "RCTRL";
    case GLFW_KEY_LEFT_ALT:       return "LALT";
    case GLFW_KEY_RIGHT_ALT:      return "RALT";
      
    default:                      return "<?>";
    }
}

#endif // GLFW_KEYS_HPP
