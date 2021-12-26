# graphene

**(Note: in development)**

Interactive simulation of a subset of Maxwell's equations in 3D. Part of an academic project involving current generated in graphene via the motion of ions in an electrolytic solution.

The field equations are solved over an Eulerian grid given initial conditions and user input, and results are rendered with a ray marching algorithm (uses CUDA).

Most parameters are configurable in the settings. With the right tuning, many patterns can be generated similar to those seen in physical electromagnetic phenomena.

[Notable screenshots/video](https://drive.google.com/drive/folders/1zEHwl77b6Ec9WtRbWh4XJpnevpABOP_p?usp=sharing)

&nbsp;
&nbsp;
| ![(signals drawn in with Ctrl+Click)](images/simple-interference.png) |
|:--:|
| *Simple 2D wave interference* |

&nbsp;
&nbsp;
| ![refraction](images/convex-lens-3d-1.png) |
|:--:|
| *Refraction through a convex lens with a higher index of refraction* |

&nbsp;
&nbsp;
| ![reflection](images/maxwells-equations-materials4.png) |
|:--:|
| *External reflection* |

&nbsp;
&nbsp;
| ![now with twice the slits!](images/double-slit-experiment-improved.png) |
|:--:|
| *Double Slit experiment* |

&nbsp;
&nbsp;
| ![(e.g. shape of a photon)](images/vector-field-photon4.png) |
|:--:|
| *Emergent patterns with similarities to physical phenomena (1)* |

&nbsp;
&nbsp;
| ![(e.g. fusion tokomak / magnetic dynamo?)](images/interesting-pattern.png) |
|:--:|
| *Emergent patterns with similarities to physical phenomena (2)* |

&nbsp;
&nbsp;
| ![complex structures](images/composite-render4.png) |
|:--:|
| *Complex evolution of 3D structures* |

&nbsp;
&nbsp;
| ![cell blending for ray traversal](images/composite-render6.png) |
|:--:|
| *Adjustable blending parameters can create interesting effects* |

&nbsp;
&nbsp;
| ![try: sin(len(r)^2/222)](images/maxwells-equations-test1.png) |
|:--:|
| *Parametric initial conditions* |

&nbsp;
&nbsp;
| ![broccoli³](images/hex-qub.png) |
|:--:|
| *...?* |

&nbsp;
&nbsp;


# Usage
##### NOTE: Currently only supports Ubuntu(20.04), but may build on other systems with minor modifications

## Basic dependencies
    $ sudo apt install build-essential cmake libglew-dev libglfw3-dev nlohmann-json3-dev libfreetype6-dev
        
&nbsp; 
&nbsp;

## CUDA
##### (requires a compatible NVIDIA graphics card)
#### Install NVIDIA driver
    $ sudo apt purge *nvidia*
    $ sudo apt autoremove
    $ sudo apt install nvidia-driver-460
#### Install CUDA (11.2)
    $ sudo apt install nvidia-cuda-toolkit
    
&nbsp;
&nbsp;
        
## Build/Run
    $ ./make-release.sh
    $ ./graphene

&nbsp;
&nbsp;

## Dependencies
- GLFW3
  - https://www.glfw.org/
- GLEW
  - http://glew.sourceforge.net/
- Dear ImGui,
  - https://github.com/ocornut/imgui
- FreeType
  - https://www.freetype.org
- stb (stb_image, std_image_write)
  - https://github.com/nothings/stb
- nlohmann/json
  - https://github.com/nlohmann/json


## Contact
* skothr@gmail.com
