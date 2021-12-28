# graphene

**(Note: in development)**

Interactive simulation of a subset of Maxwell's equations in 3D. Part of an academic project involving current generated in graphene via the motion of ions in an electrolytic solution.

The field equations are solved over an Eulerian grid given initial conditions and user input, and results are rendered with a ray marching algorithm (uses CUDA).

Most parameters are configurable in the settings. With the right tuning, many patterns can be generated similar to those seen in physical electromagnetic phenomena.

[Notable screenshots/video](https://drive.google.com/drive/folders/1zEHwl77b6Ec9WtRbWh4XJpnevpABOP_p?usp=sharing)

&nbsp;
&nbsp;
| ![images/simple-interference.png](images/simple-interference.png "(signals drawn in with Ctrl+Click)") |
|:--:|
| *Simple 2D wave interference* |

&nbsp;
&nbsp;
| ![images/convex-lens-3d-1.png](images/convex-lens-3d-1.png "refraction") |
|:--:|
| *Refraction through a convex lens with a higher index of refraction* |

&nbsp;
&nbsp;
| ![images/maxwells-equations-materials4.png](images/maxwells-equations-materials4.png "reflection") |
|:--:|
| *External reflection* |

&nbsp;
&nbsp;
| ![images/double-slit-experiment-improved.png](images/double-slit-experiment-improved.png "now with twice the slits!") |
|:--:|
| *Double Slit experiment* |

&nbsp;
&nbsp;
| ![images/vector-field-photon4.png](images/vector-field-photon4.png "(e.g. shape of a photon)") |
|:--:|
| *Emergent patterns with similarities to physical phenomena (1)* |

&nbsp;
&nbsp;
| ![images/interesting-pattern.png](images/interesting-pattern.png "(e.g. fusion tokomak/dynamo?)") |
|:--:|
| *Emergent patterns with similarities to physical phenomena (2)* |

&nbsp;
&nbsp;
| ![images/composite-render4.png](images/composite-render4.png "complex structures") |
|:--:|
| *Complex evolution of 3D structures* |

&nbsp;
&nbsp;
| ![images/composite-render6.png](images/composite-render6.png "cell blending for ray traversal") |
|:--:|
| *Adjustable blending parameters can create interesting effects* |

&nbsp;
&nbsp;
| ![images/maxwells-equations-test1.png](images/maxwells-equations-test1.png "try: sin(len(r)^2/222)") |
|:--:|
| *Parametric initial conditions* |

&nbsp;
&nbsp;
| ![images/hex-qub.png](images/hex-qub.png "broccoli³") |
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
