# graphene

**(Note: in development)**

Interactive simulation of a subset of Maxwell's equations in 3D. Part of an academic project involving current generated in graphene via the motion of ions in an electrolytic solution.

The field equations are solved over an Eulerian grid given initial conditions and user input, and results are rendered with a ray marching algorithm (uses CUDA).

Most parameters are configurable in the settings. With the right tuning, many patterns can be generated similar to those seen in physical electromagnetic phenomena.


[Notable screenshots/video](https://drive.google.com/drive/folders/1zEHwl77b6Ec9WtRbWh4XJpnevpABOP_p?usp=sharing)

| <a href="https://drive.google.com/uc?export=view&id=1zl8gu8QQvua1m2o8ijav5wlqP59IcEW6"> <img src="https://drive.google.com/uc?export=view&id=1zl8gu8QQvua1m2o8ijav5wlqP59IcEW6" style="width: 720px; max-width: 120%; height: auto" title="(signals drawn in with Ctrl+Click)" /> </a> |
|:--:|
| *Simple 2D wave interference* |

&nbsp;
&nbsp;

| <a href="https://drive.google.com/uc?export=view&id=1CpcEoaUr8yo6XmALOgEZGP-091WoK5de"> <img src="https://drive.google.com/uc?export=view&id=1CpcEoaUr8yo6XmALOgEZGP-091WoK5de" style="width: 720px; max-width: 120%; height: auto" title="refraction" /> </a> |
|:--:|
| *Refraction through a convex lens with a higher index of refraction* |

&nbsp;
&nbsp;

| <a href="https://drive.google.com/uc?export=view&id=1Kq0a_GqLWJhaHri3PnhyEGYZnWNCPXJ_"> <img src="https://drive.google.com/uc?export=view&id=1Kq0a_GqLWJhaHri3PnhyEGYZnWNCPXJ_" style="width: 720px; max-width: 120%; height: auto" title="reflection" /> </a> |
|:--:|
| *External reflection* |

&nbsp;
&nbsp;

| <a href="https://drive.google.com/uc?export=view&id=1nIr-fvvYRoXdr3MjvS1lAsmGIPYdW31c"> <img src="https://drive.google.com/uc?export=view&id=1nIr-fvvYRoXdr3MjvS1lAsmGIPYdW31c" style="width: 720px; max-width: 120%; height: auto" title="complex structures" /> </a> |
|:--:|
| *Complex evolution of 3D structures* |

&nbsp;
&nbsp;

| <a href="https://drive.google.com/uc?export=view&id=110WpiDtRWGXFw5nhP4PRF3OTfOctYbmg"> <img src="https://drive.google.com/uc?export=view&id=110WpiDtRWGXFw5nhP4PRF3OTfOctYbmg" style="width: 720px; max-width: 120%; height: auto" title="cell blending for ray traversal" /> </a> |
|:--:|
| *_Adjustable blending parameters can create interesting effects_* |

&nbsp;
&nbsp;

| <a href="https://drive.google.com/uc?export=view&id=1LyCm-bnBcS9zJG89WuR1K-Z366x7A59T"> <img src="https://drive.google.com/uc?export=view&id=1LyCm-bnBcS9zJG89WuR1K-Z366x7A59T" style="width: 720px; max-width: 120%; height: auto" title="(e.g. shape of a photon)" /> </a> |
|:--:|
| *Emergent patterns with similarities to physical phenomena* |

&nbsp;
&nbsp;

| <a href="https://drive.google.com/uc?export=view&id=1wgZSPbGA96rzVjen5N08tknNm0ihUT6E"> <img src="https://drive.google.com/uc?export=view&id=1wgZSPbGA96rzVjen5N08tknNm0ihUT6E" style="width: 720px; max-width: 120%; height: auto" title="try: sin(len(r)^2/222)" /> </a> |
|:--:|
| *Parametric initial conditions* |

&nbsp;
&nbsp;

| <a href="https://drive.google.com/uc?export=view&id=1iFxHzkrnFcHcnyn8M7z-7oy_gtHmkWu8"> <img src="https://drive.google.com/uc?export=view&id=1iFxHzkrnFcHcnyn8M7z-7oy_gtHmkWu8" style="width: 720px; max-width: 120%; height: auto" title="broccoli³" /> </a> |
|:--:|
| *...?* |

&nbsp;
&nbsp;


# Usage
##### NOTE: Currently only Ubuntu(20.04) is directly supported, but code will likely build on other systems with minor modifications (mainly CMakeLists.txt)

## Basic dependencies
    $ sudo apt install build-essential cmake libglew-dev libfreetype6-dev xorg-dev

&nbsp; 
&nbsp;

## CUDA
##### (requires a compatible NVIDIA graphics card)
##### NOTE: may need to disable Secure Boot via BIOS on laptops
#### Install NVIDIA driver
    $ sudo apt purge *nvidia*
    $ sudo apt autoremove
    $ sudo apt install nvidia-driver-470
#### Install CUDA (11.4)
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
