# graphene

**(Note: in development)**

Interactive simulation of a subset of Maxwell's equations in 3D. Part of an academic project involving current generated in graphene via the motion of an adjacent electrolytic solution.

The field equations are solved over an Eulerian grid given initial conditions and external interaction, and results are rendered via a ray marching algorithm (uses CUDA).

Most parameters are configurable, but the interface is somewhat unintuitive at the moment (currently being redesigned). But with the right tuning, many patterns can be generated similar to those seen in physical electromagnetic phenomena.

| ![simple 2D wave interference](https://raw.githubusercontent.com/skothr/graphene/main/images/maxwells-equations-materials7.png) | 
|:--:| 
| *Simple 2D wave interference (signals drawn in with Ctrl+Click)** |

| ![refraction due to different material properties](https://raw.githubusercontent.com/skothr/graphene/main/images/maxwells-equations-materials-lens.png) | 
|:--:| 
| *Refraction/lensing due to different material properties (materials drawn in with Alt+Click)** |


| ![external reflection](https://raw.githubusercontent.com/skothr/graphene/main/images/maxwells-equations-materials4.png) | 
|:--:| 
| *External reflection** |

https://raw.githubusercontent.com/skothr/graphene/main/images/maxwells-equations-materials4.png

| ![complex evolution of 3D structures](https://raw.githubusercontent.com/skothr/graphene/main/images/composite-render4.png) | 
|:--:| 
| *Complex evolution of 3D structures* |


| ![adjustable blending parameters for ray marching](https://raw.githubusercontent.com/skothr/graphene/main/images/composite-render6.png) | 
|:--:|
| *Adjustable blending parameters can create interesting effects* |


| ![emerging patterns similar to physical phenomena](https://raw.githubusercontent.com/skothr/graphene/main/images/vector-field-photon4.png) | 
|:--:|
| *Emerging patterns with similarities to physical phenomena* |


| ![parametric initial conditions](https://raw.githubusercontent.com/skothr/graphene/main/images/maxwells-equations-test1.png) | 
|:--:|
| *Parametric initial conditions* |


# Installation
## Ubuntu
### Basic dependencies:
    sudo apt install build-essential cmake libglew-dev libglfw3-dev nlohmann-json3-dev
Also requires CUDA and a compatible NVIDIA graphics card
#### Install NVIDIA driver
    $ sudo apt-get purge *nvidia*
    $ sudo apt autoremove
    $ sudo apt install nvidia-driver-460
  #### Install CUDA
    $ sudo apt install nvidia-cuda-toolkit
