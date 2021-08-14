





Installation:
 Ubuntu:
  Basic dependencies:
      * sudo apt install build-essential cmake libglew-dev libglfw3-dev nlohmann-json3-dev

  Also requires CUDA (tested with cuda-11.2) and a compatible NVIDIA graphics card
      * sudo apt-get purge *nvidia*
      * sudo apt autoremove
      * sudo apt install nvidia-driver-460
      * sudo apt install nvidia-cuda-toolkit
