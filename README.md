# NeRF: Building Scenes from Still Images

This project started as an attempt to understand how NeRF works — not just to run it, but to build it from scratch. The goal is to take a few pictures of a scene, and reconstruct it in 3D to get novel views.

It’s based on the original **Neural Radiance Fields** paper by Mildenhall et al. (2020):  
**[NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://arxiv.org/abs/2003.08934)**

The current version assumes your input images are pre-processed using COLMAP for camera pose estimation. But in the following versions, we aim to remove that dependency — ideally, making things work from raw images alone.

## What It Does

- Takes a set of posed images (COLMAP format)
- Learns to represent the 3D scene as a neural field
- Synthesizes views from novel angles

