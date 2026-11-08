# Headless WebGPU test image for Linux NVIDIA GPUs (built and used on a DGX Spark, GB10).
# Chrome runs WebGPU on the NVIDIA Vulkan driver, which the NVIDIA container runtime mounts
# when the graphics capability is requested (NVIDIA_DRIVER_CAPABILITIES=all).
#
#   docker build -t si-webgpu tools/docker
#   docker run --rm --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all --ipc=host --network host \
#     -v "$PWD":/work -w /work si-webgpu tools/docker/run.sh tests
#
# Base: plain Ubuntu. In the NGC PyTorch image (nvcr.io/nvidia/pytorch:25.09-py3) the NVIDIA
# Vulkan driver loads but cannot create an instance ("Could not get vkCreateInstance via
# vk_icdGetInstanceProcAddr"), even privileged and without the CUDA compat libraries; on plain
# Ubuntu it works once the X client libraries it links against (libX11, libXext) are present.
ARG BASE=ubuntu:24.04
FROM ${BASE}
ENV NVIDIA_DRIVER_CAPABILITIES=all DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends curl gnupg ca-certificates \
      libvulkan1 vulkan-tools libx11-6 libxext6 libegl1 libgl1 \
 && curl -fsSL https://dl.google.com/linux/linux_signing_key.pub | gpg --dearmor -o /usr/share/keyrings/google-chrome.gpg \
 && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/google-chrome.gpg] https://dl.google.com/linux/chrome/deb/ stable main" > /etc/apt/sources.list.d/google-chrome.list \
 && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
 && apt-get install -y --no-install-recommends google-chrome-stable nodejs \
 && rm -rf /var/lib/apt/lists/*
