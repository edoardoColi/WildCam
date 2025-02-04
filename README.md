- `uname -a` display system information (Kernel name, Hostname, Kernel release, Kernel version, Architecture, Hardware platform, OS name)
- `head -n 1 /etc/nv_tegra_release` useful for confirming the specific version of NVIDIA's software stack running. Here can find:
    - **`RXX (release)`**: Major software release version.
    - **`REVISION: X.X`**: Revision version of the software release.
    - **`GCID: XXXXXXXX`**: NVIDIA internal identifier for the release.
    - **`BOARD: XXXXXXX`**: Board identifier.
    - **`EABI: XXXXXXX`**: Application Binary Interface.
    - **`DATE: XXXXXXXX`**: Build date and time.

The difference between the Docker images `ultralytics/ultralytics:latest` and `ultralytics/ultralytics:latest-jetson-jetpackX` lies in their target hardware, software environment, and optimizations:

### Key Differences

| **Feature**               | **latest**                                 | **latest-jetson-jetpack6**                |
|---------------------------|--------------------------------------------|-------------------------------------------|
| **Target Hardware**       | General-purpose systems (x86_64, CUDA GPUs)| NVIDIA Jetson devices (ARM-based, aarch64)|
| **CUDA Version**          | Standard CUDA (desktop/server GPUs)        | CUDA tailored for JetPack environment     |
| **Optimization**          | General PyTorch optimizations              | TensorRT and JetPack-specific tuning      |
| **Architecture**          | x86_64                                     | ARM64 (aarch64)                           |
| **Use Case**              | Training and inference                     | Deployment on edge devices                |

### Modify Docker Daemon Configuration
```
{
  "default-runtime": "nvidia",
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  },
  "storage-driver": "overlay2",
  "data-root": "/var/lib/docker",
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "100m",
    "max-file": "3"
  },
  "no-new-privileges": true,
  "experimental": false
}
```
Add this content to the **/etc/docker/daemon.json** file and after modifying the daemon.json file, you need to restart the Docker service to apply the configuration (`sudo systemctl restart docker`).

# Setup Jetson tips

### Wifi configs
```
nmcli device wifi list
nmcli connection up id "<SSID>"

sudo nmcli device wifi connect "<SSID>" password "<password>"

nmcli connection show

sudo nmcli connection add type ethernet ifname <ethX> con-name  "<Ethernet>"

nmcli connection show

# Higher route-metric has low priority
sudo nmcli connection modify "<nome-tipo-Ethernet>" ipv4.route-metric 200 
sudo nmcli connection modify "<nome-tipo-Wifi>" ipv4.route-metric 100 
```
if needed reboot to apply changes.

### Desktop GUI
```
sudo init 3    # Stop the desktop graphical user interface
sudo init 5    # Restart the desktop graphical user interface

#to disable desktop on boot
sudo systemctl set-default multi-user.target
#to enable back
sudo systemctl set-default graphical.target

# Camera applications and programs relying on the Argus daemon will not be able to access the camera unless the service is manually started
sudo systemctl disable nvargus-daemon.service
```

### Mounting SWAP
Run those to disable ZRAM and create a swap file, if have a NVMe SSD is preferred to allocate there the swap
```
sudo systemctl disable nvzramconfig
sudo fallocate -l 16G /ssd/16GB.swap
sudo mkswap /ssd/16GB.swap
sudo swapon /ssd/16GB.swap

#so add the following line to the end of /etc/fstab to make the change persistent
/ssd/16GB.swap	none	swap	sw	0	0
```

### Good Practice for jetson running YOLOv11
```
sudo nvpmodel -q
sudo rm -rf /etc/nvpmodel.conf

# Enable MAX Power, let works with all CPU and GPU cores
#15W
sudo nvpmodel -m 0 
#7W
sudo nvpmodel -m 1
#MAXN
sudo nvpmodel -m 2

#by enabling jetson clock all CPU and GPU cores will be clocked at their maximum frequency
sudo jetson_clocks
#to return to the normal configuration
sudo jetson_clocks --restore
```

### Application for monitoring
```
sudo apt update
sudo pip install jetson-stats
sudo reboot
jtop

sudo apt update
sudo apt upgrade
sudo apt install nvidia-cuda-toolkit
nvprof --version

#tegrastats is pre-installed with the JetPack SDK
tegrastats
```

#### Key Differences

| Feature                       | **`tegrastats`**                                   | **`jtop`**                                   | **`nvprof`**                                  |
|-------------------------------|---------------------------------------------------|----------------------------------------------|-----------------------------------------------|
| **Purpose**                    | Real-time system resource monitoring             | Real-time system monitoring with graphical interface | GPU profiling and performance analysis of CUDA applications |
| **Interface**                  | Command-line, raw text output                    | Curses-based interactive graphical interface | Command-line output with detailed profiling results |
| **Primary Focus**              | CPU, GPU, memory, power, and temperature stats   | CPU, GPU, memory, power, temperature, and process stats | Performance analysis of CUDA kernels and GPU activity |
| **Real-time Updates**          | Yes, shows stats in real-time                     | Yes, real-time stats with graphical display  | Yes, real-time profiling data while running CUDA programs |
| **GPU Monitoring**             | Shows GPU utilization and frequency               | Shows GPU usage, memory usage, temperature   | Detailed profiling of GPU, kernel launches, memory usage, and more |
| **CPU Monitoring**             | CPU usage per core                               | CPU usage per core, system processes          | Does not monitor CPU usage directly            |
| **Memory Monitoring**          | Shows system RAM usage (used/total)               | Shows memory usage (RAM and swap)             | Tracks memory usage related to CUDA memory allocations |
| **Power Monitoring**           | Shows power consumption (for different components like CPU, GPU, etc.) | Shows power consumption and fan speed (on supported devices) | Does not monitor power consumption directly   |
| **Temperature Monitoring**     | Shows temperature of CPU, GPU, and other components | Shows CPU, GPU, and system temperature        | Does not provide system temperature data     |
| **Process Monitoring**         | No process-specific monitoring                    | Shows which processes are consuming resources | No process-level monitoring                    |
| **Supported Platforms**        | All Jetson devices (Nano, TX1, TX2, Xavier, AGX Xavier) | All Jetson devices (Nano, TX1, TX2, Xavier, AGX Xavier) | All platforms supporting CUDA (including Jetson and desktop GPUs) |
| **Requires CUDA**              | No                                               | No                                           | Yes, specifically designed for profiling CUDA applications |
| **Command Example**            | `sudo tegrastats`                                | `jtop` (interactive interface)               | `nvprof ./my_cuda_app`                        |

### One-Click YOLO on Jetson
The jetson-examples repository by Seeed Studio offers a seamless, one-line command deployment to run vision AI and Generative AI models on the NVIDIA Jetson platform (https://github.com/Seeed-Projects/jetson-examples).
```
pip install jetson-examples

sudo reboot

reComputer list
reComputer run ultralytics-yolo
#enter http://device_ip:5000 to access WebUI
```

### .pt model to .engine
Have to pass trough the onnx format. Using python and YOLO from ultralytics we can say to open the `.pt` model and export in `.onnx`  
After this we can use `trtexec --onnx=yolomodel.pt --saveEngine=yoloModel.engine --fp16`. If not present the command do:
```
apt update
apt install tensorrt python3-libnvinfer libnvinfer-bin

# if is in the directory export the path
ls /usr/src/tensorrt/bin
export PATH=$PATH:/usr/src/tensorrt/bin

# test with
trtexec --version
```
informations regarding the engine model can be retrieved using
```
trtexec --loadEngine=yolo11x.engine --dumpLayerInfo

trtexec --loadEngine=yolov8m.engine --exportProfile=profile8m.json
```

<!-- # Hardware Compatibility tips

###
# Cam IMX219-83 Stereo Camera Compatibility
###

#Verify using
ls /dev/video*
sudo dmesg | grep imx219
# install v4l2-ctl to query cam capabilities
sudo apt install v4l-utils
v4l2-compliance -d /dev/video0
v4l2-ctl --all
v4l2-ctl --list-devices
v4l2-ctl --device=/dev/video1 --list-formats-ext

#Test using
#... but the imx219 is a RG10 e non e' ben supportata
# Per prendere i raw dati uso... FORSE???
v4l2-ctl -d /dev/video0 --set-fmt-video=width=3280,height=2464,pixelformat=RG10 --set-ctrl=sensor_mode=0 --stream-mmap --stream-count=1 --set-ctrl bypass_mode=0 --stream-to=test.raw

#Configura il formato Bayer Usa v4l2-ctl per impostare il formato di acquisizione su Bayer. Se, ad esempio, la tua telecamera supporta il formato RG10 (10-bit Bayer), puoi configurare la telecamera con il comando:
v4l2-ctl -d /dev/video0 --set-fmt-video=pixelformat=RG10
#Controlla la risoluzione e altri parametri Puoi anche verificare la risoluzione supportata e impostarla con un comando come:
v4l2-ctl -d /dev/video0 --set-parm=30        # Imposta il frame rate (ad esempio 30 fps)
v4l2-ctl -d /dev/video0 --set-fmt-video=width=3280,height=2464
#Inoltre, verifica la configurazione della telecamera con v4l2-ctl:
v4l2-ctl -d /dev/video0 --get-fmt-video

# con la classica USB
gst-launch-1.0 v4l2src device=/dev/video2 ! videoconvert ! autovideosink

# e poi lancio questo docker per avere yolo e altri comandi
sudo docker run -it --ipc=host --runtime=nvidia -v /home/edoardo/tentativo1/:/qui ultralytics/ultralytics:latest-jetson-jetpack6
# l'immagine docker presa e riusata con
sudo docker save -o tentativo1/immagineUltalytics.tar ultralytics/ultralytics:latest-jetson-jetpack6
sudo docker load -i tentativo1/immagineUltralytics.tar


###
#
###

sudo apt-cache show nvidia-jetpack

uname -r

cat /etc/nv_tegra_release -->