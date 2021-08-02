# GPEN on AMD MIGraphX
GPEN model is a generative image restoration model on the wild. This repository explores its inference execution on AMD MIGraphX graph optimization library.

# Try GPEN release on ROCm PyTorch
1) Run ROCm PyTorch Docker, depending on your ROCm version: `rocm/pytorch:rocm4.1_ubuntu18.04_py3.6_pytorch_1.8.0`. Typical `docker run` command is given below:<br>
```
docker run -it --name <NAME_HERE> -v $HOME:/data -rm --device=/dev/kfd --device=/dev/dri --group-add video --network=host --privileged  -v=`pwd`:/code/AMDMIGraphX rocm/pytorch:rocm4.1_ubuntu18.04_py3.6_pytorch_1.8.0	
```
Please note you may not need `--network=host` or `--privileged` if you don't need additional accesses to system and host network. <br>

2) Instrall requirements `pip3 install -r requirements.txt`
3) Try `python3 GPEN/face_enhancement.py`

<b>Typical output:</b>
```
examples/gpen/GPEN# python3 face_enhancement.py 
/code/AMDMIGraphX/AMDMIGraphX/examples/gpen/GPEN/face_model/op/fused_bias_act.cpp -> /code/AMDMIGraphX/AMDMIGraphX/examples/gpen/GPEN/face_model/op/fused_bias_act.cpp ok
/code/AMDMIGraphX/AMDMIGraphX/examples/gpen/GPEN/face_model/op/fused_bias_act_kernel.cu -> /code/AMDMIGraphX/AMDMIGraphX/examples/gpen/GPEN/face_model/op/fused_bias_act_kernel.hip ok
Successfully preprocessed all matching files.
/code/AMDMIGraphX/AMDMIGraphX/examples/gpen/GPEN/face_model/op/upfirdn2d.cpp -> /code/AMDMIGraphX/AMDMIGraphX/examples/gpen/GPEN/face_model/op/upfirdn2d.cpp ok
/code/AMDMIGraphX/AMDMIGraphX/examples/gpen/GPEN/face_model/op/upfirdn2d_kernel.cu -> /code/AMDMIGraphX/AMDMIGraphX/examples/gpen/GPEN/face_model/op/upfirdn2d_kernel.hip ok
Successfully preprocessed all matching files.

0 Solvay_conference_1927.png
```
<i>Note the automatic `.cu` to `.hip` conversion, and successful execution.</i>

# Create ONNX model from PyTorch

The GPEN model has two models: RetinaFace and FaceGAN. 

- RetinaFace ONNX model can be generated using the script provided [here](https://github.com/cagery/GPEN/blob/main/pt_to_onnx_retinafacedetection.py).
- FaceGAN model requires `FusedLeakyReluFunction` operator to be defined in PyTorch to be converted to ONNX. Currently it is erroring with `RuntimeError: ONNX export failed: Couldn't export Python operator FusedLeakyReLUFunction`. This operator can be defined as a custom operator in PyTorch and then can be converted to ONNX. Deferred to be done later due to priority. 
The script of conversion is [here](https://github.com/cagery/GPEN/blob/main/pt_to_onnx_facegan.py).

