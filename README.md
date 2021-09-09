# Trajectory Prediction enabled 3D Tracking via Multi-camera collaboration (Our ongoing project)

In this project, we propose a novel real-time pipeline, future prediction assisted feedback control, to address these challenges and achieve multi-camera tracking and 3D visualization. In particular, the proposed multi-camera control system includes three functional component: 
1) Predicting future scene condition by extrapolated frames; 
2) Decision-making of optimal camera settings for predicted scene condition and further optimal control trajectory from current settings to the future settings; 
3) Cameras settings are updated by following the optimal control trajectory to actively adapt to the changes of scene condition. 

Based on the proposed solution, real-time adaption of camera settings to ever-changing scene features could be achieved. Besides, the system could alleviate the negative effect of network latency and achieve synchronous multi-camera control.


<div align=left> Application scenario:
  
<div align=center><img width="539" alt="Screen Shot 2021-09-09 at 1 48 07 PM" src="https://user-images.githubusercontent.com/37515653/132736773-e12e6b7a-2cb4-4c3d-8b30-c14cce246093.png">
 
 <div align=left> Execution flow:
  
<div align=center><img width="463" alt="Screen Shot 2021-09-09 at 1 49 25 PM" src="https://user-images.githubusercontent.com/37515653/132736949-905603f0-1a37-4838-947e-6ac5df5a1561.png">
 
 <div align=left> Collaborative LSTM model:
  
<div align=center><img width="394" alt="Screen Shot 2021-09-09 at 1 50 42 PM" src="https://user-images.githubusercontent.com/37515653/132737112-bcf44c21-cf25-4b8f-8db9-f653adcb567a.png">
  
 <div align=left> Performance evalution:
  
<div align=center><img width="704" alt="Screen Shot 2021-09-09 at 1 52 51 PM" src="https://user-images.githubusercontent.com/37515653/132737393-febc2003-2c1a-494f-bf38-31d5eb0277b1.png">

<div align=left> Pytorch program is developed by using the source code of Social LSTM implementation as a basis (https://github.com/quancore/social-lstm)
