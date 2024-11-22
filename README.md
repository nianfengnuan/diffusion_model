# 这是一个学习 diffusion model的记录笔记

diffusion model （扩散模型）包括两个过程：前向扩散过程和反向生成过程，前向扩散过程是对一张图像逐渐添加高斯噪音直至变成随机噪音，而反向生成过程则是去除噪音的过程，
这里准备从十篇经典论文入手研究扩散模型

1. **DDPM《Denoising Diffusion Probabilistic Models》**
      DDPM奠基之作，给出了严谨的数学推导，可以复现的代码，完善了整个推理过程

2. **从DDPM到DDIM:《Denoising Diffusion Implicit Models》**
   DDIM改善了反向扩散过程中的噪声水平，后面大部分的diffusion model都采用了该技术
   
3. **击败GANs:《Diffusion Models Beat GANs on Image Synthesis》**
   diffusion model被推向第一波高潮！
   
4. **条件分类器技术进一步发展：《Classifier-Free Diffusion Guidance》**

5. **Image-to-Image经典之作《Palette: Image-to-Image Diffusion Models》**
   可实现图像着色、图像修复、图像剪裁恢复、图像超分等等任务
   
6. **畅游多模态领域：GLIDE**

   经典的三篇text-to-image的论文:DALLE2、Imagen、GLIDE

7. **stable diffusion的原型：《High-Resolution Image Synthesis with Latent Diffusion Models》**

8. **高调进军视频领域：《Video Diffusion Models》**

9. **了不起的attention：《Prompt-to-Prompt Image Editing with Cross Attention Control》**

10. **Unet已死，transformer当立！《Scalable Diffusion Models with Transformers》**

    
