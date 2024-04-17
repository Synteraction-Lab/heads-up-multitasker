# Heads-Up Multitasker
Heads-Up Multitasker -- Code Repository. This is the more than the project's playground, but also replication factory, and the extension start-off place. Please wait for my updates patiently, thank you.

## Publications
- [Proceedings of the CHI Conference on Human Factors in Computing Systems]([publication_link](https://programs.sigchi.org/chi/2024/program/content/147957)), CHI'24
```
<Bibtext>
@inproceedings{bai2024hum,
  title={Heads-Up Multitasker: Simulating Attention Switching On Optical Head-Mounted Displays},
  author={Yunpeng, Bai and Aleksi, Ikkala and Oulasvirta, Antti and Shengdong, Zhao and Lucia, J., Wang and Pengzhi, Yang and Peisen, Xu},
  booktitle={Proceedings of the 2024 CHI Conference on Human Factors in Computing Systems},
  pages={1--18},
  year={2024},
  doi={10.1145/3613904.3642540},
}
```

## Contact person
- [Bai Yunpeng](https://baiyunpeng1949.github.io/bai.yunpeng/)


## Project links
- Project folder: [here](https://drive.google.com/drive/folders/1WEG9DFROf_-a5l_sA2YVunc2B2P70__6?ths=true) (New visitors need to request authority)
- Documentation: [here](guide_link)
- [Version info](VERSION.md)


## Requirements
- As in requirement.txt
- But a more effective way: 
  ```bash
  pip install -e .
  ```
- Then you need to install [PyTorch](https://pytorch.org/) that fits your hardware (windows/mac/linux)


## Installation
- Just copy the repo to your local machine.
- If you want to train the model, GPU is needed. If you want to use my pre-trained model, good luck! Because models are dependent on hardware, I trained models on two different machines, one cannot run on the other ...
- If you want to run codes on google cloud or any linux platform, remember to
  ```bash
  export MUJOCO_GL=egl
  ```

## Application Execution 
- If you want to replicate this work, run main.py; remember to set up the configuration file! 
- If you want to replicate study results, use data in the study folder, and run jupyter-notebook files. These files also contain important parameter inference process.
- At the end of the day, contact me for help, that is the most efficient way. My codes are still shitty in terms of structure. I need time to fix this.


## Contributors
- 7 Authors (七武海哈哈)



