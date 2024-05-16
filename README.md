# Heads-Up Multitasker
Heads-Up Multitasker -- Code Repository. This is the more than the project's playground, but also replication factory, and the extension start-off place. Please wait for my updates patiently, thank you.

## Publications
- [Proceedings of the CHI Conference on Human Factors in Computing Systems]([publication_link](https://programs.sigchi.org/chi/2024/program/content/147957)), CHI'24
```
@inproceedings{bai2024hum,
  title={Heads-Up Multitasker: Simulating Attention Switching On Optical Head-Mounted Displays},
  author={Bai, Yunpeng and Ikkala, Aleksi and Oulasvirta, Antti and Zhao, Shengdong and Wang, Lucia J and Yang, Pengzhi and Xu, Peisen},
  booktitle={Proceedings of the 2024 CHI Conference on Human Factors in Computing Systems},
  pages={1--18},
  year={2024},
  doi={10.1145/3613904.3642540},
}
```

## Contact person
- [Bai Yunpeng](https://baiyunpeng1949.github.io/)


## Project links
- [CHI presentation slides](https://docs.google.com/presentation/d/11h_Gqf5_tsO0lDSJ372IiiSVOCKMHB8FO7So-E6-HSQ/edit#slide=id.p)
- [Download presentation slides from Shareslide](https://www.slideshare.net/slideshow/heads-up-multitasker-chi-2024-presentation-pdf/268559304)
- [Project folder](https://drive.google.com/drive/folders/1WEG9DFROf_-a5l_sA2YVunc2B2P70__6?ths=true) 
- [Documentation](guide_link)
- [Full paper](https://github.com/Synteraction-Lab/heads-up-multitasker/blob/main/Heads-Up%20Multitasker%20Full%20Paper.pdf); Of course you may access it through ACM digital lib.
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
- If you want to train the model, GPU is needed. If you want to use my pre-trained model, good luck! Since models are dependent on hardware, I trained models on two different machines, sadly one cannot run on the other ...
- If you want to run codes on google cloud or any linux platform, remember to
  ```bash
  export MUJOCO_GL=egl
  ```

## Application Execution 
- If you want to replicate this work, run main.py; remember to set up the configuration file! 
- If you want to replicate study results, use data in the study folder, and run jupyter-notebook files. These files also contain important parameter inference process.
- Contacting me for help is the most efficient replicating way. I am still maintaining the current repo.


## Contributors
Bai Yunpeng, Aleksi Ikkala, Antti Oulasvirta, Shengdong Zhao, 



