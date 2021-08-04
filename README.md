# [Representing Syntax and Composition with Geometric Transformations](https://aclanthology.org/2021.findings-acl.296/)

Repo contains our PyTorch implementation and training code for the DM model ([Czarnowska et., al 2019](https://aclanthology.org/W19-0408.pdf)). All results presented in the paper for [MuRe](https://github.com/ibalazevic/multirelational-poincare), [RotE, RefE and AttE](https://github.com/HazyResearch/KGEmb) referes to models that have been trained using the origial code provided by the authors.

### Usage 
```bash
python train.py -emsize 300 --neg_num 20 -epochs 5
```

### Acknowledgements
We built our DM Dataloder on top of the one implemented by [Zhenisbek Assylbekov](https://github.com/zh3nis/SGNS)


### Cite
```
@inproceedings{bertolini-etal-2021-representing,
    title = "Representing Syntax and Composition with Geometric Transformations",
    author = "Bertolini, Lorenzo  and
      Weeds, Julie  and
      Weir, David  and
      Peng, Qiwei",
    booktitle = "Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-acl.296",
    doi = "10.18653/v1/2021.findings-acl.296",
    pages = "3343--3353",
}
```
