# MSDLC
A multi-source data lossless compression via parallel expansion mapping and xLSTM

# Usage
```
# Compression
python compress.py <file> <file>.ms
# Decompression
python decompress.py <file>.ms <file>.ms.out
```

# Data
| Dataset | Type        | Size (B)  | Link                                               |
|:-------:|:-----------:|:---------:|:--------------------------------------------------:|
| SAO     | homogeneous | 7251944   | https://sun.aei.polsl.pl/~sdeor/corpus/silesia.zip |
| NCI     | homogeneous | 33553445  | https://sun.aei.polsl.pl/~sdeor/corpus/silesia.zip |
| X-RAY   | image       | 8474240   | https://sun.aei.polsl.pl/~sdeor/corpus/silesia.zip |
| MR      | image       | 9970564   | https://sun.aei.polsl.pl/~sdeor/corpus/silesia.zip |
| HaHi    | genome      | 3890005   | https://sweet.ua.pt/pratas/datasets/DNACorpus.zip  |
| DrMe    | genome      | 32181429  | https://sweet.ua.pt/pratas/datasets/DNACorpus.zip  |

# Citation
If you mention this work in your manuscript, please add the following code to your TEX file.
```
@inproceedings{ma2025multi,
  title={Multi-source Data Lossless Compression via Parallel Expansion Mapping and xLSTM},
  author={Ma, Huidong and Sun, Hui and Yi, Liping and Liu, Xiaoguang and Wang, Gang},
  booktitle={ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2025},
  organization={IEEE}
}
```

# Contact us
Email: mahd@nbjl.nankai.edu.cn or sunh@nbjl.nankai.edu.cn  
Nankai-Baidu Joint Laboratory (NBJL)
