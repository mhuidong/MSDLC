# MSDLC
MSDLC: multi-source data lossless compression via parallel expansion mapping and xLSTM

# Usage
Take the file **test.txt** as an example:

compression
```
python compress.py test.txt test.msdlc --prefix test
```

decompression
```
python decompress.py test.msdlc test.msdlc.out --prefix test
```



# Data
| Dataset  | Type   | Size (B)  | Description                                                   | Link                                               |
|-------|-------------|-----------|---------------------------------------------------------------|----------------------------------------------------|
| SAO   | homogeneous | 7251944   | Files containing information of 258,996 stars                 | https://sun.aei.polsl.pl/~sdeor/corpus/silesia.zip |
| NCI   | homogeneous | 33553445  | Files in SDF format                                           | https://sun.aei.polsl.pl/~sdeor/corpus/silesia.zip |
| X-RAY | image       | 8474240   | 12-bit grayscale scaled x-ray medical image of a child's hand | https://sun.aei.polsl.pl/~sdeor/corpus/silesia.zip |
| MR    | image       | 9970564   | A magnetic resonance medical image of the head                | https://sun.aei.polsl.pl/~sdeor/corpus/silesia.zip |
| HaHi  | genome      | 3890005   | The DNA sequence of a species                                 | https://sweet.ua.pt/pratas/datasets/DNACorpus.zip  |
| DrMe  | genome      | 32181429  | The DNA sequence of a species                                 | https://sweet.ua.pt/pratas/datasets/DNACorpus.zip  |

# Additional Information
Authors: NBJL-AIGroup

Contact us: mahd@nbjl.nankai.edu.cn OR sunh@nbjl.nankai.edu.cn
