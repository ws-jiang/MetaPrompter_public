# MetaPrompter

### ENV
- environments:
  - torch=1.12.1
  - datasets=2.7.0
  - transformers=4.24.0
  - yaml=6.0
  - mysql-connector-python

- Tesla V100S-32GB x 8, NVIDIA A100-80GB x 8
- note: when the OOM problem happens, can reduce the number of query samples (in [configs/few_shot_settings.yaml](./configs/few_shot_settings.yaml))
or increase the GPU memory.

### Experiments on RepVerb
- refer to [code/run_verb.sh](./code/run_verb.sh) 

### Experiments on MetaPrompter
- refer to [code/run_meta.sh](./code/run_meta.sh) 

### References
- [MetaPrompting](https://github.com/Dousia/MetaPrompting)
- [ContrastNet](https://github.com/BDBC-KG-NLP/AAAI2022_ContrastNet)
- [MLADA](https://github.com/hccngu/MLADA)

### bibtex

```
@InProceedings{jiang2023effective,
  author    = {Jiang, Weisen and Zhang, Yu and Kwok, James},
  booktitle = {International Conference on Machine Learning},
  title     = {Effective Structured Prompting by Meta-Learning and Representative Verbalizer},
  year      = {2023},
}
```