# Artificial Intelligence Music Generation Evaluation Framework
This repo contains supplementary code for AIMGEF, which currently covers an article under review named：

*Deep learning's shallow gains: A comparative evaluation of algorithms for automatic music generation*

And the published conference paper:
```
@inproceedings{yin2021good,
  title={“A Good Algorithm Does Not Steal--It Imitates”: The Originality Report as a Means of Measuring When a Music Generation Algorithm Copies Too Much},
  author={Yin, Zongyu and Reuben, Federico and Stepney, Susan and Collins, Tom},
  booktitle={Artificial Intelligence in Music, Sound, Art and Design: 10th International Conference, EvoMUSART 2021, Held as Part of EvoStar 2021, Virtual Event, April 7--9, 2021, Proceedings 10},
  pages={360--375},
  year={2021},
  organization={Springer International Publishing}
}
```


Please find the following procedures for reproducing the results stated in:

- [Deep Learning's Shallow Gains](#deep-learnings-shallow-gains)
- [A Good Algorithm Does Not Steal - It Imitates](#a-good-algorithm-does-not-steal---it-imitates)

## Deep Learning's Shallow Gains
### Data preprocess
- CSQ: `python main.py --mode CSQ_DATA`
- CPI: `python main.py --mode CPI_DATA`
 
([MAESTRO](https://magenta.tensorflow.org/datasets/maestro) dataset need to be manually downloaded and placed under 'dataset' folder)

### Model training
- Reimplementation: `python main.py --model "model" --style "style" --mode TRAIN`
  - `"model"`: `Transformer`, `VAE`
  - `"style"`: `CSQ`, `CPI`
- MAIA Markov: check [installation guide](https://www.npmjs.com/package/maia-markov)
- Coupled recurrent model: check [authors' repository](https://github.com/jthickstun/ismir2019coupled)

### Excerpts generation
`python main.py --model "model" --style "style" --mode GEN --src "checkpoint" --epoch "epoch"`
- `"checkpoint"` is automatically generated during training, named by datetime (e.g., 20201113-182345)
- `"epoch"` is the epoch number (e.g., 10)

### Stimuli
The generated stimuli used in the listening study can be found [here](https://github.com/mstrcyork/aimgef-assets) with the corresponding "index"(.midi) indicating category:

| |CSQ|CPI|
|:-|:-|:-|
|Orig|1-25|151-175|
|MuTr|101-125|201-225|
|MVAE|76-100|176-200|
|MaMa|26-50|-|
|CoRe|51-75|-|
|BeAf|126-150|-|
|LiTr|-|226-250|

## A Good Algorithm Does Not Steal - It Imitates
### Model training and excerpts generation
#### Get Classical string quartets from KernScore
`python main.py --mode CSQ_DATA`

#### Construct dataset for Transformer training
`python ./model/utilities/preprocessor.py`

#### Train Transformer
`python main.py --model Transformer --style CSQ --mode TRAIN`

#### Generate excerpts from trained Transformer
`python main.py --model Transformer --style CSQ --mode TRAIN --src "checkpoint dir" --epoch 3`

`checkpoint dir` is automatically generated during training, named by datetime (e.g., 20201113-182345).

### Originality check
`cd ./pubs_material/evomusar_2021` before conducting the following procedures.

#### Originality baseline
`cd originality_report`

`node originality_report_strq2020.js -midiDirs "midiDirs" -testItems "testItems"`

`midiDirs` is the directory name for the target corpus.

`testItems` is the directory name for the query corpus.

It produces originality analysis and saved in json files for each test item. The R file `originality_report_human_strq2020.R` contains the sampled similarity values and calculate the mean and 0.95 confidence interval, but you can substitute your own values from the originality analysis.

#### Figures
There are four folders contains scripts used to make corresponding figures, you can copy them to [p5js](https://editor.p5js.org/) for reproduction.

Specifically, the folder `cardinality_score_figure` contains the script `evomusart2020_figure_prep.js` to get point sets with given two midi file names and locations in bars.

For example, `node evomusart2020_figure_prep.js -src 249-16-32 -tgt 1207-696-712`,

with format `midi-start-end`.