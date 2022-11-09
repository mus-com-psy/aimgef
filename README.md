# Artificial Intelligence Music Generation Evaluation Framework (AIMGEF)
This repo contains code supporting the AIMGEF project, which aims to provide a larger and more robust set of evaluation measures for automatic music generation than loss and accuracy.

## Related papers

```
@article{yin2022deep,
	title={Deep learning's shallow gains: A comparative evaluation of algorithms for automatic music generation},
  author={Yin, Zongyu and Reuben, Federico and Stepney, Susan and Collins, Tom},
	journal={Machine Learning},
	year={in press}
}

@article{yin2022measuring,
	title={Measuring when a music generation algorithm copies too much: The originality report, cardinality score, and symbolic fingerprinting by geometric hashing},
  author={Yin, Zongyu and Reuben, Federico and Stepney, Susan and Collins, Tom},
	journal={Springer Nature Computer Science},
	year={2022},
	volume={3},
	number={340},
	pages={1--18}
}

@inproceedings{yin2021good,
  title={``A good algorithm does not steal -- it imitates'': The originality report as a means of measuring when a music generation algorithm copies too much},
  author={Yin, Zongyu and Reuben, Federico and Stepney, Susan and Collins, Tom},
  booktitle={Artificial Intelligence in Music, Sound, Art and Design: 10th International Conference, EvoMUSART 2021, Held as Part of EvoStar 2021, Virtual Event, April 7--9, 2021, Proceedings 10},
  pages={360--375},
  year={2021},
  organization={Springer}
}
```

Below are instructions for reproducing the results reported in:

- [Deep Learning's Shallow Gains](#deep-learnings-shallow-gains)
- [A Good Algorithm Does Not Steal - It Imitates](#a-good-algorithm-does-not-steal---it-imitates)

## Deep Learning's Shallow Gains
### Data preprocessing
- CSQ: `python main.py --mode CSQ_DATA`
- CPI: `python main.py --mode CPI_DATA`

The CSQ dataset, which we used for training in the CSQ task, can be downloaded from [here](https://tomcollinsresearch.net/research/data/). It consists of a fast_first folder containing the kern and MIDI files used in this project, as well as a not_fast_first folder containing other string quartets by Haydn, Mozart, and Beethoven not used in this project. We have put just the MIDI files from the fast_first folder in [aimgef-assets](https://github.com/mstrcyork/aimgef-assets/) too.

The [MAESTRO](https://magenta.tensorflow.org/datasets/maestro) dataset, which we used for training in the CPI task, needs to be downloaded and placed in the "dataset" folder.

### Model training
- Reimplementation: `python main.py --model "model" --style "style" --mode TRAIN`
  - `"model"`: `Transformer`, `VAE`
  - `"style"`: `CSQ`, `CPI`
  - The hyperparameters for each model is specified in `./model/config.yaml`.
- [MAIA Markov](https://www.npmjs.com/package/maia-markov):
  - Version 0.0.5 was used to generate stimuli for the listening study.
  - The four scripts beginning strq2020_... in the examples folder are the ones to run, and a script with "analyze" in the filename will need to be run before the corresponding script with "generate" in the file name, because the former creates the state transition matrices and initial distributions from which the latter samples.
- Coupled recurrent model: please see the [authors' repository](https://github.com/jthickstun/ismir2019coupled).

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

### Hypothesis testing
The scripts of non-parametric Bayesian hypothesis testing mentioned in the paper can be found in `bayes_factor` folder.
These R scripts are adapted from the [original code](https://osf.io/gny35/). The statistic results can be replicated by running `LatentNormalTTestSimulationStudy.R` or `LatentNormalSpearmanSimulationStudy.R`,
in which the target scenario need to be specified for `allScenarios`, for example, `CSQ-MaMa-MuTr-Ss` stands for comparing stylistic success ratings between Maia Markov and Music Transformer in CSQ part of study.
After running, an Rdata file containing Bayes factor is generated.


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
