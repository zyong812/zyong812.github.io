---
title: 'A survey on deep learning applications in multimedia'
date: 2018-11-17
permalink: /posts/2018/11/A_survey_on_deep_learning_applications_in_multimedia/
tags:
  - multimodal 
  - deep_learning
---

Deep learning methods have achieved great success in many research fields.
At least by far, the performances of deep learnning methods are much better than traditional methods.
A good example is that the prediction error of ImageNet competetion dropped significantly since CNN was used in 2012.
Moreover, deep learnning methods can solve some problems which seem unsolvable by hand-crafted methods, such as *image style migration* and *image captioning*.

I will talk about two things in this post.
Firstly, I did a survey on some hot and interesitng applications of deep learning in the filed of multimedia researh.
Secondly, I studied the source code of *img2poetry* [3] and began to learn how to use deep learning tools, especially for *Google TensorFlow*.

## 1. Deep learning applications in multimedia

I did a survey on popular deep learing applications of recent years.
In [1], the author introduces many interesing deep learing applications, and it also introduces how to implement these applications by illustrating the basic ideas and tools.
Tensorflow's official tutorials also demo many interesting deep learning applications in the area of image, sequence, data representation and generative models [2].
Following these demos, we can easily develop our own applications.
Here, I created a catergorized list of popular deep learning applications in the field of multimdeia research as the following table shows.

![deep_learning_applications_on_meltimedia](/assets/images/deep_learning_applications.png)

The checked-mark (√) in the table means that which kinds of multimedia data the application uses.
Roughly, I divide these applications into 2 categories, the unimodal application and multimodal application.
Unimodal application means that it only uses data from the same modal (same multimedia data), and multimodal application uses multimedia data from different modals.

From the above figure, we may conclude that
*  Multimodal deep learning application is relatively less researched than unimodal application. There are many mature unimodal applications developed in recent years, such as image style migration, automated Q/A system. But for multimodal application, there are still a lot to research.
*  The application of deep learning methods on visual data (image/video) is mostly researched. Maybe it is because visual data contains more abundant information.

Furthermore, I categorized above multimodal applications in the table into following 3 classes.

* **Retrival**: We want to retrieve data by using data from another modal. An important work for this task is how to define the similarity between cross-modal data. Example applications from the above table are image retrival by text or audio.
* **Translation**: We want to tranlate data from one modal to another modal with the similar semantic meaning. Example-based and generative deep learning methods are applied for this task. Example applications from above table are image captioning, audio to text translation, text to audio sythesis.
* **Co-learning**: We want to make more robust and accurate predictions by using data from different modals. Example application is audio-video speech recognition. Here I have a question. If we always have sufficient unimodal data, usually we can train good enough prediction model, co-learning may not be so meanningful in this case. Therefore, can we say co-learning is meaningful only when we don't have sufficient unimodal data?


## 2. Deep learning implementation

## 2.1 Img2poem [3] source code reading

The logic of the program goes like follows:

1. Extracting features (1x4096 numeric vector) of the image with the downloaded models (scene_model, obj_model, sentiment_model). 
2. Building a predicting model with downloaded params, and using the above features as input, to generate a poem.

Here are some confusions and comments of mine:

1. Are the feature-extracting model and predicting model trained by authors or public models?
2. We can only see the testing process from the published code, but don't know how to train the models.
3. It uses TensorFlow for implementation. (Another work Multi-Human [4] also uses TensorFlow for implementation)

### 2.2 TensorFlow introduction

TensorFlow is an open-source machine learning library for research and production.
It is created by Google, and widely used for deep learning research.
I will introduce some important concepts here.

* **Tensor**: A tensor is a generalization of vectors and matrices to potentially higher dimensions. Internally, TensorFlow represents tensors as n-dimensional arrays of base datatypes [2].

* **Dataflow graph**: TensorFlow's representation of computations as dependencies between operations.
In a dataflow graph, the nodes represent units of computation, and the edges represent the data consumed or produced by a computation. 
Dataflow has several advantages in terms of parallelism, distributed execution, compilation and portability [2].

![dataflow_graph](/assets/images/tensors_flowing.gif)

* **Session**: TensorFlow's mechanism for running dataflow graphs across one or more local or remote devices [2].


## 3. Discussion

Here I list some quesitions.

Q1: Usually, there are not so much theoretical analysis for multimedia deep learning (eg. [3], [4]).
What are the innovative points for such kind of research? 
Here are some points I thought about:

* Modeling and solving a problem others haven't done
* Innovative and better neural network structure
* Tricks to make higher accuracy and better performance

What are other innovative points I have missed?
Which points are more important?

Q2: What can we do for multimodal analysis in the future?
What kind of applications or methodology we should research?

Q3: Suppose the research topic is settled, since we don't know whether it can be solved or maybe it is too hard to solve, so we may cost a lot of time without any outcome. What should I do then?
For deep learning research, it seems as long as we pay enough efforts, we can get some results more or less.
I am not sure whether it is correct.

I focus my research on multimodal analysis.
Pehaps after I have more experience in this field, I will have more ideas about what exact I should do in the future.

## References

1. https://github.com/Honlan/DeepInterests
2. https://www.tensorflow.org/tutorials/
3. Liu, Bei, et al. "Beyond Narrative Description: Generating Poetry from Images by Multi-Adversarial Training." arXiv preprint arXiv:1804.08473 (2018).
4. Zhao, Jian, et al. "Understanding Humans in Crowded Scenes: Deep Nested Adversarial Learning and A New Benchmark for Multi-Human Parsing." arXiv preprint arXiv:1804.03287 (2018).

