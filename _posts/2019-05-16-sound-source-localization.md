---
title: Sound Source Localization in Videos
permalink: /posts/2019/05/sound_source_localization
tags:
  - computer vision
  - multimodal
---

This post focuses on the sound source localization task, which aims to locate which region in videos generates the sound.

- [1. Introduction](#1-introduction)
- [2. Mutual Information Method](#2-mutual-information-method)
  - [2.1 Method](#21-method)
  - [2.2 Implementation and Evaluation](#22-implementation-and-evaluation)
    - [Visual and Auditory Intensity Changes over Time](#visual-and-auditory-intensity-changes-over-time)
    - [Sound Localization for Different Periods](#sound-localization-for-different-periods)
    - [Does the method actually find the dynamic regions (actions), instead of the sound source?](#does-the-method-actually-find-the-dynamic-regions-actions-instead-of-the-sound-source)
    - [What will happen by using unmatched audio?](#what-will-happen-by-using-unmatched-audio)
  - [2.3 Discussion](#23-discussion)
- [3. Neural Network Method](#3-neural-network-method)
  - [3.1 Method](#31-method)
  - [3.2 Evaluation](#32-evaluation)
- [4. Summary](#4-summary)
- [References](#references)



## 1. Introduction

<!-- 背景，问题描述，methods, rationale, 结构 -->

People can understand complex auditory and visual information, often using one to disambiguate the other. For example, when we are watching a movie, only by listening to the audio, we can guess with high probability what's happening on the screen. Conversely, we can guess what's one speaking by perceiving how his/her mouth moves. Acoustic and visual signals are highly correlated, since they are both effects caused by same real-world events.
From the perspective of information theory, we say that the mutual information between audio and video is not zero.

Actually, there is an active research field called audio-visual analysis, which is based on the correlation between audio and video.
The challenges or applications of audio-visual analysis can be categorized into 2 kinds as follows:
* The first kind is based on the intuition that, the joint information of audio and video is larger the one of the signal (Figure 1). Hence, by fusing the information of these 2 modalities, it helps computers better understand what is happening in the real world, which will help to solve many tasks. Audio visual speech recognition is a famous example, which fuses the information of facial movements and audio for recognizing speech. Another example is audio-aid video understanding, which includes audio information to help computers understand visual contents, in order to solve tasks such as video classification, action recognition.
* The second kind is based on the the intuition that, the mutual information between audio and video is not zero (Figure 1). Therefore, knowing one of them, we have some information about the other. Sound source localization, audio-visual cross-modality retrieval are examples.

<p align="center">
  <img style="width:80%" src="/assets/images/audio_visual_201905/AV_intuition.png">
  <center>Figure 1: The left figure shows intuition on which audio-visual analysis base, and the right figure is the aim for sound source localization</center>
</p>

My project mainly focuses on the sound source localization task, which aims to find which region in videos generates the sound, as Figure 1 shows.
Especially, I've applied classic mutual information method and deep neural networks. The details are described in next parts of this report. Section 2 will describe the method and evaluation results by using mutual information method, and Section 3 will present how neural network method works, and Section 4 is the discussion and summary.


## 2. Mutual Information Method

### 2.1 Method

The classic method to study the correlation between visual and audio data is based on mutual information. It is a very natural idea, since mutual information tells us how much one random variable tells us about the other. If we consider the acoustic signal and the visual signal for each region as random variables, the mutual information measurement tells us the synchrony between these two modalities, i.e. audition and vision. 

The basic idea of this method can be described as: compute the mutual information between the audio signal and every region of the video, and see which region has larger mutual information. Then we consider this region as the sound source.
Formally, denote the associated random variable for audio as $\pmb{A}(t) \in \Re^n$ at time $t$, and the random variable for video at time $t$ and location $(x,y)$ as $\pmb{V}(x,y,t) \in \Re^m$, and assume that within a short period of time, they have the same distribution respectively.
During a period of time, we have a sequences of observations $\{(\pmb{a}(t_1), \pmb{v}(x,y,t_1)), (\pmb{a}(t_2), \pmb{v}(x,y,t_2)), ... , (\pmb{a}(t_k), \pmb{v}(x,y,t_k))\}$, where $|t_i - t| < \delta \: \forall i = {1,2,...k}$ and $\delta$ is a small value.
Then we can estimate the joint distribution of $\pmb{A}(t)$ and $\pmb{V}(x,y,t)$, based on which, mutual information between audio and all regions of video are computed as

$$
I(\pmb{A}(t);\pmb{V}(x,y,t)) = H(\pmb{A}(t)) + H(\pmb{V}(x,y,t)) - H(\pmb{A}(t), \pmb{V}(x,y,t)).
$$

In practice, consider that $I(\pmb{A}(t);\pmb{V}(x,y,t))$ is sensitive to the absolute value of $H(\pmb{A}(t))$ and $H(\pmb{V}(x,y,t))$, we can use the normalized mutual information instead, which might be a better measurement. The normalized mutual information could be obtained by

$$
I(\pmb{A}(t);\pmb{V}(x,y,t)) = \frac{H(\pmb{A}(t)) + H(\pmb{V}(x,y,t)) - H(\pmb{A}(t), \pmb{V}(x,y,t))}{\sqrt{H(\pmb{A}(t))H(\pmb{V}(x,y,t))}}.
$$

Other normalization methods, such as normalized by max or arithmetic average of $H(\pmb{A}(t))$ and $H(\pmb{V}(x,y,t))$), could also be used, but the idea is similar.

The biggest challenge for this method is how to choose features to represent audio and video, i.e. the components of $A(t)$ and $V(x,y,t)$. We want to choose features which can reflect the audio-visual synchrony well. Only in this way, the mutual information metric between them can reflect the audio-visual synchrony well. Generally, features for audio can be signal amplitude, pitch measurement etc, and features for video pixels can be intensity, RGB values etc. 
But we should aware that whatever features we choose with a hand-crafted way, there will always be mutual information loss.
This can be showed as: 

$$
\begin{aligned}
I(A_1, A_2, ..., A_n ; V_1, V_2, ..., V_m) &= I(A_1; V_1, V_2, ..., V_m) + I(A_2, ..., A_n ; V_1, V_2, ..., V_m | A_1) \\
& \geq  I(A_1; V_1, V_2, ..., V_m) \\
& \geq  I(A_1; V_1),
\end{aligned}
$$

where $A_1, V_1$ can be any single or set of features without losing generality.
However, in order to make this method practical for computing, the most simple choice is: the intensity of video at each pixel, and the audio amplitude. Such choice has been proved to be effective by [1], which is evaluated in the next section.


### 2.2 Implementation and Evaluation

I've chosen a 10s TV news video for analyzing. During the first half of the video, the man reads the news, and woman keeps in silence, vice versa for the second half.

#### Visual and Auditory Intensity Changes over Time

<p align="center">
  <img style="width:90%" src="/assets/images/audio_visual_201905/intensity_changes.png">
  <center>Figure 2</center>
</p>

The first problem is: Are features we have chosen correlated to each other?
Figure 2 gives the intuition, which shows how visual and audio intensity change over time for 2 example pixels - the sound source pixel $(x_1, y_1)$ and a uncorrelated pixel $(x_2, y_2)$.
The black line is the intensity of the audio signal, and the blue line shows the intensity changes for the pixel $(x_1, y_1)$ around the sound source, the speaker’s mouth.
We see that the intensity change of the sound pixel shows obvious synchrony with the audio signal.
By computing the mutual information, we obtain $I(\pmb{a}(t), V(x_1,y_1,t)) = 0.2329$ and $I(\pmb{a}(t), V(x_1,y_1,t)) = 0.078$.
This verifies that it is reasonable to use pixel intensity and audio amplitude to study the synchrony between audio and video.

#### Sound Localization for Different Periods

To identify which parts of the video are more related to the audio, we just need to compute the mutual information measurement for all pixels, and then visualize it as an image.
For different period of time, the parts with high response are different, which could be used to localize the speaker at some point.

<p align="center">
  <img style="width:90%" src="/assets/images/audio_visual_201905/sound_localization.png">
  <center>Figure 3</center>
</p>

Figure 3 shows the results by testing with the sample video.
Here, I applied this method for 2 segments of the video. In the first segment, the right person is speaking, we see that the pixel around this person are highlighted (the middle in Figure 3). In the second segment, the left person is speaking, we see that this person is highlighted (the rightmost in Figure 3).
From Figure 3, we can conclude that:
* This method can roughly identify which region produces sound for this example video. Sound source pixels have larger response.
* Except for the pixels corresponding to the speaker's mouth, other pixels which are not around the sound source also have relatively large responses. For example, in the rightmost figure, pixels corresponding to the woman's clothes also have large responses.

#### Does the method actually find the dynamic regions (actions), instead of the sound source?

From the above experiment, you may say that what the method does is to find the dynamic regions in the video, which correspond to actions, and these regions are accidentally the source source regions. 
Therefore, I computed the pixel variances over time to find the dynamic regions of the video, and then compared it with the image obtained by the audio-visual mutual information method.

<p align="center">
  <img style="width:90%" src="/assets/images/audio_visual_201905/dynamic_vs_mi.png">
  <center>Figure 4</center>
</p>

From the Figure 4, we can see that these 2 results are similar to some extent, but they also have differences. For example, the variance image has many noisy regions, but this problem doesn't exist for the mutual-information image. The sound source region is more obvious in the mutual-information image.

#### What will happen by using unmatched audio?

I also tested this method with an unmatched segment of audio.
Using the audio file of a popular song, which has no correlation with the video content, what will happen if I repeat the experiment? Figure 5 shows the result.

<p align="center">
  <img style="width:90%" src="/assets/images/audio_visual_201905/unmatched_audio.png">
  <center>Figure 5</center>
</p>

We see that the resulting image can still roughly find the sound source, although it is not as good as the result by using the matched audio.
A little surprising, right?
Maybe it can be explained by the fact that, it is also difficult for human to judge whether the an segment of audio is matched with a video, if we don’t know this is a news broadcasting video. 
Similar situations happen in our daily life. For example, it is hard for us to identify whether singers are singing by themselves, or they just follow the recorded song and pretend to sing.

### 2.3 Discussion

The experiments showed that the evaluated mutual information method has limited ability for sound source localization.
The reasons might be: 
1. The audio-visual correlation is very complex. Using audio and video intensity for representation might be over-simplified. The correlation might be more obvious for other features, such as the audio pitch and how the mouth moves.
2. The mutual information might not be well estimated. For a short period time, the audio signal might change a lot, but the number of sampled video frames is very limited, which results in inaccurate mutual information estimation.

Although the results in [1] are better than mine, I think this method will not be very effective in real situations. The results in [3,4] based on deep networks seems to be more promising. 


## 3. Neural Network Method

Another family of methods for audio-visual research use deep neural networks.
In recent years, deep learning has showed amazingly powerful capability to solve challenges in computer vision, speech recognition etc, due to their strength to learn good representations for visual and audio data.
Researchers are thinking about how capable deep neural networks are for challenges with multiple modalities. Audio-visual research is a typical example.
We know that the weakness for mutual information method is having used over-simplified features.
If neural networks can learn sophisticated features, such that these representations are highly correlated, they are supposed to have better performance for the sound source localization task.

### 3.1 Method

Owens et al. [1] have trained a multimodal neural network to predict whether a video and its audio are temporally aligned. The framework is showed in Figure 6, which takes 2 streams of inputs, i.e., the audio and video. After the network is well trained, attention mechanism is applied, to find out which regions contribute more to the prediction result. And these regions are considered as sound source. This is reasonable, since for those regions which are not sound source, there are no difference whether the audio is aligned or not aligned because they are not correlated. Therefore, regions which make the difference for the alignment task correspond to the sound source.

<p align="center">
  <img style="width:80%" src="/assets/images/audio_visual_201905/multsensory_framework2.png">
  <center>Figure 6</center>
</p>

Yapeng et al. [4] trained neural network for different task, i.e., to predict a sequence of event labels for videos. The neural network takes aligned audio and video pair as inputs, and output a event label (e.g. speaking, ) for each 1-second video segment. They also implemented the attention mechanism to the network. An interesting observation is that the visual attention tends to focus on sounding regions. The reason might be: the visual attention is highly-correlated with the output event labels, and these events are usually the cause of sound.
This work use pre-trained model to extract features for videos and audios, so it is not end-to-end. And it totally includes 28 event labels. For videos with uncovered events, using this to locate sound source will not work well. Owens's [3] method will be more general for predicting, which will be evaluated next.

### 3.2 Evaluation

Using Owens' [3] published implementation and inputing the network with the same video as before, we have the results as Figure 7 shows. We see that this method can locate the true speakers. Compared with the results in Figure 3, regions far from the sound source are not misclassified anymore, and with no noises like before. However, the located regions are too large, and it would be better they shrink to only cover the speaker's mouths.

<p align="center">
  <img style="width:90%" src="/assets/images/audio_visual_201905/Owens_res.png">
  <center>Figure 7: Result of neural network method [3]</center>
</p>

The researchers also tested the model responds to videos that do not contain speech, and how the model's attention varies with motion over large dataset. All these results shows that it is effective to attend to sound-making objects rather than actions (The results are shown in [3]).
From the results and discussion, we can conclude that neural network method is more effective to locate the real sound source.

## 4. Summary

This work studied how to locate sound source in videos using classic mutual information method and modern neural network method. The mutual information method is effective to measure the synchrony between video and audio to some extent, but not accurate enough to locate source, since the synchrony is too complex. Neural network method is more capable to capture the correspondence between modalities, thanks to deep network's ability to learn complex features to represent audios and videos, which contain much more mutual information. 

The ultimate goal is to inspect the correlation between videos and audio. We ask how much is the mutual information between video and audio, so that we can solve challenges like cross-modality localization, alignment, retrieval and translation. Also, we should ask how much is the joint information of video and audio. Then, we will know how much supplementary information we gain by knowing data from the other modality. If the supplementary information is large, by fusing the information between video and audio, it would be helpful for many video-understanding tasks, such as video classification, action recognition etc.

Although neural network method usually outperform classic hand-crafted methods for many tasks, we still need the classic information theory to guide us, which may help design neural networks or explain why neural network work to solve these problems.

## References

1. Hershey, John R., and Javier R. Movellan. "Audio vision: Using audio-visual synchrony to locate sounds." Advances in neural information processing systems. 2000.
2. Fisher III, John W., et al. "Learning joint statistical models for audio-visual fusion and segregation." Advances in neural information processing systems. 2001.
3. Owens Andrew, and Alexei A. Efros. "Audio-visual scene analysis with self-supervised multisensory features." Proceedings of the European Conference on Computer Vision (ECCV). 2018.
4. Tian, Yapeng, et al. "Audio-visual event localization in unconstrained videos." Proceedings of the European Conference on Computer Vision (ECCV). 2018.
