---
title: On Spectral Clustering
---


Clustering problems a very common and classical problem in many fields.
This post will introuduce a set of amazingly good algorithms to solve clustering problem.

- [1. Introduction](#1-introduction)
- [2. Problem statement](#2-problem-statement)
- [3. Methods](#3-methods)
- [4. Simulation Results and Discussion](#4-simulation-results-and-discussion)
  - [Compare with Matlab *kmeans* function](#compare-with-matlab-kmeans-function)
  - [Constructing similarity graph}](#constructing-similarity-graph)
  - [Choosing numbers of clusters $k$ automatically](#choosing-numbers-of-clusters-k-automatically)
- [5. Potential applications](#5-potential-applications)
- [References](#references)

## 1. Introduction

Clustering problems a very common and classical problem in many fields.
*K-means* is the simplest cluster algorithm, and it is commonly-used but has many limitations.
In this report, I did a survey on spectral clustering algorithms.
And by experimental testing, it has much better performance than *k-means*


The main reference for this report is [2].
And I have also made some analysis and improvements for the algorithm by my own, which includes:


- A explanation for why clustering on spectral can cluster the original data points (In Section 2).
- A strategy to adjust $\sigma$ (Gaussian similarity function parameter) automatically (In section 4.2) for different data points scale.
- A strategy to select number of clusters $k$ according to eigengaps (In section 4.3).


The improved clustering algorithm can run fully automatically without tuning any parameters. I've tested it over several dataset, and it has fine acceptable results.
This algorithm is the best clustering algorithm I've found till now.

## 2. Problem statement
Given a set of data points $x_1, x_2 ... x_n$, we want to partition them into $k$ clusters, such that for data points in the same cluster they are similar, and for data points in different clusters they are not similarity to each other.

From the perspective of graph theory, the clustering problems can be reformulated into a graph cut problem.
If we regard each data points as a vertex, and the similarity measurement as the edge weight between vertices, then we have an undirected graph $G = (V, E)$,
and the clustering problem is to find a partition for graph $G$.

For computation, a graph is usually represented with a affinity matrix $W$, for which the row or column indices represent vertices, and $w_{ij}=w_{ji}$ represents the similarity between vertex $i$ and vertex $j$.
There are many ways to measure the similarity between vertices [2].
The first idea is to make $w_{ij}=1$ if the distance between vertex $i$ and $j$ is less than a threshhold, otherwise $w_{ij}=0$.
The second idea is based on $k$-nearest neighborhood, for example $w_{ij}=1$ if vertex $j$ is among $k$ nearest vertices of $i$, otherwise $w_{ij}=0$.
Another idea is to compute the similarity using Gaussian function $w_{ij}=exp(-\frac{||x_i - x_j||^2}{2 \sigma^2} )$, which will construct a fully connected graph. (I've used the combination of the first and the second idea for implementation, which will described in Section 4.2)

We can define similarity sum $S$ as: for two groups of vertex set $A_1, A_2$ (can be the same set),
$$
S = \sum_{i \in A_1, j \in A_2} w_{ij}
$$
For a well-partitioned $G$, the similarity sum for the same group of vertices should be small, and the similarity sum for different groups should be large.

The above problem is a multi-objective and discrete optimization problem, which has been proved to NP-hard.
*K-means} is a commonly-used and simple algorithm to solve this problem, but it has obvious limitations. It can be easily stuck in some local minimum, and there is no implication for how to choose $k$ automatically.
Moreover, for complex data points (e.g. circle-in-circle data points), usually it cannot find good solution.

## 3. Methods

This section will describe the mathematical analysis and a group of algorithms based on spectral clustering.
Firstly, I give some important definitions here.

**Definition 1 (Indicator vector):** For graph $G=(V, E)$ and $A \subset V$, we can define a indicator vector of $A$ as $1_A \in \Re^n$, for which $[1_A]_i = 1$ if vertex $i \in A$, otherwise $[1_A]_i = 0$

**Definition 2 (Degree matrix):** The degree matrix $D$ is defined as

$$D = \begin{bmatrix}
    d_1 & \\ & d_2 \\ && ...\\ &&&d_n
\end{bmatrix},$$

where $d_i = \sum_{j=1}^{n} w_{ij}$ (row sum of $W$).

**Definition 3 (Graph Laplacian matrix):** The graph Laplacian matrix is defined as

$$L = D - W.$$

For any vector $f \in \Re^n$, we can show that

$$f^T L f = \frac{1}{2} \sum_{i,j=1}^{n} w_{ij} (f_i - f_j)^2$$

By the definition of $L$ and equation (1), we can conclude some important properties of the graph Laplacian matrix $L$.

  *  **Property 1:** $L$ is symmetric and positive definite.
  *  **Property 2:** $L$ has $n$ non-negative, real-valued eigenvalues $0 = \lambda_1 \leq \lambda_2 \leq ... \leq \lambda_n$, and the smallest one is 0.
  *  **Property 3:** The multiplicity $k$ of the eigenvalue 0 of $L$ equals the number of connected components in the graph $A_1, A_2, ..., A_k$, and the eigenspace of eigenvalue 0 is spanned by the indicator vectors $1_{A_1}, ..., 1_{A_k}$ of those components.


Property 3 is an amazingly important property, which implies that by analyzing the eigenvalue and eigenspace of $L$, we can find the partition of $L$, which is also the partition for $W$ and the original graph.

Some researcher made normalization for $L$ to achieve better performance.
Shi et.al [3] defined the normalized $L$ as $L_{rw} = D^{-1} L = I - D^{-1}W$, and Ng et.al normalized $L$ as [1] $L_{sym} = D^{-\frac{1}{2}} L D^{-\frac{1}{2}} = I - D^{-\frac{1}{2}} W D^{-\frac{1}{2}}$.
Properties of $L_rw$ and $L_sym$ can be easily derived by the properties of $L$, and some of them are closely related.
Based on different graph Laplacian, new algorithms are developed.
In my experiment, I choose the algorithm based on $L_rw$ as suggested by [2].

The algorithm can be described as follows:

 *  Input: datapoints, number of clusters to construct.
 *  Construct the similarity matrix $W$ based on given data points.
 *  Compute $L$.
 *  Compute the first $k$ generalized eigenvectors $u_1, u_2...u_k$ of the generalized eigenproblem $Lu = \lambda D u$ (which is the same as $L_{rw} u = \lambda u$).
 *  For $i = 1,...,n$, let $y_i \in \Re^k$ be the vector corresponding to the $i$-th row of $U$.
 * Cluster the points $(y_i)_{i = 1,...,n} \in \Re^k$ with the *k-means} algotithm into clusters $C_1, ..., C_k$.
 * Output: Clusters $A_1, ..., A_k$ with
 $A_i = \{j | y_j \in C_i\}$.

By previous mathematical analysis, we can eaisly understand all the steps of the above algorithm except for step 6.
Why using *k-means* on rows of $U$ can cluster the original data points?
Here is my understanding.
Recall that the eigenspace of eigenvalue $0$ is spanned by $1_{A_1}, ..., 1_{A_k}$, therefore each column of $U$ is a linear combination of $1_{A_1}, ..., 1_{A_k}$.
Let $A = [1_{A_1} \: 1_{A_2} \:...\: 1_{A_k}]$, we have $[A]_{ij} \in \{0, 1\}$ and each row sum of $A$ is 1.
Then, we have

$$A \cdot C = U,$$

where $C \in \Re^k$ is the linear combination coefficients matrix.
Considering the rows of $U$, they are linear combination of rows of $C$.
Since each row of $A$ has only one element equal to 1 and other elements are zeros, each row of $U$ is actually one of the rows of $C$.
Therefore, rows of $C$ can be obtained by clustering rows of $U$.
And more importantly, if the $i$-th row of $U$ is the $k_i$-th row of $C$, we know that only $[A]_{i,k_i}=1$, which means that data points $i$ is partitioned to cluster $k_i$

## 4. Simulation Results and Discussion

In this section, I will talk about the implementation and experimental results of the spectral clustering algorithm described in the last section.
Section 4.1 will show the basic version for implementation, and I will compare the performance with that by using Matlab *kmeans} function.
Section 4.2 wil show a strategy I've found to automatically tune parameter $\sigma$ in order to clustering for dataset with different scale.
Section 4.3 will present a strategy I've found to choose number of clusterings $k$ automatically and show the results.

### Compare with Matlab *kmeans* function

The clustering algorithm is implemented in source code file *sc\_cluster.m*.
As I methioned before, *K-means} usually cannot cluster data points with complex shapes.
I've generated two complex data: one is like *two spirals*, and the other is like *cluster in cluster*.
I test these two set of data using my implementation and Matlab function *kmeans*, and Figure 1-4 shows the results.


<!-- Figures -->
***Figure 1 - kmeans:***

![1](/assets/images/spectral_clustering/km_cinc.png)

***Figure 2 - Spectral clustering:***

![2](/assets/images/spectral_clustering/sc_cinc.png)

***Figure 3 - kmeans:***

![3](/assets/images/spectral_clustering/km_2sp.png)

***Figure 4 - Spectral clustering:***

![4](/assets/images/spectral_clustering/sc_2sp.png)
<!-- \begin{figure}[!h]
    \begin{minipage}{0.45\linewidth}
        \centering
        \includegraphics[width=1\linewidth]{../res/km_cinc.png}
        \caption{*kmeans}}
    \end{minipage}
    \begin{minipage}{0.45\linewidth}
        \centering
        \includegraphics[width=1\linewidth]{../res/sc_cinc.png}
        \caption{*Spectral clustering}}
    \end{minipage}
    \begin{minipage}{0.45\linewidth}
        \centering
        \includegraphics[width=1\linewidth]{../res/km_2sp.png}
        \caption{*kmeans}}
    \end{minipage}
    \begin{minipage}{0.45\linewidth}
        \centering
        \includegraphics[width=1\linewidth]{../res/sc_2sp.png}
        \caption{*Spectral clustering}}
    \end{minipage}
\end{figure} -->

The results are obvious.
Spectral clustering algorithm has much better for complex data points.
I have also tested dataset that $kmeans$ can handle, and the result shows that the clustering result of spectral clustering algorithm have more stable solution.

### Constructing similarity graph}
There are many choices for constructing the similarity matrix as described in Section 2.
In my implementation, I've chosen Gaussian function

$$s(x_i, x_j) = exp{(-||x_i - x_j||^2 / (2 \sigma^2))}$$

to measure the similarity between data points.
After I compute the pairwise similarity and store in $n \times n$ matrix $W_{pairwise}$, I set $[W_{pairwise}]_{i,j} = 0$ if $[W_{pairwise}]_{i,j} < 80\% \: percentile \: value \: of [W_{pairwise}]$.
The reason behind this is that we consider two data points are not similar at all if the Gaussian similarity is less then 80\% percentile value of possible similarity.

For dataset with different scale, we have to tune $\sigma$, which is annoying.
Bad results will occur if we don't do this.
Can we find a strategy to adjust $\sigma$ automatically according to the scale of the data?
Here, I've use the strategy to achieve this.
The parameter $\sigma$ is adjusted as

$$\sigma = \rho \cdot std(W(:)).$$

Where $std(W(:))$ is the standard deviation of $W$ to measure the scale of $W$, and $\rho$ is positive number which is insensitive to the scale.
By experiments, set $\rho = 0.08$ can usually get good results.

Figure 5 shows the clustering result over the original data.
After scaling original data points 10 times larger, there is a much difference between with and without automatically adjusting $\sigma$.
With the above strategy, the algorithm can still perfectly clustering the data points, as Figure 6 shows.
The result in Figure 8 is really bad when not adjusting $\sigma$.

<!-- Figures -->
***Figure 5 - Original results ($std(W(:)) = 4.5442$):***

![5](/assets/images/spectral_clustering/sc_scale1.png)

***Figure 6 - Automatically adjusted $\sigma$ ($std(W(:)) = 45.442$):***

![6](/assets/images/spectral_clustering/sc_scale10.png)

***Figure 7 - Without adjusted $\sigma$ ($std(W(:)) = 45.442$:***

![7](/assets/images/spectral_clustering/sc_scale10_nosigmatune.png)

***Figure 8 - Automatically chosen $k$:***

![8](/assets/images/spectral_clustering/automatic_k.png)

<!--
\begin{figure}[!h]
    \begin{minipage}{0.45\linewidth}
        \centering
        \includegraphics[width=1\linewidth]{../res/sc_scale1.png}
        \caption{\tiny*Original results ($sta(W(:)) = 4.5442$)}}
    \end{minipage}
    \begin{minipage}{0.45\linewidth}
        \centering
        \includegraphics[width=1\linewidth]{../res/sc_scale10.png}
        \caption{\tiny Automatically adjusted $\sigma$ ($std(W(:)) = 45.442$)}
    \end{minipage} \\
    \begin{minipage}{0.45\linewidth}
        \centering
        \includegraphics[width=1\linewidth]{../res/sc_scale10_nosigmatune.png}
        \caption{\tiny Without adjusted $\sigma$ ($std(W(:)) = 45.442$)}
    \end{minipage}
    \begin{minipage}{0.45\linewidth}
        \centering
        \includegraphics[width=1\linewidth]{../res/automatic_k.png}
        \caption{\tiny Automatically chosen $k$}
    \end{minipage}
\end{figure} -->

### Choosing numbers of clusters $k$ automatically

As the mathematical analysis implies, the multiplicity of eigenvalue 0 is the number of clusters.
However, in practice, usually the similarity is fully connected if we use Gaussian function to construct the similarity matrix.
Therefore, we should consider several smallest eigenvalues to decide the cluster numbers $k$.
One of the way is two decide the number of $k$ by analyzing the eigengaps $|\lambda_{i+1} - \lambda_{i}|$.

I've used a strategy to automatically choose $k$ according to the eigengap.
Suppose we have $N$ data points, we can compute the average eigengap as $\frac{\lambda_{max} - \lambda_{min} }{N}$.
Then, from the smallest eigenvalue $\lambda_1=0$, we campare the eigengap $|\lambda_{i+1} - \lambda_{i}|$ with the average eigengap.
Until the first time $|\lambda_{i+1} - \lambda_{i}|$ is greater than the average eigengap, we stop the process and set $k=i$.

Figure 8 is the result when using the above strategy to choose $k$ automatically.
I've test such strategy over other dataset, and it can also have acceptable results.
The above strategy works well for some dataset, although it is not perfect.
It provides a option for us if we want the clustering process runs fully automatically.

## 5. Potential applications

This is the best clustering algorithm I've used till now.
I am doing some research on video classification. One basic idea is to do classification for each frame of the video, based on which we can categorize the video, but it is too computationally expensive.
Considering the fact that most videos are consist of several scenes, which means many consecutive image frames of the video are taken under the similar situation (e.g. scenary video), we can only process one image for each scene.
The problem is how to cluster all the image frames into scenes, and the key is to use a good clustering algorithm.

The algorithm introduced in this report is very promising to solve the video scene clustering problem.
There are many mature methods to measure the similarity between images (e.g. using ImageNet model to extract features), so it is easy to build the similarity graph.
The proposed algorithm can detect a rough $k$, which is enough because usually for video classification we don't need a very precise number of clusters.
Therefore, the whole process can run totally automatically.

## References

1. Ng, Andrew Y., Michael I. Jordan, and Yair Weiss. "On spectral clustering: Analysis and an algorithm." Advances in neural information processing systems. 2002.
2. Von Luxburg, Ulrike. "A tutorial on spectral clustering." Statistics and computing 17.4 (2007): 395-416.
3. Shi, Jianbo, and Jitendra Malik. "Normalized cuts and image segmentation." IEEE Transactions on pattern analysis and machine intelligence 22.8 (2000): 888-905.
