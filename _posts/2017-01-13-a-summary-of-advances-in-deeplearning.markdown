---
layout: default
title:  "A brief summary of advancement in deep learning and AI when going into 2017"
date:   2017-01-21 10:50:00
categories: main
---
# A brief summary of advancement in deep learning and AI when going into 2017

In the last few years we have seen remarkable progress in machine learning applications such as image classification and segmentation, face detection, speech recognition, text-to-speech as well as in reinforcement learning. The year 2016 was filled with major advancements in several AI research areas. In this post I will review some of the great advancements in AI research made in the past year and try to make some speculation of what we might see in 2017.

## Real time artistic style transfer

Neural style transfer is a reasonably new subfield of machine learning research where the goal is to turn a photo into a painting by imitating the style of another painting. It was in mid 2015 [Gatys et al.](https://arxiv.org/abs/1508.06576) showed how to transform any photo into a painting using neural networks. While producing novel and impressive result their method were very slow since a neural network had to be trained and optimized every time a style is transfered onto a photo.

In 2016 we saw a lot of new improvements in style transfer algorithms in both visual quality and transfer speed. For example [Johnson et al.](https://arxiv.org/abs/1603.08155) showed that by training feed forward neural network to imitate the style transfer of Gatys algorithm, a style of a painting can be transfer to any image in real time. This enables creation of never before seen "animated paintings".

Later in the year [Dumoulin et al.](https://arxiv.org/abs/1610.07629) showed that it is in fact possible to train a single model to represent multiple styles. Another interesting progress comes from the team at nucl.ai who showed an exiting application called [Neural Doodle](https://nucl.ai/blog/neural-doodles/) where MS Paint doodles can be transformed into amazing artworks with little effort.

<!--

-->
<center><img src="/images/neural_doodle.gif" class="inline"/></center>


## Generating photo realistic images

Another reasonably new development in machine learning are algorithms that are able to generate novel random images that look completely photo realistic. While the early results usually were distorted and blurry recently proposed Generative Advarisal Networks (GANs) have been able to generate convincing photo realistic images. The main idea is to instead of training a neural network with a constant loss function such as cross-entropy, there is an adaptive loss function modeled by an other neural network. Ian Goodfellow provides an excellent technical introduction to GANs in his [NIPS 2016 Tutorial](https://arxiv.org/abs/1701.00160).

While GANs were initially introduced in 2014, it was in the past year we throughly started to see how magical they can be. In fact Yann LeCun, Director of AI Research at Facebook, even called GANs ["the most interesting idea in machine learning in the last 10 years"](https://www.quora.com/What-are-some-recent-and-potentially-upcoming-breakthroughs-in-deep-learning). These faces below are not real images. They are random generated output images of a GAN as shown by [Radfold et al](https://arxiv.org/abs/1511.06434). Eventhough the model sometimes failed to produce realistic images, it is remarkable to see how the network learned how to draw shadows, highlights and even makeup realistically!

<center><img src="/images/DCGAN_faces.png" class="inline"/></center>

## Taming generative networks

While generating images of [random photos of faces or bedrooms](https://github.com/Newmu/dcgan_code) is theoretically interesting, it becomes more practical when the user can interact with the image generation using recently introduced Conditional GANs. The generated image can be conditioned and a user variable such as text or a class, or even another image to allow the user to influence the output of the generated image.

<center><img src="/images/image-to-image_handbags.png" class="inline"/></center>

[Zhang et al.](https://arxiv.org/abs/1612.03242) shows impressive results of generating high quality images of flowers and birds, which are condition on a text description. As late as December 2016 [Shrivastava et al.](https://arxiv.org/abs/1612.07828) showed another exciting application of GANs where they generated new realistic training images by refining the details of CGI images. This is also huge since in many domains collecting good labeled data for supervised learning is an expensive process. The results of these publications are almost science fictional, and yet are they all real publications from 2016.

 [Isola et al.](https://arxiv.org/abs/1611.07004) are demonstrating a very impressive image-to-image translation framework where an image such as a hand sketch of a handbag can be transformed into a photo realistic image. [Zuh et al.](https://arxiv.org/abs/1609.03552v2) used GANs to create a tool that lets the user interactively manipulate images while remaining the realistic look.

<center><img src="/images/StackGAN_birds.png" class="inline"/></center>

## Improved text to speech and AI beating human players

While the progress of generative networks are the one of the most exciting advancements in AI research in 2016, its far from the only AI brake through. For example the researchers at Deepmind showed how to generate convincing speech using [dilated convolutional neural networks](https://deepmind.com/blog/wavenet-generative-model-raw-audio/). Make sure to listen to the samples.

There has also recently been major progress the field of deep reinforcement learning. In the beginning of the year an AI player was for the first time [beating a professional human player at the game of GO](http://www.nature.com/nature/journal/v529/n7587/full/nature16961.html). Later in the year [Lample et.al](https://arxiv.org/abs/1609.05521) among others showed how agents can learn how to play beat human players at the first person shooter game Doom. One of the most exciting development in deep reinforcement learning lately is a new algorithm called Asynchronous Advantage Actor-Critic (A3C) by [Mnih et al.](https://arxiv.org/abs/1602.01783) which surpassed previous state-of-the-art on playing Atari games in half the training time, using just a single multicore CPU.

<center><iframe width="560" height="315" src="https://www.youtube.com/embed/oo0TraGu6QY" frameborder="0" allowfullscreen></iframe></center>

## What can we expect in 2017

In 2017 we will probably see more improvements in the visual quality of transfered painting and hopefully also more interactive applications where the user has a better control of the transfered style.

We have seen that GANs, even though tricky to train, are very powerful. I believe that in 2017 we will see further improvements of GANs. Likely we will see new applications of GANs, not only in the image generation domain, but also adaption of GANs for speech and music generation and perhaps even for 3d content generation. Together with style transfer networks, GANs are now laying up the basis for next generation of artistic tools.

Secondly we will continue to see advancement in reinforcement learning and self learning agents. As [DeepMind recently announced their cooperation with Blizzard](https://deepmind.com/blog/deepmind-and-blizzard-release-starcraft-ii-ai-research-environment/) we will start to see agents that are able to learn to manouver in larger and more dynamic environments such as StarCraft 2. As algorithms get better the agents will both learn faster and perform better at complex tasks.

Finally I also think we will in 2017 see more research in the field of [meta learning](https://en.wikipedia.org/wiki/Meta_learning_(computer_science)), where machine learning algorithms are improved using machine learning. The research article [Learning to learn by gradient decent by gradient decent](https://arxiv.org/abs/1606.04474) not only has a clever title but also shows some interesting results how neural networks can learn faster using learned optimization algorithms.
