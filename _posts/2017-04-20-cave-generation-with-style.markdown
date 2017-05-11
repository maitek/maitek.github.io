---
layout: default
title:  "Cave generation with style: Create awesome levels using machine learning"
date:   2017-04-22 17:50:00
categories: main
---

<script src='https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML'></script>

# Cave generation with style: Create awesome levels using machine learning

Developing a computer game is both challenging and time consuming. As a hobby game developer I've been working on multiple small game projects myself. It can take several months or even years to complete a game with large and detailed levels. Using modern game engines such as [Unity](https://unity3d.com/) or [Unreal Engine](https://www.unrealengine.com/) speeds up development by providing tools and frameworks for developing game mechanics such as physics and and player controls, but you will still need to design the environment, characters and other content for your game. This tedious design work is also just taking longer and longer as the expectation of graphics and level of details in games rises.

This is why the current trend is for game studios to more and more use generated procedural content in their games. This way the artists and level designers do not need to design every detail for every terrain, forest, and city by themselves. The boring work such as placement of trees and bushes is done by a computer algorithm and the designer can instead focus their time on making the game fun.

In this post I will show an example of how neural networks can be used to generate art for a simple 2D game. Specifically I will show how to create random procedurally generated caves for your game and then stylize them in the style of your game them using machine learning. The output of the program is an color texture that can be imported as a level for your game. If you want to directly jump into the code everything is available on my [GitHub](https://github.com/maitek/cave-generation-with-style).

## Procedural is not random

Procedural level design is when a level or part of the level is generated according to some set of rules. Procedural content is often called randomly generated content, but procedural is not as random as you might think. In fact procedural can be completely deterministic and not random at all.

**As an example:** if you are creating a forest, and you are placing trees by randomly setting their x- and y-coordinate it is very likely that you end up with trees that overlap each other. Especially if you have a large dense forest with many trees. To prevent this you can instead create the forest *procedurally* where every new tree is placed using some constraints. We can define a rules to constrain the tree generation so that it can only be placed at the edge of the forest with a given distance from other trees. This way the forest grows procedurally with the number of added trees. You can then start define more rules in order to give more variation to the forest. You can for example change the size and type of the tree depending on the slope, terrain elevation and type of soil. The more rules you define the more structure will forest have and the more real and "designed" will it look, without having to place every single tree manually.

<center><img src="/images/procedural_vs_random.png" class="inline"/></center>

## Generating random caves

In this tutorial we will generate procedural caves using some very simple pixel rules called [cellular automata](http://www.roguebasin.com/index.php?title=Cellular_Automata_Method_for_Generating_Random_Cave-Like_Levels). The algorithm starts out with a random binary grid, where every cell has the value 1 or 0. A pixel with value 1 defines the walls and 0 defines the air inside the cave. The algorithm then applies what is commonly known as the 4-5 rule to every cell in the grid. The 4-5 rule says that if cell has 5 or more pixels with value 1 in its 3x3 neighborhood, the cell should also be set to 1. If it has 4 or less it should be set to 0. This is then repeated a few times until a cave like structure appears.

The code for generating a cave can simply be written in Python like this:

``` python

def generate_map(map_size=(64,64),num_iterations=10, ksize = 3):
    """
        Generate a random map using cellular automata
    """
    # generate random binary image
    random_map = np.random.rand(map_size[0],map_size[1])
    random_map[random_map < 0.5] = 0
    random_map[random_map > 0.5] = 1

    # cellular automata
    for k in range(num_iterations):
        kernel = np.ones((ksize,ksize))
        conv_map = convolve2d(random_map,kernel,mode='same')
        threshold = (ksize*ksize)/2
        random_map[conv_map <= threshold] = 0 # 4 rule
        random_map[conv_map > threshold] = 1 # 5 rule
    return random_map
```

The 4-5 rule is here implemented as a 2D convolution with $$k \times k$$ sized kernel. The 2D convolution is an operation simply calculating a weighted sum of all pixels with the kernel sized neighborhood of the pixel. In this case, since the image is binary and the weights are just one the images, the convolution is just calculating the number set pixels (pixel with value 1) around a given pixel. Each pixel then set or unset depending on the corresponding value in the convoluted map.

The convolution operation not only allows for compact code but also gives us additional flexibility. We can for example increase the kernel size to generalize the 4-5 rule to larger neighborhood. This is useful for maps of larger resolution. So instead of thresholding at 4 and 5 calculate the threshold to be half of the number of pixels in the kernel. We can now get different structure simply by changing the *ksize* parameter. The method returns a binary image where ones defines the walls an zeros defines the caves.

<!-- Image of different maps with different kernel sizes here! -->
<center><img src="/images/cave_generation.png" class="inline"/></center>

## Neural style transfer

Now we can easily generate hundreds of awesome caves levels for our game on the fly. The only thing we need to do now is to tell our artist to import each of the levels into photoshop, and to make them look nice for the game. A few months later we should have a pretty awesome looking game. But what if we then decide to change the art style of our game. Our artist would have to go back and redo all of the 1000 levels. What if we could transform our caves into the art style of the game in similar way as we generated our levels. This is where neural style transfer come into play.

Neural style transfer is a recent technique in machine learning where a neural network is trained to transfer a art style from one image to another. The style of an image is decouple from its content by using a deep neural network trained for object recognition. This can be done since the deeper layers of the neural network (closer to the output) are more activated for high level features in the images such as objects, while the shallower layers (closer to the input) are activated only for the image style such as brush strokes. This is the same technique which is used in some popular photo apps such as [Prisma](https://prisma-ai.com/).

<!-- style transfer image -->
<center><img src="/images/style-transfer.png" class="inline"/></center>

Since what we want is to transform our cave into a certain art style this technique sounds actually promising. In order to apply a specific art style we would have to train a neural network on an image of that style. Training a style neural network from scratch can take hours of processing even on a high-end GPU, so in this tutorial we will use pre-trained style networks instead.

In the end of 2016 Google published a research paper where they show that multiple styles can be embedded in a single neural network. Googles model has 32 different styles, so using this we can choose the styles that suits us best, and we do not have to train a neural network ourselves. We can even generate styles which are linear combinations of styles in the model. The code for Googles neural style transfer algorithm is completely [open source](https://github.com/tensorflow/magenta/tree/master/magenta/models/image_stylization) and is part of the [Google Magenta project](https://magenta.tensorflow.org/welcome-to-magenta). We will use this in our project.

We define a new method **stylize_image** which takes our generated cave image and style index (0-31) as input and returns the stylized image. The method also needs the saved checkpoint file of the trained neural network model. We will just use the [*"Varied"*](https://github.com/tensorflow/magenta/tree/master/magenta/models/image_stylization#stylizing-an-image) model checkpoint from Google Magenta team, which can transform into 32 different styles.

Magenta uses Googles deep learning framework Tensorflow in order to define the neural network to transform the image into a style. The image has first to be converted to a 4D floating point tensor. We then instantiate the Magenta neural network model as tensor flow graph, load the checkpoint model. The style transform is the performed using the **eval()** method which returns the stylized image.

``` python

def stylize_image(input_image, which_styles, checkpoint):

    # convert image to tensor
    tensor = input_image[...,np.newaxis].astype(np.float32)
    tensor = np.dstack((tensor,tensor,tensor))
    tensor = tensor[np.newaxis,...]

    """Stylizes an image into a set of styles and writes them to disk."""
    with tf.Graph().as_default(), tf.Session() as sess:
        style_network = model.transform(
            tf.concat([tensor for _ in range(len(which_styles))], 0),
            normalizer_params={
                'labels': tf.constant(which_styles),
                'num_categories': 32,
                'center': True,
                'scale': True})
        load_checkpoint(sess, checkpoint)

        output_image = style_network.eval()
        return output_image[0,:,:,:]


```

## Generate levels with style

Now we can generate levels with different styles by simply chaining the style transfer with the cave generation. Style transfer will work on binary images but since the lack texture the end result might look a bit dull. We could load a texture before style generation, but since we already can generate style we can also use it for generating a texture. In order to generate a random texture we can simply create a random noise image and pass it through the style transfer with a random style.

It is also important to have a good contrast between the foreground and the background. We therefore make the foreground darker by multiplying it with a small constant. We also rotate the texture by 90 degrees to prevent continues texture patterns between the foreground and the background. We now pass our cave another time through the style transfer in order to give its final style.

``` python

# Generate caves with all styles
for i in range(32):

    print("Generating map in style {}".format(i))

    # generate map and resize it to output size
    random_map = generate_map(map_size,num_iterations,ksize)
    random_map = imresize(random_map,output_size, interp="nearest")
    random_map = random_map.astype(np.float32)/255
    mask = np.dstack((random_map,random_map,random_map)).astype(np.float32)

    # generate a random texture to give the level some more detail
    noise_map = np.random.rand(random_map.shape[0],random_map.shape[1])
    bg_style_map = stylize_image(noise_map,[np.random.randint(32)],model_checkpoint)
    bg_map =  noise_map * mask
    fg_map = np.swapaxes(bg_style_map,0,1) * (np.ones_like(mask)-mask)
    combined_map = bg_map+fg_map*0.1

    # apply style transfer to get stylized levels
    test_map_style = stylize_image(np.mean(combined_map,axis=2),[i],model_checkpoint)
    imsave('map_{}.png'.format(i), test_map_style)

```

Generating an $$1024 \times 1024$$ image on my laptop using without GPU takes about 50 seconds. Here are some art styles generated by this algorithm:

|<img src="/images/caves/1.png" class="inline"/>|<img src="/images/caves/2.png" class="inline"/>|
|<img src="/images/caves/3.png" class="inline"/>|<img src="/images/caves/4.png" class="inline"/>|

## Summary

Deep neural networks can become powerful tools for procedural art generation in the future. The end results here are not perfect but can vastly speed up the design process. We could further add details such as stones and vegetation to the level before applying the final style transfer to get even more interesting results. I don't see neural networks as a replacement of the artist, but rather as a tool that can make artist more productive. If you want to try generating some caves yourself you can get the full source code at my [GitHub](https://github.com/maitek/cave-generation-with-style).
