---
layout: default
title:  "Calculating map coordinates when you only know the distance"
date:   2016-03-22 17:50:00
categories: main
unpublished: True
---

<script src='https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML'></script>

# Calculating map coordinates when you only know the distance



Imagine that you have list of cities and their coordinates on the map. Calculating the distance between each city is easily done by taking the euclidian distance between each and every city coordinate. But what if you only know the distances and not the coordinates? The solution to that problem is not as trivial as its inverse problem. In fact just get a pen and paper and take a moment to think about how to write such an algorithm!

<!--
| City          | Coordinate       |
| ------------- |:----------------:|
| Brussels	    | 50°51'N	04°21'E  |
| London        | 51°36'N	00°05'W  |  
| Paris         | 48°50'N	02°20'E  |
| Copenhagen    | 55°41'N	12°34'E  |
-->


<center><img src="http://i.stack.imgur.com/EUf6U.png" class="inline"/></center>
What if I told you that a solution could be found in ... BOOM! ... just 3 lines of python?!

Here it is:

```python
M = (D[:,0]**2 + D[0,:,np.newaxis]**2 -D**2)/2
U, S, V = svd(M)
X = U*np.sqrt(S)
```

Not convinced? Okay lets break it down.

First of all, the variables here $$M$$, $$D$$, $$U$$, $$S$$, $$V$$ and $$X$$ are all matrices not just floating point numbers. Representing everything as matrices lets us write compact code and allows us to use some handy tools from the linear algebra package in numpy.

$$D$$ here is our distance matrix, where every element $$D_{ij}$$ represents the distance between coordinate $$i$$ and $$j$$. If we have $$n$$ coordinates the size of $$D$$ is $$n \times n$$.

To start our calculation we define another matrix

$$M(i,j) = \frac{D_{1j}^2 + D_{i1}^2 - D_{ij}^2}{2},$$

which has the same size as $$D$$. We then apply singular value decomposition (SVD) of matrix $$M$$. This factorizes the matrix into two matrices $$U$$ and $$V$$ and a $$n \times n$$ vector $$S$$ that contains the singular values of $$M$$. Since our coordinates are in two dimension only the two values of $$S$$ will be non-zero.

You can think of $$S$$ as describing how skewed your coordinate system is. For example if your coordinates represented cities in Chile the first value of $$S$$ will be much larger than the second one since the country is long and narrow, where as if they represented cities in Germany the singular values would be approximately the same size.

In order to get the coordinates we simply multiply $$U$$ with the square root of $$S$$. $$X$$ is now also an $$n \times n$$ matrix, but only the first two columns are non-zero (since our coordinates were in two dimensions). Each row in $$X$$ represents a coordinate. Note that X is not necessarily the same coordinates as the one we started with since we do not know the rotation of the coordinate system. In other words we can find an infinite number of solutions to this problem just by rotating our coordinate system. To test our algorithm we can just recompute the distance matrix and compare it to what we started with.

The full code is here:
``` python
def find_coords(D):
    """
        Finds coordinates from distance matrix D
        return:
            X: coordinate matrix where rows are samples, cols are dimensions
    """
    # Define M[i,j] = (D[0,j]^2 + D[i,0]^2 - D[i,j]^2)/2
    M = (D[:,0]**2 + D[0,:,np.newaxis]**2 -D**2)/2
    # Singlualar value decomposition of M
    U, S, V = svd(M)
    # Compute coordinates
    X = U*np.sqrt(S)

    return X

# Test algorithm by distance matrix using some random coordinates
coords = np.matrix('1 2; 3 4; 5 2')

num_coords = 30 # number of coordinates
num_dims = 10 # number of dimension

coords = np.random.rand(num_coords,num_dims)
D = squareform(pdist(coords))
print("Distance matrix:")
print(D)

X = find_coords(D)
print("Coordinates (first 2-dims):")
print(X[:,:2])

# Test our algorithm by computing back the distance matrix
D2 = squareform(pdist(X))
error = np.sum(D2-D)
print("Differnece beween original and recomputed distance matrices (should be zero):")
print(error)

# Plot coordinates
plt.figure()
plt.scatter(X[:,0],X[:,1])
plt.show()
```


Until now we only the 2D case but, the algorithm also works for higher dimensions. The only difference is that $$S$$ will have more than two values which are non-zero.

*As a bonus:* If you want to visualize your high dimensional map you can just select the two first dimensions (columns) of $$X$$ to plot. This is similary to principal component analysis (PCA) which is often used in machine learning for reducing the dimensions of high dimensional data.

For a more formal explanation how and why this algorithm works check out Legendre17's post [here](http://math.stackexchange.com/questions/156161/finding-the-coordinates-of-points-from-distance-matrix).
