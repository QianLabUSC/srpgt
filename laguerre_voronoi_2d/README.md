# 2d Laguerre-Voronoi diagrams

This code sample demonstrates how to compute a 
[Laguerre-Voronoi diagram](https://en.wikipedia.org/wiki/Power_diagram) 
(also known as *power diagram*) in 2d. 

![thumbnail](https://i.imgur.com/JJZ8Ck7.png)

Power diagrams have a wonderful property : they decompose the union of 
(overlapping) circles into clipped circles that don't overlap. The cells have 
a simple geometry, just straight lines.

It works as following

* The circles centers are considered as points associated with a positive weight : the circle's radius
* Transform the points (called here *lifting* the points) from 2d to 3d.
* Compute the convex hull of the *lifted* points
* The lower enveloppe of the convex hull gives us the *power triangulation*
* The *power diagram* is the dual of the *power triangulation*.

The complexity of this algorithm is O(n log(n)), with most of the heavy lifting
done by the convex hull routine.

## Prerequisites

To run this sample, you will need

* Python 2.7 or above
* [Numpy](http://www.numpy.org)
* [Scipy](http://www.scipy.org)
* [Matplotlib](https://matplotlib.org), 2.0 or above

