gpusmoldyn
==========

GPU Smoldyn: Smoldyn algorithm ported to the GPU using CUDA 2.*

This is the code I wrote during my PhD, when I was doing research on high performance method for simulations, in particular parallel execution of biochemical "programs".

Biochemical simulations use different level of abstractions to cope with the incredible amount of computations needed to recreate inside a "virtual world" the observable behaviour. At mid-level, there is a sweet spot of methods that are sufficiently "concrete", precise, to reproduce biochemical behaviour accurately, and present themselves as perfect candidates for a stream processing kind of parallelism. 

One of this algorithms is Steven Andrew's [Smoldyn](http://www.smoldyn.org/); I wrote a PoC Smoldyn simulator on the GPU for the [HiBi 2010 conference](http://www.google.it/url?sa=t&rct=j&q=&esrc=s&source=web&cd=7&cad=rja&ved=0CGsQFjAG&url=ftp%3A%2F%2Fftp.heanet.ie%2Fmirrors%2Fsourceforge%2Fg%2Fgp%2Fgpusmoldyn%2FHIBI2010.pptx&ei=IdgMUdawMaSr4ASu44GgCQ&usg=AFQjCNEVDjIFNIn3yE8vTCLXiDFK1iEeYA&bvm=bv.41867550,d.bGE) and later I enriched it and completed it for an article on IEEE TCSB, [Smoldyn on Graphics Processing Units: Massively Parallel Brownian Dynamics Simulations](http://www.computer.org/csdl/trans/tb/2012/03/05963635-abs.html)

The code is written using nVidia CUDA, and it is based on Steven's code; therefore, this code is (C) Lorenzo Dematte' (2010-2013), released under the same [license](http://www.smoldyn.org/download.html) (GPLv2).

