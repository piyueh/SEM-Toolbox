SEM Exercises
=============

This repo contains Python codes to the exercise problems in the book:
Spectral/hp Element Methods for Computational Fluid Dynamics,
2nd edition, by George Em Karniadakis & Spencer Sherwin.

The repository is divided into two parts: 

1. a Python package -- **utils**: This package includes re-usable 
components, such as Polynomial, quadrature, expansions, assembly, solvers, etc.
I use these components to write solutions to the exercises. These components
are in object-oriented style. Also, I try not to use math libraries.
Typically only numpy ndarray is used. This causes some performance problems and
some naive implementations. For example, fractional function is very naive.
2. a collection of Jupyter notebooks -- **solutions**: 
This folder includes Jupyter notebooks to each exercises in the book. My 
intention is not to provide a teaching material, 
so normally there are only codes and answers to the 
exercises. Normally no explainations or detailed derivations are provided.
And I don't really care how the codes in these notebooks look like, 
so they are typically a mess. I care about the **utils** package more.

##Warning:  

The intention of these codes are to check whether I really understand the 
theory described in the book, not to develop a serious SEM solver 
(though I hope these code can one day be used in real applications ...), so

1. the perfomance is not guaranteed;
2. the consistency of naming convention and coding style is not guaranteed; and
3. don't expect to see a good practice of software engineering here.


## Known issues:

1. Root-finding function is vary naieve, so it can not handle too many multiple
roots.
2. Jacobi polynomials with very high orders (P>=30, I think) cant not correctly
find their roots. It will return complex roots due to the root-finding method I
used. This means quadratures with very high orders may not work, either.
