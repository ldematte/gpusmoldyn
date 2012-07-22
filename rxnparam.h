/*
 * GPU Smoldyn: Smoldyn algorithm ported to the GPU using CUDA 2.2
 * Writtern By Lorenzo Dematté, 2010-2011
 *
 * This file is part of GPU Smoldyn
 * 
 *     GPU Smoldyn is free software: you can redistribute it and/or modify
 *     it under the terms of the GNU General Public License as published by
 *     the Free Software Foundation, either version 3 of the License, or
 *     (at your option) any later version.
 * 
 *     GPU Smoldyn is distributed in the hope that it will be useful,
 *     but WITHOUT ANY WARRANTY; without even the implied warranty of
 *     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *     GNU General Public License for more details.
 * 
 *     You should have received a copy of the GNU General Public License
 *     along with Foobar.  If not, see <http://www.gnu.org/licenses/>. 
 * 
 * Based on algorithm and source code of Smoldyn, written by Steve Andrews, 2003.
 * 
 * Portions taken by code examples in NVIDIA Whitepapers, GPU Gems 2 and 3, 
 * Copyright 1993-2009 NVIDIA Corporation, Addison-Wesley and the original authors. 
 * 
 */

/* File rxnparam.h, written by Steven Andrews, 2003.
This code is in the public domain.  It is not copyrighted and may not be
copyrighted.

This is a header file for rxnparam.c.  See rxnparam_doc.doc or rxn_param_doc.pdf
for documentation. */

#ifndef __rxnparam_h
#define __rxnparam_h

/***  LOOK-UP FUNCTIONS FOR REACTION RATES AND BINDING AND UNBINDING RADII  ***/

double numrxnrate(double step,double a,double b);
double actrxnrate(double step,double a);
double bindingradius(double rate,double dt,double difc,double b,int rel);
double unbindingradius(double pgem,double dt,double difc,double a);

/************    FUNCTIONS FOR INVESTIGATING AN ABSORBING SPHERE    ***********/

double rdfabsorb(double *r,double *rdf,int n);
void rdfdiffuse(double *r,double *rdfa,double *rdfd,int n,double step);
void rdfreverserxn(double *r,double *rdf,int n,double step,double b,double flux);
double rdfsteadystate(double *r,double *rdfa,double *rdfd,int n,double step,double b,double eps);
void rdfmaketable();


#endif
