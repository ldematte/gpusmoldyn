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
 
#ifndef RAND_KERNEL_H_INCLUDED
#define RAND_KERNEL_H_INCLUDED

#include "MersenneTwister.cuh"

inline __device__ float randReal(MersenneTwisterState* randState, unsigned int threadId)
{
   uint v = MersenneTwisterGenerate(randState, threadId);
   return v * (1.0f/4294967296.0f);  // divided by 2^32
}

inline __device__ float thetarandCCF(MersenneTwisterState* randState, unsigned int threadId) 
{
	return acosf(1.0f-2.0f*randReal(randState, threadId)); 
}

inline __device__ float unirandCOF(float lo, float hi, MersenneTwisterState* randState, unsigned int threadId) 
{
	return randReal(randState, threadId)*(hi-lo)+lo; 
}

inline __device__ int coinrandF(float p, MersenneTwisterState* randState, unsigned int threadId) 
{
	return (int)(randReal(randState, threadId) < p); 
}

#endif //RAND_KERNEL_H_INCLUDED
