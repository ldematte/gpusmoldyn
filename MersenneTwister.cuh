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
 
#include "MersenneTwister.h"

const unsigned int MT_MM = 9;
const unsigned int MT_UMASK = 0xFFFFFFFEU;
const unsigned int MT_LMASK = 0x1U;
const unsigned int MT_SHIFT0 = 12;
const unsigned int MT_SHIFTB =7;
const unsigned int MT_SHIFTC =15;
const unsigned int MT_SHIFT1 =18;

// Preloaded, offline-generated seed data structure.
__device__ static mt_struct_stripped MT[MT_RNG_COUNT];

__device__ void MersenneTwisterInitialise(MersenneTwisterState* state, unsigned int threadID) 
{
	state->mt[0] = MT[threadID].seed;
	for(int i = 1; i < MT_NN; ++ i) 
   {
		state->mt[i] = (1812433253U * (state->mt[i - 1] ^ (state->mt[i - 1] >> 30)) + i) & MT_WMASK;
   }
	state->iState = 0;
	state->mti1 = state->mt[0];
}

__device__ unsigned int MersenneTwisterGenerate(MersenneTwisterState* state, unsigned int threadID) 
{
   threadID = threadID % MT_RNG_COUNT;

	int iState1 = state->iState + 1;
	int iStateM = state->iState + MT_MM;

	if(iState1 >= MT_NN) iState1 -= MT_NN;
	if(iStateM >= MT_NN) iStateM -= MT_NN;

	unsigned int mti = state->mti1;
	state->mti1 = state->mt[iState1];
	unsigned int mtiM = state->mt[iStateM];

	unsigned int x = (mti & MT_UMASK) | (state->mti1 & MT_LMASK);
	x = mtiM ^ (x >> 1) ^ ((x & 1) ? MT[threadID].matrix_a : 0);
	state->mt[state->iState] = x;
	state->iState = iState1;

	// Tempering transformation.
	x ^= (x >> MT_SHIFT0);
	x ^= (x << MT_SHIFTB) & MT[threadID].mask_b;
	x ^= (x << MT_SHIFTC) & MT[threadID].mask_c;
	x ^= (x >> MT_SHIFT1);

	return x;
}

// Call one with one thread per array element
__global__ void initializeMersenneTwister(MersenneTwisterState* stateArray) 
{
   unsigned int tid = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	MersenneTwisterState* mtState = &(stateArray[tid]);
   MersenneTwisterInitialise(mtState, tid);

	//OPTIONAL "Warm-up" the Twister to avoid initial correlation with others.
   //for(int i = 0; i < 10000; ++ i) {
   //   MersenneTwisterGenerate(mtState, tid);
   //}
}

