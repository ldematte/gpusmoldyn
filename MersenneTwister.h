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

#ifndef CUDA_MERSENNE_H_INCLUDED
#define CUDA_MERSENNE_H_INCLUDED

const int MT_NN = 19;
const int MT_RNG_COUNT = 32768;
const unsigned int MT_WMASK = 0xFFFFFFFFU;

// Record format for MersenneTwister.dat, created by spawnTwisters.c
struct mt_struct_stripped {
	unsigned int matrix_a;
	unsigned int mask_b;
	unsigned int mask_c;
	unsigned int seed;
};

// Per-thread state object for a single twister.
struct MersenneTwisterState {
	unsigned int mt[MT_NN];
	int iState;
	unsigned int mti1;
};

void InitializeRandomNumbers();
//void InitializeMT(MersenneTwisterState* stateArray);

#endif //CUDA_MERSENNE_H_INCLUDED

