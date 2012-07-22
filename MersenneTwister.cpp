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
 * This file is derived from the NVIDIA CUDA SDK example 'MersenneTwister'.
 *
 *
 * Copyright 1993-2007 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.  Users and possessors of this source code
 * are hereby granted a nonexclusive, royalty-free license to use this code
 * in individual and commercial software.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.   This source code is a "commercial item" as
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer  software"  and "commercial computer software
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 *
 * Any use of this source code in individual and commercial software must
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */

// Some parts contain code from Makoto Matsumoto and Takuji Nishimura's dci.h

/* Copyright (C) 2001-2006 Makoto Matsumoto and Takuji Nishimura.  */
/* This library is free software; you can redistribute it and/or   */
/* modify it under the terms of the GNU Library General Public     */
/* License as published by the Free Software Foundation; either    */
/* version 2 of the License, or (at your option) any later         */
/* version.                                                        */
/* This library is distributed in the hope that it will be useful, */
/* but WITHOUT ANY WARRANTY; without even the implied warranty of  */
/* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.            */
/* See the GNU Library General Public License for more details.    */
/* You should have received a copy of the GNU Library General      */
/* Public License along with this library; if not, write to the    */
/* Free Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA   */
/* 02111-1307  USA                                                 */

#include <assert.h>
#include <stdio.h>
#include <vector>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cutil.h>

#include "MersenneTwister.h"

//#define CUDA_SAFE_CALL(call)                                        \
//  do {                                                              \
//    cudaError_t err = call;                                         \
//    if( cudaSuccess != err) {                                       \
//      fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n", \
//        __FILE__, __LINE__, cudaGetErrorString( err) );             \
//      exit(EXIT_FAILURE);                                           \
//    }                                                               \
//  } while(0)





//
//__global__ void TestMersenneTwister(MersenneTwisterState* outArr, int nNumbers) {
//	unsigned int tid = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
//
//	MersenneTwisterState mtState;
//	MersenneTwisterInitialise(mtState, tid);
//
//	// "Warm-up" the Twister to avoid initial correlation with others.
//	for(int i = 0; i < 10000; ++ i) {
//		MersenneTwisterGenerate(mtState, tid);
//	}
//
//	for(int i = tid; i < nNumbers; i += __mul24(blockDim.x, gridDim.x)) {
//		// Make a floating-point number between 0...1 from integer 0...UINT_MAX.
//		outArr[i] = float(MersenneTwisterGenerate(mtState, tid)) / 4294967295.0f;
//	}
//}

void InitializeRandomNumbers()
{
	// Read offline-generated initial configuration file.
	mt_struct_stripped *mtStripped = new mt_struct_stripped[MT_RNG_COUNT];

	FILE *datFile = fopen(".\\MersenneTwister.dat", "rb");
	assert(datFile);
	assert(fread(mtStripped, sizeof(mt_struct_stripped) * MT_RNG_COUNT, 1, datFile));
	fclose(datFile);

	// Seed the structure with low-quality random numbers. Twisters will need "warming up"
	// before the RNG quality improves.
	srand(time(0));
	for(int i = 0; i < MT_RNG_COUNT; ++ i) {
		mtStripped[i].seed = rand();
	}

	// Upload the initial configurations to the GPU.
   // Use the current CUDA context to upload the initial configurations.
		//CUdeviceptr mtDev;
		//CU_SAFE_CALL(cuModuleGetGlobal(&mtDev, 0, cuModule, "MT"));
		//CU_SAFE_CALL(cuMemcpyHtoD(mtDev, mtStripped, sizeof(mt_struct_stripped) * MT_RNG_COUNT));

   // TODO!! verify is working!!
	CUDA_SAFE_CALL(cudaMemcpyToSymbol("MT", mtStripped, sizeof(mt_struct_stripped) * MT_RNG_COUNT, 0, cudaMemcpyHostToDevice));
	delete[] mtStripped;
}

//(float *randomNumbers, int nRandomNumbers) 

	// Run the CUDA MersenneTwister program.
	//float *randomNumbersDev;
	//CUDA_SAFE_CALL(cudaMalloc((void **)&randomNumbersDev, sizeof(float) * nRandomNumbers));

	//dim3 threads(512, 1);
	//dim3 grid(MT_RNG_COUNT / 512, 1, 1);

	//TestMersenneTwister<<<grid, threads>>>(randomNumbersDev, nRandomNumbers);

	//CUDA_SAFE_CALL(cudaMemcpy(randomNumbers, randomNumbersDev, sizeof(float) * nRandomNumbers, cudaMemcpyDeviceToHost));
	//CUDA_SAFE_CALL(cudaFree(randomNumbersDev));




