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

// This file contains C wrappers around the some of the CUDA API and the
// kernel functions so that they can be called from "particleSystem.cpp"

#include <cutil.h>
#include <cutil_inline.h>
#include <cstdlib>
#include <cstdio>
#include <string.h>

#include "common.h"
#include <GL/glut.h>

#include <cuda_gl_interop.h>

#include "particles_kernel.cu"

extern "C"
{

void cudaInit(int argc, char **argv)
{   
    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    if( cutCheckCmdLineFlag(argc, (const char**)argv, "device") ) {
        cutilDeviceInit(argc, argv);
    } else {
        cudaSetDevice( cutGetMaxGflopsDeviceId() );
    }
}

void cudaGLInit(int argc, char **argv)
{   
    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    if( cutCheckCmdLineFlag(argc, (const char**)argv, "device") ) {
        cutilDeviceInit(argc, argv);
    } else {
        cudaGLSetGLDevice( cutGetMaxGflopsDeviceId() );
    }
}

void allocateArray(void **devPtr, size_t size)
{
   cutilSafeCall(cudaMalloc(devPtr, size));
   CUT_CHECK_ERROR("allocateArray");
}

void freeArray(void *devPtr)
{
    cutilSafeCall(cudaFree(devPtr));
}

void threadSync()
{
    cutilSafeCall(cudaThreadSynchronize());
}

void copyArrayFromDevice(void* host, const void* device, unsigned int vbo, int size)
{   
    if (vbo)
        cutilSafeCall(cudaGLMapBufferObject((void**)&device, vbo));

    cutilSafeCall(cudaMemcpy(host, device, size, cudaMemcpyDeviceToHost));
    
    if (vbo)
        cutilSafeCall(cudaGLUnmapBufferObject(vbo));
}

void copyArrayToDevice(void* device, const void* host, int offset, int size)
{
    cutilSafeCall(cudaMemcpy((char *) device + offset, host, size, cudaMemcpyHostToDevice));
}

void registerGLBufferObject(uint vbo)
{
    cutilSafeCall(cudaGLRegisterBufferObject(vbo));
}

void unregisterGLBufferObject(uint vbo)
{
    cudaError_t err = cudaGLUnregisterBufferObject(vbo);
    cutilSafeCall(err);
}

void *mapGLBufferObject(uint vbo)
{
    void *ptr;
    cutilSafeCall(cudaGLMapBufferObject(&ptr, vbo));
    return ptr;
}

void unmapGLBufferObject(uint vbo)
{
    cutilSafeCall(cudaGLUnmapBufferObject(vbo));
}

void setParameters(SimParams *hostParams)
{
    // copy parameters to constant memory
    cutilSafeCall( cudaMemcpyToSymbol(params, hostParams, sizeof(SimParams)) );
}

// compute grid and thread block size for a given number of elements
void computeGridSize(uint n, uint blockSize, uint &numBlocks, uint &numThreads)
{
    numThreads = min(blockSize, n);
    numBlocks = iDivUp(n, numThreads);
}

void copyGaussianToConstant(float* h_gaussianLookupTable, uint gaussianTableDim)
{
   cudaMemcpyToSymbol(gaussianLookupTable, h_gaussianLookupTable, sizeof(float) * gaussianTableDim);
}

void initializeMT(MersenneTwisterState* stateArray)
{
   dim3 threads(512, 1);
	dim3 grid(MT_RNG_COUNT / 512, 1, 1);

	initializeMersenneTwister<<<grid, threads>>>(stateArray);
}


void integrateSystem(Reaction*          reactionList,
                     int* localToGlobalReaction1,       // length = rxnss[1].LocalToGlobalSize
                     int* reactionsPerTypeIdx1,         // lenght = numTypes
                     int* numberOfReactionsPerType1,    // lenght = numTypes
                     int numberOfReactions1,    
                     float*             pos,            // lenght = numParticles
                     int*               types,          // lenght = numParticles
                     float*             addedPos,       // lenght = numParticles
                     int*               addedTypes,     // lenght = numParticles
                     int*               birthArray,     // lenght = numParticles
                     int*               deathArray,     // lenght = numParticles

                     float*             diffusionRates, // lenght = numTypes
                     MersenneTwisterState* rngStateArray, //length = MT_RNG_COUNT (num max particles/num max threads)
                     float  deltaTime,
                     uint   numParticles,
                     uint   numTypes)
{
    uint numThreads, numBlocks;
    computeGridSize(numParticles, 256, numBlocks, numThreads);

    //__asm int 3;

    cutilSafeCall(cudaBindTexture(0, diffusionRatesTex, diffusionRates, numTypes*sizeof(float)));


    // execute the diffuse kernel
    diffuse<<< numBlocks, numThreads >>>((float4*)pos,
                                           types,
                                           rngStateArray,
                                           deltaTime,                                           
                                           numParticles);


    if (numberOfReactions1 > 0)
    {
      uniReact<<< numBlocks, numThreads >>>(numberOfReactionsPerType1,
                                          localToGlobalReaction1, 
                                          reactionsPerTypeIdx1,
                                          //rxnss.reactionIndexes, 
                                          (float4*)pos,
                                          types,
                                          (float4*)addedPos,
                                          addedTypes,                                          
                                          birthArray,
                                          deathArray,
                                          rngStateArray,
                                          reactionList,
                                          numParticles);
    }

    cutilSafeCall(cudaUnbindTexture(diffusionRatesTex));
    
    // check if kernel invocation generated an error
    cutilCheckMsg("integrate kernel execution failed");
}

void adjustPositions(float*  pos,            // lenght = numParticles
                     int*    types,          // lenght = numParticles
                     uint    numParticles,
                     float   particleRadius)
{
    uint numThreads, numBlocks;
    computeGridSize(numParticles, 256, numBlocks, numThreads);

    // execute the diffuse kernel
    adjustPositions<<< numBlocks, numThreads >>>((float4*)pos,
                                                 types,
                                                 numParticles,
                                                 particleRadius);

}

void calcHash(uint*  gridParticleHash,
              uint*  gridParticleIndex,
              float* pos, 
              int    numParticles)
{
    uint numThreads, numBlocks;
    computeGridSize(numParticles, 256, numBlocks, numThreads);

    // execute the kernel
    calcHashD<<< numBlocks, numThreads >>>(gridParticleHash,
                                           gridParticleIndex,
                                           (float4 *) pos,
                                           numParticles);
    
    // check if kernel invocation generated an error
    cutilCheckMsg("Kernel execution failed");
}

void reorderDataAndFindCellStart(
                          uint*  cellStart,
							     uint*  cellEnd,
							     float* sortedPos,
							     //float* sortedVel,
                          uint*  gridParticleHash,
                          uint*  gridParticleIndex,
							     float* oldPos,
							     //float* oldVel,
							     uint   numParticles,
							     uint   numCells)
{
    uint numThreads, numBlocks;
    computeGridSize(numParticles, 256, numBlocks, numThreads);

    // set all cells to empty
	cutilSafeCall(cudaMemset(cellStart, 0xffffffff, numCells*sizeof(uint)));

#if USE_TEX
    cutilSafeCall(cudaBindTexture(0, oldPosTex, oldPos, numParticles*sizeof(float4)));
    //cutilSafeCall(cudaBindTexture(0, oldVelTex, oldVel, numParticles*sizeof(float4)));
#endif

    uint smemSize = sizeof(uint)*(numThreads+1);
    reorderDataAndFindCellStartD<<< numBlocks, numThreads, smemSize>>>(
        cellStart,
        cellEnd,
        (float4 *) sortedPos,
        //(float4 *) sortedVel,
		  gridParticleHash,
		  gridParticleIndex,
        (float4 *) oldPos,
        //(float4 *) oldVel,
        numParticles);
    cutilCheckMsg("Kernel execution failed: reorderDataAndFindCellStartD");

#if USE_TEX
    cutilSafeCall(cudaUnbindTexture(oldPosTex));
    cutilSafeCall(cudaUnbindTexture(oldVelTex));
#endif
}


int compactPosAndType(int* dCompactedTemp, // [IN] array of compacted indexes (prefix scan)
                      int* deathBirth,     // [IN] array of live/death flags
                      float* dPos,         // [IN/OUT] pos array (of float4) to compact
                      int* dTypes,         // [IN/OUT] type array (of int) to compact
                      int numParticles)    // dimension of the 4 arrays
{
   
   uint numThreads, numBlocks;
   computeGridSize(numParticles, 256, numBlocks, numThreads);

   compactPosAndTypeD<<<numBlocks, numThreads>>>(dCompactedTemp, 
                                               deathBirth,    
                                               dTypes,
                                               dPos,                                                  
                                               numParticles);

   // readback total number of particles generated    
   uint lastElement, lastScanElement;
   cutilSafeCall(cudaMemcpy((void *) &lastElement, 
                       (void *) (deathBirth + numParticles - 1), 
                       sizeof(int), cudaMemcpyDeviceToHost));
   cutilSafeCall(cudaMemcpy((void *) &lastScanElement, 
                       (void *) (dCompactedTemp + numParticles - 1), 
                       sizeof(int), cudaMemcpyDeviceToHost));
   return lastElement + lastScanElement;
}

void collide(float* newPos,
             float* sortedPos,
             uint*  gridParticleIndex,
             int*   types,
             int*   birthDeath,
             float* diffusionCoefficients,
             int*   reactionTable,
             Reaction* reactionList,
             uint*  cellStart,
             uint*  cellEnd,
             uint   numParticles,
             uint   numCells,
             uint   numTypes,
             MersenneTwisterState* rngStateArray)

{
#if USE_TEX
    cutilSafeCall(cudaBindTexture(0, oldPosTex, sortedPos, numParticles*sizeof(float4)));
    //cutilSafeCall(cudaBindTexture(0, oldVelTex, sortedVel, numParticles*sizeof(float4)));

    // use sorted arrays
    cutilSafeCall(cudaBindTexture(0, cellStartTex, cellStart, numCells*sizeof(uint)));
    cutilSafeCall(cudaBindTexture(0, cellEndTex, cellEnd, numCells*sizeof(uint)));    
#endif

    // thread per particle
    uint numThreads, numBlocks;
    computeGridSize(numParticles, 64, numBlocks, numThreads);

    CUT_CHECK_ERROR("");

    // execute the kernel
    collideD<<< numBlocks, numThreads >>>((float4*)newPos,
                                          (float4*)sortedPos,
                                          gridParticleIndex,
                                          types, 
                                          numTypes,
                                          birthDeath,
                                          diffusionCoefficients,
                                          reactionTable,
                                          reactionList,
                                          cellStart,
                                          cellEnd,
                                          numParticles,
                                          rngStateArray);


    // check if kernel invocation generated an error
    cutilCheckMsg("Kernel execution failed");

    //reset the ones that did not interact at all; just keep them
    resetNonInteracting<<< numBlocks, numThreads >>> (birthDeath, numParticles);

#if USE_TEX
    cutilSafeCall(cudaUnbindTexture(oldPosTex));
    cutilSafeCall(cudaUnbindTexture(cellStartTex));
    cutilSafeCall(cudaUnbindTexture(cellEndTex));
#endif
}

}   // extern "C"
