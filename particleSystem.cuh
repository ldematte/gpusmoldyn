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

#include "reaction.h"

 extern "C"
{
void cudaInit(int argc, char **argv);

void allocateArray(void **devPtr, int size);
void freeArray(void *devPtr);

void threadSync();

void copyArrayFromDevice(void* host, const void* device, unsigned int vbo, int size);
void copyArrayToDevice(void* device, const void* host, int offset, int size);
void registerGLBufferObject(unsigned int vbo);
void unregisterGLBufferObject(unsigned int vbo);
void *mapGLBufferObject(uint vbo);
void unmapGLBufferObject(uint vbo);

void setParameters(SimParams *hostParams);

void copyGaussianToConstant(float* h_gaussianLookupTable, uint gaussianTableDim);

void initializeMT(MersenneTwisterState* stateArray);

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
                     uint   numTypes);

void adjustPositions(float*  pos,            // lenght = numParticles
                     int*    types,          // lenght = numParticles
                     uint    numParticles,
                     float   particleRadius);

void calcHash(uint*  gridParticleHash,
              uint*  gridParticleIndex,
              float* pos, 
              int    numParticles);

void reorderDataAndFindCellStart(
                          uint*  cellStart,
							     uint*  cellEnd,
							     float* sortedPos,
                          uint*  gridParticleHash,
                          uint*  gridParticleIndex,
							     float* oldPos,
							     uint   numParticles,
							     uint   numCells);

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
             MersenneTwisterState* rngStateArray);

int compactPosAndType(int* dCompactedTemp, // [IN] array of compacted indexes (prefix scan)
                      int* deathBirth,     // [IN] array of live/death flags
                      float* dPos,         // [IN/OUT] pos array (of float4) to compact
                      int* dTypes,         // [IN/OUT] type array (of int) to compact
                      int numParticles);   // dimension of the 4 arrays

}
