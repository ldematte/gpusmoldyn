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
 
#ifndef _PARTICLES_KERNEL_H_
#define _PARTICLES_KERNEL_H_

#include "cutil_math.h"
#include "math_constants.h"
#include "particles_kernel.cuh"
#include "surfaces_kernel.cuh"

#include "reaction.h"
#include "reaction_kernel.cuh"
#include "rand_kernel.cuh"

#include "MersenneTwister.h"


#if USE_TEX
// textures for particle position and velocity
texture<float4, 1, cudaReadModeElementType> oldPosTex;
texture<float4, 1, cudaReadModeElementType> oldVelTex;

texture<uint, 1, cudaReadModeElementType> gridParticleHashTex;
texture<uint, 1, cudaReadModeElementType> cellStartTex;
texture<uint, 1, cudaReadModeElementType> cellEndTex;
#endif

texture<float, 1, cudaReadModeElementType> diffusionRatesTex; //difstep

// simulation parameters in constant memory
__constant__ SimParams params;

__constant__ float gaussianLookupTable[gaussianTableDim];
const int gaussianTableDimMinusOne = gaussianTableDim -1;

__global__
void adjustPositions(float4* posArray,       // lenght = numParticles
                     int*    types,          // lenght = numParticles
                     uint    numParticles,
                     float   particleRadius)
{
    uint index = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;
    if (index >= numParticles) return;          // handle case when no. of particles not multiple of block size

  	 float4 pos = posArray[index];    // ensure coalesced read
    //volatile int typeId = typeArray[index];
    
    // check collisions with impenetrable surfaces
    if (pos.x > 1.0f - particleRadius)  
       pos.x = 1.0f - particleRadius;
    if (pos.x < -1.0f + particleRadius)
       pos.x = -1.0f + particleRadius; 

    if (pos.y > 1.0f - particleRadius)
       pos.y = 1.0f - particleRadius; 
    if (pos.y < -1.0f + particleRadius) 
       pos.y = -1.0f + particleRadius; 

    if (pos.z > 1.0f - particleRadius)
       pos.z = 1.0f - particleRadius; 
    if (pos.z < -1.0f + particleRadius)
       pos.z = -1.0f + particleRadius; 

    

    // TODO: periodic boundaries

    // TODO: compute probability of absorbtion

    posArray[index] = pos;
}

__global__
void compactPosAndTypeD(int* dCompactedTemp, // [IN] array of compacted indexes (prefix scan)
                      int* deathBirth,     // [IN] array of live/death flags
                      int* dTypes,         // [IN/OUT] type array (of int) to compact
                      float* dPos,         // [IN/OUT] pos array (of float4) to compact
                      int numParticles)  // dimension of the 3 arrays
{
    uint i = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;
    if (i >= numParticles) return;          // handle case when no. of particles not multiple of block size
   
   //TODO: coalesced read/write?
   if (deathBirth[i] != 0) 
   {
      int newIndex = dCompactedTemp[i];
      dTypes[newIndex] = dTypes[i];

      // multiply by 4
      newIndex = newIndex * 4;
      i = i * 4;

      dPos[newIndex] = dPos[i];
      dPos[newIndex + 1] = dPos[i + 1];
      dPos[newIndex + 2] = dPos[i + 2];
      dPos[newIndex + 3] = dPos[i + 3];      
   }
   __syncthreads();
}

// integrate particle attributes
__global__
void diffuse(float4* posArray,  // input/output
               int* typeArray, // input/output
               //float4* velArray,  
               MersenneTwisterState* rngStateArray,
               float deltaTime,
               uint numParticles)
{
    uint index = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;
    if (index >= numParticles) return;          // handle case when no. of particles not multiple of block size

	 volatile float4 posData = posArray[index];    // ensure coalesced read
    volatile int typeId = typeArray[index];

    int rngIndex = index % MT_RNG_COUNT;
    MersenneTwisterState* rngState = &(rngStateArray[rngIndex]);
    //volatile float4 velData = velArray[index];
    float3 pos = make_float3(posData.x, posData.y, posData.z);
    //float3 vel = make_float3(velData.x, velData.y, velData.z);

    //vel += params.gravity * deltaTime;
    //vel *= params.globalDamping;

    // TODO: compute new position !!
    //pos += vel * deltaTime;

    // requires 3 random numbers
    // difstep: pre-computed texture
    // gtable: gaussian table
    float rate = tex1Dfetch(diffusionRatesTex, typeId);
    int randX = MersenneTwisterGenerate(rngState, index) & gaussianTableDimMinusOne;
    int randY = MersenneTwisterGenerate(rngState, index) & gaussianTableDimMinusOne;
    int randZ = MersenneTwisterGenerate(rngState, index) & gaussianTableDimMinusOne;

    pos.x += rate * gaussianLookupTable[randX]; 
    pos.y += rate * gaussianLookupTable[randY]; 
    pos.z += rate * gaussianLookupTable[randZ]; 

    // store new position
    posArray[index] = make_float4(pos, posData.w);
}


// calculate position in uniform grid
__device__ int3 calcGridPos(float3 p)
{
    int3 gridPos;
    gridPos.x = floor((p.x - params.worldOrigin.x) / params.cellSize.x);
    gridPos.y = floor((p.y - params.worldOrigin.y) / params.cellSize.y);
    gridPos.z = floor((p.z - params.worldOrigin.z) / params.cellSize.z);
    return gridPos;
}

// calculate address in grid from position (clamping to edges)
__device__ uint calcGridHash(int3 gridPos)
{
    gridPos.x = gridPos.x & (params.gridSize.x-1);  // wrap grid, assumes size is power of 2
    gridPos.y = gridPos.y & (params.gridSize.y-1);
    gridPos.z = gridPos.z & (params.gridSize.z-1);        
    return __umul24(__umul24(gridPos.z, params.gridSize.y), params.gridSize.x) + __umul24(gridPos.y, params.gridSize.x) + gridPos.x;
}

// calculate grid hash value for each particle
__global__
void calcHashD(uint*   gridParticleHash,  // output
               uint*   gridParticleIndex, // output
               float4* pos,               // input: positions
               uint    numParticles)
{
    uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (index >= numParticles) return;
    
    volatile float4 p = pos[index];

    // get address in grid
    int3 gridPos = calcGridPos(make_float3(p.x, p.y, p.z));
    uint hash = calcGridHash(gridPos);

    // store grid hash and particle index
    gridParticleHash[index] = hash;
    gridParticleIndex[index] = index;
}

// rearrange particle data into sorted order, and find the start of each cell
// in the sorted hash array
__global__
void reorderDataAndFindCellStartD(
                           uint*   cellStart,        // output: cell start index
							      uint*   cellEnd,          // output: cell end index
							      float4* sortedPos,        // output: sorted positions
  							      //float4* sortedVel,        // output: sorted velocities
                           uint *  gridParticleHash, // input: sorted grid hashes
                           uint *  gridParticleIndex,// input: sorted particle indices
				               float4* oldPos,           // input: sorted position array
							      //float4* oldVel,           // input: sorted velocity array
							      uint    numParticles)
{
	extern __shared__ uint sharedHash[];    // blockSize + 1 elements
    uint index = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;
	
    uint hash;
    // handle case when no. of particles not multiple of block size
    if (index < numParticles) {
        hash = gridParticleHash[index];

        // Load hash data into shared memory so that we can look 
        // at neighboring particle's hash value without loading
        // two hash values per thread
	    sharedHash[threadIdx.x+1] = hash;

	    if (index > 0 && threadIdx.x == 0)
	    {
		    // first thread in block must load neighbor particle hash
		    sharedHash[0] = gridParticleHash[index-1];
	    }
	}

	__syncthreads();
	
	if (index < numParticles) {
		// If this particle has a different cell index to the previous
		// particle then it must be the first particle in the cell,
		// so store the index of this particle in the cell.
		// As it isn't the first particle, it must also be the cell end of
		// the previous particle's cell

	    if (index == 0 || hash != sharedHash[threadIdx.x])
	    {
		    cellStart[hash] = index;
            if (index > 0)
                cellEnd[sharedHash[threadIdx.x]] = index;
	    }

        if (index == numParticles - 1)
        {
            cellEnd[hash] = index + 1;
        }

	    // Now use the sorted index to reorder the pos and vel data
	    uint sortedIndex = gridParticleIndex[index];
	    float4 pos = FETCH(oldPos, sortedIndex);       // macro does either global read or texture fetch
       // float4 vel = FETCH(oldVel, sortedIndex);       // see particles_kernel.cuh

       sortedPos[index] = pos;
       // sortedVel[index] = vel;
   }
}

// collide two spheres using DEM method
__device__
float3 collideSpheres(float3 posA, float3 posB,
                      float3 velA, float3 velB,
                      float radiusA, float radiusB,
                      float attraction)
{
	// calculate relative position
    float3 relPos = posB - posA;

    float dist = length(relPos);
    float collideDist = radiusA + radiusB;

    float3 force = make_float3(0.0f);
    if (dist < collideDist) {
        float3 norm = relPos / dist;

		// relative velocity
        float3 relVel = velB - velA;

        // relative tangential velocity
        float3 tanVel = relVel - (dot(relVel, norm) * norm);

        // spring force
        force = -params.spring*(collideDist - dist) * norm;
        // dashpot (damping) force
        force += params.damping*relVel;
        // tangential shear force
        force += params.shear*tanVel;
		// attraction
        force += attraction*relPos;
    }

    return force;
}

inline __device__ float lengthSquared(float3 v)
{
    return dot(v, v);
}


// collide a particle against all other particles in a given cell
__device__
void collideCell(int3    gridPos,
                 uint    index,
                 float3  pos1,
                 float4* oldPos,
                 float4* newPos,
                 uint* gridParticleIndex,      //length = numParticles
                 int* types,
                 int numTypes,
                 int* birthDeath,
                 float* diffusionCoefficients,
                 int* reactionTable,
                 Reaction* reactionList,
                 uint*   cellStart,
                 uint*   cellEnd,
                 MersenneTwisterState* randState,
                 unsigned int threadId)
{
    uint gridHash = calcGridHash(gridPos);

    // get start of bucket for this cell
    uint startIndex = FETCH(cellStart, gridHash);

    uint mol1Index = gridParticleIndex[index];

    //if (birthDeath[mol1Index] >= 0) //this molecule was already processed
    //   return;
    //birthDeath[mol1Index] = 1; // we keep it, as a default

    int mol1Type = types[mol1Index];
    //invalid molecule?
    if (mol1Type <= 0)
    {
       birthDeath[mol1Index] = 0;// should never happen!
       return;
    }

    float3 force = make_float3(0.0f);
    if (startIndex != 0xffffffff) {        // cell is not empty
        // iterate over particles in this cell
        uint endIndex = FETCH(cellEnd, gridHash);
        for(uint j=startIndex; j<endIndex; j++) 
        {
           // index is the current particle index, j the one we will test on
           // both are indexes to the sorted array 

           // check not colliding with self
            if (j != index) 
            {
               // get back the original unsorted location
               uint mol2Index = gridParticleIndex[j];

               //if (birthDeath[mol2Index] >= 0) //this molecule was already processed
               //   continue;
               //birthDeath[mol2Index] = 1; //for now, flag as "save it"

               int mol2Type = types[mol2Index];
               if (mol2Type == -1)
               {
                  birthDeath[mol2Index] = 0;// should never happen!
                  continue;
               }

               // is a reaction possible between them?
               // TODO: multiple reactions possible!
               int tableIdx = mol1Type * numTypes + mol2Type;
               int r = reactionTable[tableIdx];
               if (r == -1)
                  continue;               
               
	            float3 pos2 = make_float3(FETCH(oldPos, j));

               // calculate relative position
               float3 relPos = pos1 - pos2;
               float distSquared = lengthSquared(relPos);
               
               if (distSquared <= reactionList->bindRadiusSquared[r])
               {
                  float p = reactionList->prob[r];
                  if (p == 1 || randReal(randState, threadId) <p)    
                  {
                     bimReact(reactionList, r, 
                        mol1Index, mol2Index,  
                        mol1Type, mol2Type, 
                        pos1, pos2, 
                        newPos, types, birthDeath,
                        gridParticleIndex,
                        diffusionCoefficients, 
                        randState, threadId); 
                  }
               }
            }
        }
    }
}

__global__ 
void checkSurfaceCollisions(float4* newPos,               // input: new positions
                            float4* oldPos,               // input: old positions
                            int* birthDeath,              // output: who was absorbed (dead) TODO check is handled by later steps
                            Panel* panels,
                            uint numParticles)
{
   uint molIndex = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;
   if (molIndex >= numParticles) return;          // handle case when no. of particles not multiple of block size

   float4 molPos = newPos[molIndex];
   float4 molOldPos = oldPos[molIndex];
   
   uint2 gridIndex = getSurfaceGridIndex(molPos);
   uint startIndex = gridIndex.x, endIndex = gridIndex.y;
   // iterate over panels in this cell
   for(uint j = startIndex; j < endIndex; j++) 
   {
      Panel* panel = &(panels[j]);
      uint numCross = 2;
      float4 crossPoint = lineXpanel(pos, oldPos, panel, &numCross);     
      if(numCross < 2) 
      {
         //a panel was crossed, deal with it
         doSurfaceInteraction(molIndex, panels, j, crossPoint);
      }
   }
}


__global__
void collideD(float4* newPos,               // output: new positions
              float4* oldPos,               // input: sorted positions
              uint*   gridParticleIndex,    // input: sorted particle indices
              int* types,
              int numTypes,
              int* birthDeath,
              float* diffusionCoefficients,
              int* reactionTable,
              Reaction* reactionList,
              uint*   cellStart,
              uint*   cellEnd,
              uint    numParticles, 
              MersenneTwisterState* rngStateArray)
{
    uint index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
    if (index >= numParticles)
       return;

    int rngIndex = index % MT_RNG_COUNT;
    MersenneTwisterState* rngState = &(rngStateArray[rngIndex]);
    
    // read particle data from sorted arrays
	 float3 pos = make_float3(FETCH(oldPos, index));

    // get address in grid
    int3 gridPos = calcGridPos(pos);

    // examine neighbouring cells
    float3 force = make_float3(0.0f);
    for(int z=-1; z<=1; z++) 
    {
        for(int y=-1; y<=1; y++) 
        {
            for(int x=-1; x<=1; x++) 
            {
                int3 neighbourPos = gridPos + make_int3(x, y, z);
                collideCell(neighbourPos, index, pos,
                            oldPos, newPos, gridParticleIndex, 
                            types, numTypes, birthDeath, 
                            diffusionCoefficients, 
                            reactionTable, reactionList,
                            cellStart, cellEnd,
                            rngState, index);
            }
        }
    }

    // collide with cursor sphere
    //force += collideSpheres(pos, params.colliderPos, vel, make_float3(0.0f, 0.0f, 0.0f), params.particleRadius, params.colliderRadius, 0.0f);

    // write new velocity back to original unsorted location
    //uint originalIndex = gridParticleIndex[index];
    //newVel[originalIndex] = make_float4(vel + force, 0.0f);
}

__global__
void resetNonInteracting(int* birthDeath, int numParticles)
{
   uint index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
   if (index >= numParticles)
      return;

   if (birthDeath[index] == -1)
      birthDeath[index] = 1;
}


#endif
