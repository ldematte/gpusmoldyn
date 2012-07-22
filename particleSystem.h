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

#ifndef __PARTICLESYSTEM_H__
#define __PARTICLESYSTEM_H__


const int MAX_PARTICLES = 8388608; //8 MB

#define DEBUG_GRID 0
#define DO_TIMING 0

#pragma warning(disable:4201)
#pragma warning(disable:4408)

#include "particles_kernel.cuh"
#include "vector_functions.h"
#include "radixsort.h"

#include "MersenneTwister.h"
#include "reaction.h"
#include "stats.h"

#include "cudpp/cudpp.h"

// Particle system class
class ParticleSystem
{

private:
   void test1Reactions();
   int* test1Types();

   void test2Reactions();
   int* test2Types();

   void test3Reactions();
   int* test3Types();

public:
    ParticleSystem(uint numParticles, uint3 gridSize, bool bUseOpenGL);
    ~ParticleSystem();

    enum ParticleConfig
    {
	    CONFIG_RANDOM,
	    CONFIG_GRID,
	    _NUM_CONFIGS
    };

    enum ParticleArray
    {
        POSITION,
        VELOCITY,
    };

    bool update();
    void reset(ParticleConfig config);

    float* getArray(ParticleArray array);
    void   setArray(ParticleArray array, const float* data, int start, int count);

    int   getNumParticles() const { return m_numParticles; }
    int   getNumTypes() const { return m_numTypes; }
    float getDeltaTime() const { return deltaTime; }
    float getSystemVolume() const { return systemVolume; }

    unsigned int getPosBuffer()         const { return m_posVBO; }
    unsigned int getTypesBuffer()       const { return m_typesVBO; }

    void * getCudaPosVBO()              const { return (void *)m_cudaPosVBO; }
    void * getCudaTypesVBO()            const { return (void *)m_cudaTypesVBO; }

    void dumpGrid();
    void dumpParticles(uint start, uint count);

    void setCollideSpring(float x) { m_params.spring = x; }
    void setCollideDamping(float x) { m_params.damping = x; }
    void setCollideShear(float x) { m_params.shear = x; }
    void setCollideAttraction(float x) { m_params.attraction = x; }

    void setColliderPos(float3 x) { m_params.colliderPos = x; }

    float getParticleRadius() { return m_params.particleRadius; }
    float3 getColliderPos() { return m_params.colliderPos; }
    float getColliderRadius() { return m_params.colliderRadius; }
    uint3 getGridSize() { return m_params.gridSize; }
    float3 getWorldOrigin() { return m_params.worldOrigin; }
    float3 getCellSize() { return m_params.cellSize; }

    // for now, the world is just -1, +1. 
    void randPos(float3& pos) 
    {  
       pos.x = unirandOOF(-1.0f, +1.0f);
       pos.y = unirandOOF(-1.0f, +1.0f);
       pos.z = unirandOOF(-1.0f, +1.0f);
    }

    float* getDiffusionCoefficients() { return m_hDiffusionCoefficients; }

public: //members
    ReactionOrderInfo reactionOrderInfo[3];
    Reaction* reactionList;

protected: // methods
    ParticleSystem() {}
    uint createVBO(uint size);

    void _initialize(int numParticles);
    void _finalize();

    void allocateSpace(uint numParticles);
    void reallocateSpace(uint allocateParticles, uint prevNumParticles, float* oldPos, int* oldTypes);

    void initGrid(uint *size, float spacing, float jitter, uint numParticles);

protected: // data
    bool m_bInitialized, m_bUseOpenGL;
    uint m_numParticles;
    uint m_numTypes;

    uint m_allocatedParticles;

    float deltaTime;
    float systemVolume;

    // CPU data
    float* m_hPos;              // particle positions
    int* m_hType;             // particle types  TODO: to initialize

    float* m_hAdditionalPos;
    int* m_hAdditionalTypes; 


    uint*  m_hParticleHash;
    uint*  m_hCellStart;
    uint*  m_hCellEnd;

    float* m_hDiffusionCoefficients;

    //// CPU, for zeroth order reactions
    //float* m_hAddedPos;
    //int* m_hAddedTypes;

    // GPU data
    float* m_dPos;
    
    float* m_dAddedPos;    
    int* m_dAddedTypes;
    int* m_dDeath;
    int* m_dBirth;
    
    float* m_dDiffusionRates;  // diffusion rate for each type 
    float* m_dDiffusionCoefficients;
    MersenneTwisterState* m_dRngStateArray;

    int m_numReactions;
    Reaction* m_dReactionList;
    int* m_dReactionTable;

    // ReactionOrderInfo (for first reaction)
    int* m_dNumberOfReactionsPerType1;
    int* m_dLocalToGlobalReaction1;
    int* m_dReactionsPerTypeIdx1;


    float* m_dSortedPos;
    //float* m_dSortedVel;

    // grid data for sorting method
    uint*  m_dGridParticleHash; // grid hash value for each particle
    uint*  m_dGridParticleIndex;// particle index for each particle
    uint*  m_dCellStart;        // index of start of each cell in sorted list
    uint*  m_dCellEnd;          // index of end of cell

    uint   m_gridSortBits;

    uint   m_posVBO;            // vertex buffer object for particle positions
    uint   m_typesVBO;          // vertex buffer object for colors
    
    float* m_cudaPosVBO;        // these are the CUDA deviceMem Pos
    int*   m_cudaTypesVBO;      // these are the CUDA deviceMem Color

    int* dCompactedTemp;
    CUDPPHandle scanHandle; // CUDPP plan handle for prefix sum

    RadixSort *m_sorter;

    // params
    SimParams m_params;
    uint3 m_gridSize;
    uint m_numGridCells;

    uint m_timer;
};

#endif // __PARTICLESYSTEM_H__
