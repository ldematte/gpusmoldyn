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

#include "stats.h"
#include "smolreact.h"
#include "particleSystem.h"
#include "particleSystem.cuh"
#include "particles_kernel.cuh"

#include <cutil_inline.h>

#include <assert.h>
#include <math.h>
#include <memory.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <GL/glew.h>

int zeroReact(ParticleSystem* sim, float* newPos, int* newTypes);

//#ifndef CUDART_PI_F
//#define CUDART_PI_F         3.141592654f
//#endif

ParticleSystem::ParticleSystem(uint numParticles, uint3 gridSize, bool bUseOpenGL) :
m_bInitialized(false),
m_bUseOpenGL(bUseOpenGL),
m_numParticles(numParticles),
m_hPos(0),
m_dPos(0),
m_gridSize(gridSize),
m_timer(0)
{
   m_allocatedParticles = 0;
   m_numGridCells = m_gridSize.x*m_gridSize.y*m_gridSize.z;
   //float3 worldSize = make_float3(2.0f, 2.0f, 2.0f);

   m_gridSortBits = 18;    // increase this for larger grids

   // set simulation parameters
   m_params.gridSize = m_gridSize;
   m_params.numCells = m_numGridCells;
   //m_params.numBodies = m_numParticles;

   m_params.particleRadius = 1.0f / 64.0f;
   m_params.colliderPos = make_float3(-1.2f, -0.8f, 0.8f);
   m_params.colliderRadius = 0.2f;

   m_params.worldOrigin = make_float3(-1.0f, -1.0f, -1.0f);
   //    m_params.cellSize = make_float3(worldSize.x / m_gridSize.x, worldSize.y / m_gridSize.y, worldSize.z / m_gridSize.z);
   float cellSize = m_params.particleRadius * 2.0f;  // cell size equal to particle diameter
   m_params.cellSize = make_float3(cellSize, cellSize, cellSize);

   m_params.spring = 0.5f;
   m_params.damping = 0.02f;
   m_params.shear = 0.1f;
   m_params.attraction = 0.0f;
   m_params.boundaryDamping = -0.5f;

   m_params.gravity = make_float3(0.0f, -0.0003f, 0.0f);
   m_params.globalDamping = 1.0f;

   _initialize(numParticles);
}

ParticleSystem::~ParticleSystem()
{
   _finalize();
   m_numParticles = 0;
}

uint
ParticleSystem::createVBO(uint size)
{
   GLuint vbo;
   glGenBuffers(1, &vbo);
   glBindBuffer(GL_ARRAY_BUFFER, vbo);
   glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
   glBindBuffer(GL_ARRAY_BUFFER, 0);
   registerGLBufferObject(vbo);
   return vbo;
}

inline float lerp(float a, float b, float t)
{
   return a + t*(b-a);
}

// create a color ramp
void colorRamp(float t, float *r)
{
   const int ncolors = 7;
   float c[ncolors][3] = {
      { 1.0, 0.0, 0.0, },
      { 1.0, 0.5, 0.0, },
      { 1.0, 1.0, 0.0, },
      { 0.0, 1.0, 0.0, },
      { 0.0, 1.0, 1.0, },
      { 0.0, 0.0, 1.0, },
      { 1.0, 0.0, 1.0, },
   };
   t = t * (ncolors-1);
   int i = (int) t;
   float u = t - floor(t);
   r[0] = lerp(c[i][0], c[i+1][0], u);
   r[1] = lerp(c[i][1], c[i+1][1], u);
   r[2] = lerp(c[i][2], c[i+1][2], u);
}


/* molsettimestep.  Sets the rms step lengths according to the simulation time
step.  This may be called during setup or afterwards. */
inline void molSetTimestep(float* diffusionRates, float* diffusionCoefficients, float dt, int nTypes) 
{
   int i;
   for(i=0; i < nTypes; i++)
      diffusionRates[i] = sqrt(2.0f * diffusionCoefficients[i] * dt);
}

// oldCol, oldPos: mapped VBO. It will give unmap them.
void ParticleSystem::reallocateSpace(uint allocateParticles, uint prevNumParticles, float* oldPos, int* oldTypes)
{
   //int* dTypes;
   int posVBO, typesVBO;
   float* cudaPosVBO;
   int* cudaTypesVBO;

   //float* dSortedPos;
   //uint* dGridParticleHash;
   //uint* dGridParticleIndex


   //re-allocate CPU data
   delete[] m_hAdditionalPos;
   delete[] m_hAdditionalTypes;
   m_hAdditionalPos = new float [4 * allocateParticles];
   m_hAdditionalTypes = new int [allocateParticles];

   //re-allocate GPU data
   unsigned int memSize = sizeof(float) * 4 * allocateParticles;

   // create types (species) array
   //allocateArray((void**)&dTypes, sizeof(int) * allocateParticles);

   // temp space for compact operations
   freeArray(dCompactedTemp);
   allocateArray((void**)&dCompactedTemp, sizeof(int) * allocateParticles);   

   if (m_bUseOpenGL) 
   {
      posVBO = createVBO(memSize);    
      typesVBO = createVBO(sizeof(int) * allocateParticles);

      cudaPosVBO = (float *) mapGLBufferObject(posVBO);
      cudaTypesVBO = (int *) mapGLBufferObject(typesVBO);
   }
   else 
   {
      cutilSafeCall( cudaMalloc( (void **)&cudaPosVBO, memSize )) ;
      cutilSafeCall( cudaMalloc( (void **)&cudaTypesVBO, sizeof(int) * allocateParticles));
   }

   freeArray(m_dSortedPos);
   freeArray(m_dGridParticleHash);
   freeArray(m_dGridParticleIndex);

   freeArray(m_dAddedPos);
   freeArray(m_dAddedTypes);
   freeArray(m_dBirth);
   freeArray(m_dDeath);

   allocateArray((void**)&m_dSortedPos, memSize);
   allocateArray((void**)&m_dGridParticleHash, allocateParticles*sizeof(uint));
   allocateArray((void**)&m_dGridParticleIndex, allocateParticles*sizeof(uint));

   allocateArray((void**)&m_dAddedPos, memSize);
   allocateArray((void**)&m_dAddedTypes, allocateParticles*sizeof(int));
   allocateArray((void**)&m_dBirth, allocateParticles*sizeof(int));
   allocateArray((void**)&m_dDeath, allocateParticles*sizeof(int));


   // assume already compacted pos/color
   cudaMemcpy(cudaPosVBO, oldPos, prevNumParticles * 4 * sizeof(float), cudaMemcpyDeviceToDevice);
   cudaMemcpy(cudaTypesVBO, oldTypes, prevNumParticles * sizeof(int), cudaMemcpyDeviceToDevice);

   if (m_bUseOpenGL) 
   {
      unmapGLBufferObject(m_posVBO);
      unmapGLBufferObject(m_typesVBO);

      unregisterGLBufferObject(m_posVBO);
      unregisterGLBufferObject(m_typesVBO);
      glDeleteBuffers(1, (const GLuint*)&m_posVBO);
      glDeleteBuffers(1, (const GLuint*)&m_typesVBO);

      m_posVBO = posVBO;
      m_typesVBO = typesVBO;

      unmapGLBufferObject(posVBO);
      unmapGLBufferObject(typesVBO);
   }
   else
   {
      freeArray(oldPos);
      freeArray(oldTypes);
      m_cudaPosVBO = cudaPosVBO;
      m_cudaTypesVBO = cudaTypesVBO;
   }

   delete m_sorter;
   cudppDestroyPlan(scanHandle);
   

   // no need to copy, will be re-created
   //float* dSortedPos; 
   //uint* dGridParticleHash;
   //uint* dGridParticleIndex

   m_sorter = new RadixSort(allocateParticles);

   // Initialize scan
   //CUDPPConfiguration scanConfig;
   //scanConfig.algorithm = CUDPP_SCAN;
   //scanConfig.datatype  = CUDPP_INT;
   //scanConfig.op        = CUDPP_ADD;
   //scanConfig.options   = CUDPP_OPTION_EXCLUSIVE | CUDPP_OPTION_FORWARD;
   //cudppPlan(&m_cudppPlan, scanConfig, allocateParticles, 1, 0);

   CUDPPConfiguration config;
   config.algorithm    = CUDPP_SCAN;
   config.datatype     = CUDPP_INT;
   config.op           = CUDPP_ADD;
   config.options      = CUDPP_OPTION_FORWARD | CUDPP_OPTION_EXCLUSIVE;
   
   CUDPPResult result = cudppPlan(&scanHandle, config, allocateParticles, 1, 0);
   assert (result == CUDPP_SUCCESS);
}

void ParticleSystem::allocateSpace(uint allocateParticles)
{
   // allocate host storage    
   m_hPos = new float[allocateParticles*4];
   memset(m_hPos, 0, allocateParticles*4*sizeof(float));     
   
   m_hAdditionalPos = new float [4 * allocateParticles];
   m_hAdditionalTypes = new int [allocateParticles];

   // allocate GPU data
   unsigned int memSize = sizeof(float) * 4 * allocateParticles;

   // temp space for compact operations
   allocateArray((void**)&dCompactedTemp, sizeof(int) * allocateParticles);   

   if (m_bUseOpenGL) 
   {
      m_posVBO = createVBO(memSize);    
      m_typesVBO = createVBO(allocateParticles * sizeof(int));
   }
   else 
   {
      cutilSafeCall( cudaMalloc( (void **)&m_cudaPosVBO, memSize )) ;
      cutilSafeCall( cudaMalloc( (void **)&m_cudaTypesVBO, allocateParticles * sizeof(int)) );
   }


   allocateArray((void**)&m_dSortedPos, memSize);
   allocateArray((void**)&m_dGridParticleHash, allocateParticles*sizeof(uint));
   allocateArray((void**)&m_dGridParticleIndex, allocateParticles*sizeof(uint));
   
   allocateArray((void**)&m_dAddedPos, memSize);
   allocateArray((void**)&m_dAddedTypes, allocateParticles*sizeof(int));
   allocateArray((void**)&m_dDeath, allocateParticles*sizeof(int));
   allocateArray((void**)&m_dBirth, allocateParticles*sizeof(int));

   m_sorter = new RadixSort(allocateParticles);   

   // Initialize scan
   /*CUDPPConfiguration scanConfig;
   scanConfig.algorithm = CUDPP_SCAN;
   scanConfig.datatype  = CUDPP_INT;
   scanConfig.op        = CUDPP_ADD;
   scanConfig.options   = CUDPP_OPTION_EXCLUSIVE | CUDPP_OPTION_FORWARD;
   cudppPlan(&m_cudppPlan, scanConfig, allocateParticles, 1, 0);*/

   CUDPPConfiguration config;
   config.algorithm    = CUDPP_SCAN;
   config.datatype     = CUDPP_INT;
   config.op           = CUDPP_ADD;
   config.options      = CUDPP_OPTION_FORWARD | CUDPP_OPTION_EXCLUSIVE;
   
   CUDPPResult result = cudppPlan(&scanHandle, config, allocateParticles, 1, 0);
   assert (result == CUDPP_SUCCESS);
}

void ParticleSystem::test1Reactions()
{
   // TODO: set values really!
   m_numReactions = 3;

   // Create/allocate reactionList da structure
   reactionList = new Reaction;
   reactionList->order = new int[m_numReactions];
   reactionList->bindRadiusSquared  = new float[m_numReactions];
   reactionList->unbindRadius = new float[m_numReactions];
   reactionList->geminateProbability  = new float[m_numReactions];
   reactionList->prob  = new float[m_numReactions];
   reactionList->rate = new float[m_numReactions];
   reactionList->nprod = new int[m_numReactions];
   reactionList->products = new int2[m_numReactions];
   reactionList->reactants = new int2[m_numReactions];
   reactionList->productPositions = new float[m_numReactions * 6];

   int reactionsPerOrder[3];
   reactionsPerOrder[0] = 0;
   reactionsPerOrder[1] = 0;
   reactionsPerOrder[2] = 0;

   init_gen_rand(0);


   //TODO: fill arrays!!
   int rxn = 0;
   ++(reactionsPerOrder[1]);
   reactionList->order[rxn] = 1;
   reactionList->prob[rxn] = 1.0f;
   reactionList->rate[rxn] = 10;
   reactionList->nprod[rxn] = 2;
   reactionList->reactants[rxn] = make_int2(1, -1);
   reactionList->products[rxn] = make_int2(1, 1);
   reactionList->unbindRadius[rxn] = 0.0;
   
   ++rxn;
   ++(reactionsPerOrder[2]);
   reactionList->order[rxn] = 2;
   reactionList->prob[rxn] = 1.0f;
   reactionList->rate[rxn] = 2000;
   reactionList->nprod[rxn] = 2;
   reactionList->reactants[rxn] = make_int2(1, 2);
   reactionList->products[rxn] = make_int2(2, 2);
   reactionList->unbindRadius[rxn] = 0.0;

   ++rxn;
   ++(reactionsPerOrder[1]);
   reactionList->order[rxn] = 1;
   reactionList->prob[rxn] = 1.0f;
   reactionList->rate[rxn] = 10;
   reactionList->nprod[rxn] = 0;
   reactionList->reactants[rxn] = make_int2(2, -1);
   reactionList->products[rxn] = make_int2(-1, -1);
   reactionList->unbindRadius[rxn] = 0.0;

   // Initialize reactionOrderInfo
   fillReactionOrderInfo(reactionsPerOrder, m_numReactions,
                         reactionList, reactionOrderInfo, m_numTypes);
}

int* ParticleSystem::test1Types()
{
   ///////TODO
   // use a real time step!
   deltaTime = 0.1f;
   // usa a real volume!
   systemVolume = 10.0f; 

   m_numTypes = 3; //2 + 1 dummy

   int* types = new int[m_numParticles];

   m_hDiffusionCoefficients = new float[m_numTypes];
   m_hDiffusionCoefficients[0] = 0.0f; // for the dummy
   m_hDiffusionCoefficients[1] = 0.001f;
   m_hDiffusionCoefficients[2] = 0.001f;

   ///////
   
   for(uint i = 0; i < m_numParticles / 2; i++)   
      types[i] = 1;
   for(uint i = m_numParticles / 2; i < m_numParticles; i++) 
      types[i] = 2;

   return types;
}


void ParticleSystem::test2Reactions()
{
   //reaction fwd E + S -> ES 0.01
   //reaction back ES -> E + S 1
   //reaction prod ES -> E + P 1
   //product_placement back pgemmax 0.2

   // TODO: set values really!
   m_numReactions = 3;

   // Create/allocate reactionList da structure
   reactionList = new Reaction;
   reactionList->order = new int[m_numReactions];
   reactionList->bindRadiusSquared  = new float[m_numReactions];
   reactionList->unbindRadius = new float[m_numReactions];
   reactionList->geminateProbability  = new float[m_numReactions];
   reactionList->prob  = new float[m_numReactions];
   reactionList->rate = new float[m_numReactions];
   reactionList->nprod = new int[m_numReactions];
   reactionList->products = new int2[m_numReactions];
   reactionList->reactants = new int2[m_numReactions];
   reactionList->productPositions = new float[m_numReactions * 6];

   int reactionsPerOrder[3];
   reactionsPerOrder[0] = 0;
   reactionsPerOrder[1] = 0;
   reactionsPerOrder[2] = 0;

   init_gen_rand(0);


   //TODO: fill arrays!!
   int rxn = 0;

   //reaction fwd E + S -> ES 0.01
   ++(reactionsPerOrder[2]);
   reactionList->order[rxn] = 2;
   reactionList->prob[rxn] = 1.0f;
   reactionList->rate[rxn] = 0.01f;
   reactionList->nprod[rxn] = 1;
   reactionList->reactants[rxn] = make_int2(1, 2);
   reactionList->products[rxn] = make_int2(3, -1);
   reactionList->unbindRadius[rxn] = 0.0;
   
   ++rxn;
   //reaction back ES -> E + S 1
   ++(reactionsPerOrder[1]);
   reactionList->order[rxn] = 1;
   reactionList->prob[rxn] = 1.0f;
   reactionList->rate[rxn] = 1.0f;
   reactionList->nprod[rxn] = 2;
   reactionList->reactants[rxn] = make_int2(3, -1);
   reactionList->products[rxn] = make_int2(2, 2);
   reactionList->geminateProbability[rxn] = 0.2f;
   //reactionList->unbindRadius[rxn] = 0.0;

   ++rxn;
   //reaction prod ES -> E + P 1
   ++(reactionsPerOrder[1]);
   reactionList->order[rxn] = 1;
   reactionList->prob[rxn] = 1.0f;
   reactionList->rate[rxn] = 1.0f;
   reactionList->nprod[rxn] = 2;
   reactionList->reactants[rxn] = make_int2(3, -1);
   reactionList->products[rxn] = make_int2(1, 4);
   reactionList->unbindRadius[rxn] = 0.0;

   // Initialize reactionOrderInfo
   fillReactionOrderInfo(reactionsPerOrder, m_numReactions,
                         reactionList, reactionOrderInfo, m_numTypes);
}

int* ParticleSystem::test2Types()
{
   ///////TODO
   // use a real time step!
   deltaTime = 0.001f;
   // usa a real volume!
   systemVolume = 90.0f; 

   //E, S, ES, P
   m_numTypes = 5; //4 + 1 dummy

   int* types = new int[m_numParticles];

   m_hDiffusionCoefficients = new float[m_numTypes];
   m_hDiffusionCoefficients[0] = 0.0f; // for the dummy
   m_hDiffusionCoefficients[1] = 0.05f;
   m_hDiffusionCoefficients[2] = 0.05f;
   m_hDiffusionCoefficients[3] = 0.05f;
   m_hDiffusionCoefficients[4] = 0.05f;

   ///////
   int bound = (m_numParticles/11);
   
   for(uint i = 0; i < bound; i++)   
      types[i] = 1;
   for(uint i = bound; i < m_numParticles; i++) 
      types[i] = 2;

   return types;
}


void ParticleSystem::test3Reactions()
{
   // TODO: set values really!
   m_numReactions = 0;

   // Create/allocate reactionList da structure
   reactionList = new Reaction;
   reactionList->order = new int[m_numReactions];
   reactionList->bindRadiusSquared  = new float[m_numReactions];
   reactionList->unbindRadius = new float[m_numReactions];
   reactionList->geminateProbability  = new float[m_numReactions];
   reactionList->prob  = new float[m_numReactions];
   reactionList->rate = new float[m_numReactions];
   reactionList->nprod = new int[m_numReactions];
   reactionList->products = new int2[m_numReactions];
   reactionList->reactants = new int2[m_numReactions];
   reactionList->productPositions = new float[m_numReactions * 6];

   int reactionsPerOrder[3];
   reactionsPerOrder[0] = 0;
   reactionsPerOrder[1] = 0;
   reactionsPerOrder[2] = 0;

   init_gen_rand(0);
   

   // Initialize reactionOrderInfo
   fillReactionOrderInfo(reactionsPerOrder, m_numReactions,
                         reactionList, reactionOrderInfo, m_numTypes);
}

int* ParticleSystem::test3Types()
{
   ///////TODO
   // use a real time step!
   deltaTime = 0.01f;
   // usa a real volume!
   systemVolume = 90.0f; 

   m_numTypes = 3; //2 + 1 dummy

   int* types = new int[m_numParticles];

   m_hDiffusionCoefficients = new float[m_numTypes];
   m_hDiffusionCoefficients[0] = 0.0f; // for the dummy
   m_hDiffusionCoefficients[1] = 0.3f;
   m_hDiffusionCoefficients[2] = 0.1f;


   ///////
   int bound = 3 * (m_numParticles/4);
   
   for(uint i = 0; i < bound; i++)   
      types[i] = 1;
   for(uint i = bound; i < m_numParticles; i++) 
      types[i] = 2;

   return types;
}

void ParticleSystem::_initialize(int numParticles)
{
   assert(!m_bInitialized);

   m_numParticles = numParticles;

   allocateSpace(numParticles * 2);   
   m_allocatedParticles = numParticles * 2;

    m_hCellStart = new uint[m_numGridCells];
   memset(m_hCellStart, 0, m_numGridCells*sizeof(uint));

   m_hCellEnd = new uint[m_numGridCells];
   memset(m_hCellEnd, 0, m_numGridCells*sizeof(uint));


   // Create gaussian table, copy it to constant memory
   float h_gaussianLookupTable[gaussianTableDim];
   createGaussTable(h_gaussianLookupTable, gaussianTableDim);
   copyGaussianToConstant(h_gaussianLookupTable, gaussianTableDim);
    
   
   // TODO: read species and diffusionCoefficients from input!
   //int* types = test1Types();
   //int* types = test2Types();
   int* types = test3Types();

   if (m_bUseOpenGL) 
   {
      unregisterGLBufferObject(m_typesVBO);
      glBindBuffer(GL_ARRAY_BUFFER, m_typesVBO);
      glBufferSubData(GL_ARRAY_BUFFER, 0, numParticles * sizeof(int), types);
      glBindBuffer(GL_ARRAY_BUFFER, 0);
      registerGLBufferObject(m_typesVBO);
   }
   else
   {
      cudaMemcpy(m_cudaTypesVBO, types, m_numParticles * sizeof(int), cudaMemcpyHostToDevice);
   }

   delete [] types;

   float* diffusionRates = new float[m_numTypes];
   memset(diffusionRates, 0, m_numTypes*sizeof(float));


   // create diffuse array, will be bound to a texture in integrateSystem (particleSystem.cu)
   allocateArray((void**)&m_dDiffusionRates, m_numTypes * sizeof(float));
   allocateArray((void**)&m_dDiffusionCoefficients, m_numTypes * sizeof(float));

   cudaMemcpy(m_dDiffusionCoefficients, m_hDiffusionCoefficients, m_numTypes * sizeof(float), cudaMemcpyHostToDevice);

   molSetTimestep(diffusionRates, m_hDiffusionCoefficients, deltaTime, m_numTypes);
   cudaMemcpy(m_dDiffusionRates, diffusionRates, m_numTypes * sizeof(float), cudaMemcpyHostToDevice);
   delete[] diffusionRates;


   // REACTIONS
   //test1Reactions();
   //test2Reactions();
   test3Reactions();

   for (int i = 0; i < m_numReactions; ++i)
   {
      rxnsetrate(i, reactionList, this);
      rxnsetproduct(this, reactionList, reactionList->order[i], i); //TODO: verify the order parameter
   }

   //copy rxnss[1].numberOfReactionsPerType and rxnss[1].localToGlobalReaction to GPU memory
   allocateArray((void**)&m_dNumberOfReactionsPerType1, m_numTypes * sizeof(int));
   CUDA_SAFE_CALL(cudaMemcpy(m_dNumberOfReactionsPerType1, reactionOrderInfo[1].numberOfReactionsPerType, m_numTypes * sizeof(int), cudaMemcpyHostToDevice));
   allocateArray((void**)&m_dLocalToGlobalReaction1, reactionOrderInfo[1].localToGlobalSize * sizeof(int));
   CUDA_SAFE_CALL(cudaMemcpy(m_dLocalToGlobalReaction1, reactionOrderInfo[1].localToGlobal, reactionOrderInfo[1].localToGlobalSize * sizeof(int), cudaMemcpyHostToDevice));
   allocateArray((void**)&m_dReactionsPerTypeIdx1, m_numTypes * sizeof(int));
   CUDA_SAFE_CALL(cudaMemcpy(m_dReactionsPerTypeIdx1, reactionOrderInfo[1].reactionsPerTypeIdx, m_numTypes * sizeof(int), cudaMemcpyHostToDevice));

   cudaMalloc((void**)&m_dReactionList, sizeof(Reaction));
   //allocateArray((void**)&m_dReactionTable, m_numReactions * m_numReactions * sizeof(int));//??
   allocateArray((void**)&m_dReactionTable, m_numTypes * m_numTypes * sizeof(int));

   //fill the reaction table
   int* reactionTable = new int[m_numTypes * m_numTypes];
   memset(reactionTable, -1, m_numTypes * m_numTypes * sizeof(int));
   for (int i = 0; i < m_numReactions; ++i)
   {
      if (reactionList->order[i] == 2)
      {
         int2 reacts = reactionList->reactants[i];
         assert(reacts.x != -1);
         assert(reacts.y != -1);

         int idx1 = m_numTypes * reacts.x + reacts.y;
         int idx2 = m_numTypes * reacts.y + reacts.x;

         reactionTable[idx1] = i;
         reactionTable[idx2] = i;
      }
   }
   //copy it
   cudaMemcpy(m_dReactionTable, reactionTable, m_numTypes * m_numTypes * sizeof(int), cudaMemcpyHostToDevice);
   delete[] reactionTable;

   Reaction gpuReactionStruct;

   allocateArray((void**)&(gpuReactionStruct.order), m_numReactions * sizeof(int));
   allocateArray((void**)&(gpuReactionStruct.bindRadiusSquared), m_numReactions * sizeof(float));
   allocateArray((void**)&(gpuReactionStruct.unbindRadius), m_numReactions * sizeof(float));
   allocateArray((void**)&(gpuReactionStruct.geminateProbability), m_numReactions * sizeof(float));
   allocateArray((void**)&(gpuReactionStruct.prob), m_numReactions * sizeof(float));
   allocateArray((void**)&(gpuReactionStruct.rate), m_numReactions * sizeof(float));
   allocateArray((void**)&(gpuReactionStruct.nprod), m_numReactions * sizeof(int));
   allocateArray((void**)&(gpuReactionStruct.products), m_numReactions * sizeof(int2));
   allocateArray((void**)&(gpuReactionStruct.reactants), m_numReactions * sizeof(int2));
   allocateArray((void**)&(gpuReactionStruct.productPositions), m_numReactions * sizeof(float) * 6);

   //REACTIONS copy arrays
   cudaMemcpy(gpuReactionStruct.order, reactionList->order, m_numReactions * sizeof(int), cudaMemcpyHostToDevice);
   cudaMemcpy(gpuReactionStruct.bindRadiusSquared, reactionList->bindRadiusSquared, m_numReactions * sizeof(float), cudaMemcpyHostToDevice);
   cudaMemcpy(gpuReactionStruct.unbindRadius, reactionList->unbindRadius, m_numReactions * sizeof(float), cudaMemcpyHostToDevice);
   cudaMemcpy(gpuReactionStruct.geminateProbability, reactionList->geminateProbability, m_numReactions * sizeof(float), cudaMemcpyHostToDevice);
   cudaMemcpy(gpuReactionStruct.prob,reactionList->prob, m_numReactions * sizeof(float), cudaMemcpyHostToDevice);
   cudaMemcpy(gpuReactionStruct.rate,reactionList->rate, m_numReactions * sizeof(float), cudaMemcpyHostToDevice);
   cudaMemcpy(gpuReactionStruct.nprod,reactionList->nprod, m_numReactions * sizeof(int), cudaMemcpyHostToDevice);
   cudaMemcpy(gpuReactionStruct.products,reactionList->products, m_numReactions * sizeof(int2), cudaMemcpyHostToDevice);
   cudaMemcpy(gpuReactionStruct.reactants,reactionList->reactants, m_numReactions * sizeof(int2), cudaMemcpyHostToDevice);
   cudaMemcpy(gpuReactionStruct.productPositions,reactionList->productPositions, m_numReactions * sizeof(float) * 6, cudaMemcpyHostToDevice);

   cudaMemcpy(m_dReactionList, &gpuReactionStruct, sizeof(Reaction), cudaMemcpyHostToDevice);


   // initialize the RNG
   allocateArray((void**)&m_dRngStateArray, sizeof(MersenneTwisterState) * MT_RNG_COUNT);
   initializeMT(m_dRngStateArray);



   allocateArray((void**)&m_dCellStart, m_numGridCells*sizeof(uint));
   allocateArray((void**)&m_dCellEnd, m_numGridCells*sizeof(uint));


   //fill color buffer
   //glBindBufferARB(GL_ARRAY_BUFFER, m_colorVBO);
   //float *data = (float *) glMapBufferARB(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
   //float *ptr = data;
   //for(uint i=0; i<m_numParticles; i++) 
   //{
   //   //float t = i / (float) m_numParticles;

   //   //*ptr++ = rand() / (float) RAND_MAX;
   //   //*ptr++ = rand() / (float) RAND_MAX;
   //   //*ptr++ = rand() / (float) RAND_MAX;

   //   //colorRamp(t, ptr);
   //   //ptr+=3;

   //   if (types[i] == 0)
   //   {
   //      *ptr++ = 1.0;
   //      *ptr++ = 0.0;
   //      *ptr++ = 0.0;
   //   }
   //   else
   //   {               
   //      *ptr++ = 0.0;
   //      *ptr++ = 1.0;
   //      *ptr++ = 0.0;            
   //   }            

   //   *ptr++ = 1.0f;
   //}
   //glUnmapBufferARB(GL_ARRAY_BUFFER);
   //delete[] types;

   cutilCheckError(cutCreateTimer(&m_timer));

   setParameters(&m_params);
   
   m_bInitialized = true;
}

void ParticleSystem::_finalize()
{
   assert(m_bInitialized);

   delete [] m_hPos;
   delete [] m_hAdditionalPos;
   delete [] m_hAdditionalTypes;
   delete [] m_hCellStart;
   delete [] m_hCellEnd;
   delete [] m_hDiffusionCoefficients;

   //freeArray(m_dTypes);
   freeArray(dCompactedTemp);
   freeArray(m_dSortedPos);
   freeArray(m_dDiffusionRates);
   freeArray(m_dDiffusionCoefficients);
   freeArray(m_dRngStateArray);

   freeArray(m_dAddedPos);
   freeArray(m_dAddedTypes);
   freeArray(m_dDeath);
   freeArray(m_dBirth);

   freeArray(m_dGridParticleHash);
   freeArray(m_dGridParticleIndex);
   freeArray(m_dCellStart);
   freeArray(m_dCellEnd);

   if (m_bUseOpenGL) 
   {
      unregisterGLBufferObject(m_posVBO);
      unregisterGLBufferObject(m_typesVBO);
      glDeleteBuffers(1, (const GLuint*)&m_posVBO);
      glDeleteBuffers(1, (const GLuint*)&m_typesVBO);
   } 
   else
   {
      cutilSafeCall( cudaFree(m_cudaPosVBO) );
      cutilSafeCall( cudaFree(m_cudaTypesVBO) );
   }

   cudppDestroyPlan(scanHandle);
   delete m_sorter;



   delete[] reactionList->order;
   delete[] reactionList->bindRadiusSquared;
   delete[] reactionList->unbindRadius;
   delete[] reactionList->geminateProbability;
   delete[] reactionList->prob;
   delete[] reactionList->rate;
   delete[] reactionList->nprod;
   delete[] reactionList->products;
   delete[] reactionList->reactants;
   delete[] reactionList->productPositions;

   delete reactionList;

   freeArray(m_dReactionTable);

   freeArray(m_dReactionList->order);
   freeArray(m_dReactionList->bindRadiusSquared);
   freeArray(m_dReactionList->unbindRadius);
   freeArray(m_dReactionList->geminateProbability);
   freeArray(m_dReactionList->prob);
   freeArray(m_dReactionList->rate);
   freeArray(m_dReactionList->nprod);
   freeArray(m_dReactionList->products);
   freeArray(m_dReactionList->reactants);
   freeArray(m_dReactionList->productPositions);

   cudaFree(m_dReactionList);
}

// step the simulation
bool ParticleSystem::update()
{
   assert(m_bInitialized);

   // zeroth order -> on CPU   
   
   uint zerothNum = 0;
   if (reactionOrderInfo[0].reactionsPerOrder > 0)
      zerothNum = zeroReact(this, m_hAdditionalPos, m_hAdditionalTypes);

   float *dPos;
   int* dTypes;

   if (m_bUseOpenGL)
   {
      dPos = (float *) mapGLBufferObject(m_posVBO);
      dTypes = (int *) mapGLBufferObject(m_typesVBO);
   }
   else
   {
      dPos = m_cudaPosVBO;
      dTypes = m_cudaTypesVBO;
   }

   // update constants
   setParameters(&m_params);

   // diffuse and first order
   integrateSystem(m_dReactionList, 
      m_dLocalToGlobalReaction1,
      m_dReactionsPerTypeIdx1,
      m_dNumberOfReactionsPerType1,
      reactionOrderInfo[1].reactionsPerOrder,
      dPos, // lenght = numParticles
      dTypes, 
      m_dAddedPos,
      m_dAddedTypes,
      m_dBirth,
      m_dDeath,
      m_dDiffusionRates, // lenght = numTypes
      m_dRngStateArray,
      deltaTime,
      m_numParticles,
      m_numTypes);

   
   size_t leftAfterMono = m_numParticles;
   size_t addedByMono = 0;

   //cudaError_t err = cudaGetLastError();
   
   
   if (reactionOrderInfo[1].reactionsPerOrder > 0)
   {
      // obtain a compacted sum-prefix for the added elements
      cudppScan(scanHandle, dCompactedTemp, m_dBirth, m_numParticles);
      addedByMono = compactPosAndType(dCompactedTemp, m_dBirth, 
         m_dAddedPos, m_dAddedTypes, m_numParticles);

      cudppScan(scanHandle, dCompactedTemp, m_dDeath, m_numParticles);
      leftAfterMono = compactPosAndType(dCompactedTemp, m_dDeath, 
         dPos, dTypes, m_numParticles);

      if (leftAfterMono == 0)
         return false;
   }


   adjustPositions(dPos,           
                   dTypes,      
                   leftAfterMono,
                   m_params.particleRadius);

   size_t totalAfterBim = leftAfterMono;   

   if (reactionOrderInfo[2].reactionsPerOrder > 0)
   {
      // calculate grid hash
      calcHash(
         m_dGridParticleHash,
         m_dGridParticleIndex,
         dPos,
         leftAfterMono);

      // sort particles based on hash
      m_sorter->sort(m_dGridParticleHash, m_dGridParticleIndex, leftAfterMono, m_gridSortBits);

      // reorder particle arrays into sorted order and
      // find start and end of each cell
      reorderDataAndFindCellStart(
         m_dCellStart,
         m_dCellEnd,
         m_dSortedPos,
         //m_dSortedVel,
         m_dGridParticleHash,
         m_dGridParticleIndex,
         dPos,
         //m_dVel,
         leftAfterMono,
         m_numGridCells);

      // Reinitialize: no one dies
      cudaMemset(m_dDeath, -1, leftAfterMono * sizeof(int));
   
      // process collisions
      collide(dPos,
              m_dSortedPos,
              m_dGridParticleIndex,
              dTypes,
              m_dDeath, 
              m_dDiffusionCoefficients, 
              m_dReactionTable, 
              m_dReactionList, 
              m_dCellStart,
              m_dCellEnd,
              leftAfterMono,
              m_numGridCells, 
              m_numTypes,
              m_dRngStateArray);

      // also after collide?
      // adjustPositions()

      // cudpp scan on dDeath
      // then, compact (on the GPU, to avoid copy back-forth)      
      cudppScan(scanHandle, dCompactedTemp, m_dDeath, leftAfterMono);

      // Here we can only lose something (A + B -> C), putting the B index to 0. So, we compact on pos and types
      // directly.
      totalAfterBim = compactPosAndType(dCompactedTemp, m_dDeath, dPos, dTypes, leftAfterMono);
   }
  
   size_t newNumParticles = totalAfterBim + zerothNum + addedByMono;

   if (newNumParticles == 0)
      return false;

   m_numParticles = newNumParticles;

   if (newNumParticles > MAX_PARTICLES)
      return false;

   if (newNumParticles > m_allocatedParticles)
   {
      m_allocatedParticles = m_numParticles * 2; //should perform well
      reallocateSpace(m_allocatedParticles, totalAfterBim, dPos, dTypes);
      
      // re-map
      if (m_bUseOpenGL)
      {
         dPos = (float *) mapGLBufferObject(m_posVBO);
         dTypes = (int *) mapGLBufferObject(m_typesVBO);
      }
   }

   // Copy m_dAddedPos (mono) and m_dTypes at the end of those got back from bim
   cudaMemcpy(dPos + (totalAfterBim * 4), m_dAddedPos, (addedByMono * 4 * sizeof(float)), cudaMemcpyDeviceToDevice);
   cudaMemcpy(dTypes + totalAfterBim, m_dAddedTypes, addedByMono * sizeof(int), cudaMemcpyDeviceToDevice);


   // Copy additionalPos (zeroth)
   cudaMemcpy(dPos + ((totalAfterBim + addedByMono) * 4), m_hAdditionalPos, (zerothNum * 4 * sizeof(float)), cudaMemcpyHostToDevice);
   cudaMemcpy(dTypes + totalAfterBim + addedByMono, m_hAdditionalTypes, zerothNum * sizeof(int), cudaMemcpyHostToDevice);  

   // note: do unmap at end here to avoid unnecessary graphics/CUDA context switch
   if (m_bUseOpenGL) 
   {
      unmapGLBufferObject(m_posVBO);
      unmapGLBufferObject(m_typesVBO);
   }

   return true;
}

void ParticleSystem::dumpGrid()
{
   // dump grid information
   copyArrayFromDevice(m_hCellStart, m_dCellStart, 0, sizeof(uint)*m_numGridCells);
   copyArrayFromDevice(m_hCellEnd, m_dCellEnd, 0, sizeof(uint)*m_numGridCells);
   uint maxCellSize = 0;
   for(uint i=0; i<m_numGridCells; i++) {
      if (m_hCellStart[i] != 0xffffffff) {
         uint cellSize = m_hCellEnd[i] - m_hCellStart[i];
         //            printf("cell: %d, %d particles\n", i, cellSize);
         if (cellSize > maxCellSize) maxCellSize = cellSize;
      }
   }
   printf("maximum particles per cell = %d\n", maxCellSize);
}

void
ParticleSystem::dumpParticles(uint start, uint count)
{
   // debug
   copyArrayFromDevice(m_hPos, 0, m_posVBO, sizeof(float)*4*count);

   for(uint i=start; i<start+count; i++) {
      //        printf("%d: ", i);
      printf("pos: (%.4f, %.4f, %.4f, %.4f)\n", m_hPos[i*4+0], m_hPos[i*4+1], m_hPos[i*4+2], m_hPos[i*4+3]);
   }
}

float* 
ParticleSystem::getArray(ParticleArray array)
{
   assert(m_bInitialized);

   float* hdata = 0;
   float* ddata = 0;

   unsigned int vbo = 0;

   switch (array)
   {
   default:
   case POSITION:
      hdata = m_hPos;
      ddata = m_dPos;
      vbo = m_posVBO;
      break;
   case VELOCITY:
      break;
   }

   copyArrayFromDevice(hdata, ddata, vbo, m_numParticles*4*sizeof(float));
   return hdata;
}

void
ParticleSystem::setArray(ParticleArray array, const float* data, int start, int count)
{
   assert(m_bInitialized);

   switch (array)
   {
   default:
   case POSITION:
      {
         if (m_bUseOpenGL) {
            unregisterGLBufferObject(m_posVBO);
            glBindBuffer(GL_ARRAY_BUFFER, m_posVBO);
            glBufferSubData(GL_ARRAY_BUFFER, start*4*sizeof(float), count*4*sizeof(float), data);
            glBindBuffer(GL_ARRAY_BUFFER, 0);
            registerGLBufferObject(m_posVBO);
         }
      }
      break;
   case VELOCITY:
      //copyArrayToDevice(m_dVel, data, start*4*sizeof(float), count*4*sizeof(float));
      break;
   }       
}

inline float frand()
{
   return rand() / (float) RAND_MAX;
}

void
ParticleSystem::initGrid(uint *size, float spacing, float jitter, uint numParticles)
{
   srand(1973);

   float zspacing = 2.0f / (size[2] - 2);
   float yspacing = 2.0f / (size[1] - 2);
   float xspacing = 2.0f / (size[0] - 2);

   for(uint z=0; z<size[2]; z++) {
      for(uint y=0; y<size[1]; y++) {
         for(uint x=0; x<size[0]; x++) {
            uint i = (z*size[1]*size[0]) + (y*size[0]) + x;
            if (i < numParticles) {
               m_hPos[i*4] = (xspacing * x) + m_params.particleRadius - 1.0f + (frand()*2.0f-1.0f)*jitter;
               m_hPos[i*4+1] = (yspacing * y) + m_params.particleRadius - 1.0f + (frand()*2.0f-1.0f)*jitter;
               m_hPos[i*4+2] = (zspacing * z) + m_params.particleRadius - 1.0f + (frand()*2.0f-1.0f)*jitter;
               m_hPos[i*4+3] = 1.0f;

               //TODO: set particle types (species)

               //m_hVel[i*4] = 0.0f;
               //m_hVel[i*4+1] = 0.0f;
               //m_hVel[i*4+2] = 0.0f;
               //m_hVel[i*4+3] = 0.0f;
            }
         }
      }
   }
}

void
ParticleSystem::reset(ParticleConfig config)
{
   switch(config)
   {
   default:
   case CONFIG_RANDOM:
      {
         int p = 0;
         for(uint i=0; i < m_numParticles; i++) 
         {
            float point[3];
            point[0] = frand();
            point[1] = frand();
            point[2] = frand();
            m_hPos[p++] = 2 * (point[0] - 0.5f);
            m_hPos[p++] = 2 * (point[1] - 0.5f);
            m_hPos[p++] = 2 * (point[2] - 0.5f);
            m_hPos[p++] = 1.0f; // radius
            //m_hVel[v++] = 0.0f;
            //m_hVel[v++] = 0.0f;
            //m_hVel[v++] = 0.0f;
            //m_hVel[v++] = 0.0f;
         }
      }
      break;

   case CONFIG_GRID:
      {
         float jitter = m_params.particleRadius*0.01f;
         uint s = (int) ceilf(powf((float) m_numParticles, 1.0f / 3.0f));
         uint gridSize[3];
         gridSize[0] = gridSize[1] = gridSize[2] = s;

         //float spacing = 2.0 / (float) m_numParticles;
         float spacing = m_params.particleRadius*2.0f;
         initGrid(gridSize, spacing, jitter, m_numParticles);
      }
      break;
   }

   setArray(POSITION, m_hPos, 0, m_numParticles);
   //setArray(VELOCITY, m_hVel, 0, m_numParticles);
}
