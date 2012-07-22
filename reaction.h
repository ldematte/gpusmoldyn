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

#ifndef REACTION_H_INCLUDED
#define REACTION_H_INCLUDED

#pragma warning(disable:4201)
#pragma warning(disable:4408)

#include "vector_types.h"
#include <list>

struct Reaction
{
   int* order;
   float* bindRadiusSquared;
   float* unbindRadius;
   float* geminateProbability;
   float* prob;
   float* rate; //unit: molecules*volume^(order–1)/time
   int* nprod;
   int2* products;  //[2];
   int2* reactants; //[2];
   float* productPositions; //[6];
};

//One for reaction order
//CPU struct
struct ReactionOrderInfo //rxnsuperstruct
{
   int order;
   int reactionsPerOrder;

   int* reactionsPerTypeIdx;      // where is reactant i inside the localToGlobal table
   int* localToGlobal;            // reaction number for i -> reaction number (global) (table)
                                  // WARNING: in Smoldyn this is per-order, and used as an index inside the reactions (rxn) array
                                  // here is directly an index to the global Reaction array.
   int localToGlobalSize;

   int* numberOfReactionsPerType; // number of reactions for reactant i (rxnsuperstruct->nrxn)
   int* reactionIndexes;          // reaction indexes in global list for this order (substitutes rxn for zero-th order)
   //Reaction* reactions          // pointers to reactions in global list (rxn - we use the Reaction array directly instead)
};

#endif //REACTION_H_INCLUDED

