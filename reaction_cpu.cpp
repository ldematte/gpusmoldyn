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
 
#include "particleSystem.h"
#include "stats.h"
#include <list>

// CPU??

/* zeroreact.  Figures out how many molecules to create for each zeroth order
reaction and then tells doreact to create them.  It returns 0 for success or 1
if not enough molecules were allocated initially. */
int zeroReact(ParticleSystem* sim, float* newPos, int* newTypes)
{
   int p = 0;
	float3 pos;

	ReactionOrderInfo& rxnss = sim->reactionOrderInfo[0];
   Reaction* reactionList = sim->reactionList;

   for(int i = 0; i < rxnss.reactionsPerOrder; ++i) 
   {
      // Get the (global) reaction number for the I-th reaction in this order (0)
      int rxn = rxnss.reactionIndexes[i]; //OK!!
      int nmol = poisrandF(reactionList->prob[rxn]);
		for(int j = 0; j < nmol; ++j) 
      {
			sim->randPos(pos);
         int type = reactionList->products[rxn].x;
         newPos[p * 4] = pos.x;
         newPos[p * 4 + 1] = pos.y;
         newPos[p * 4 + 2] = pos.z;
         newPos[p * 4 + 3] = 1;

         newTypes[p] = type;
         ++p;
      }
   }
   return p; 
}

// CPU??
/* unireact.  Identifies and performs all unimolecular reactions.  Reactions
that should occur are sent to doreact to process them.  The function returns 0
for success or 1 if not enough molecules were allocated initially. */
//int unireact(ParticleSystem* sim) 
//{
//	ReactionOrderInfo& rxnss = sim->reactionOrderInfo[1];
//   Reaction* reactionList = sim->reactionList;
//
//   int* nrxn = rxnss.numberOfReactionsPerType;
//   int** table = rxnss.localToGlobalReaction;
//
//   //int ntypes = sim->getNumTypes();
//   int nParticles = sim->getNumParticles();
//
//	for (int p = 0; p < nParticles; ++p) 
//   {
//      // type of this particle
//      int type = sim->types[p]; // TODO: is on GPU!!
//
//      if (type <= 0)
//         break;
//
//      // scroll the list of reactions for that particle
//      for (int j=0; j < nrxn[type]; ++j) 
//      {
//         int rxn = table[type][j];
//         if (coinrandD(reactionList->prob[rxn])) {
//            // do reaction
//            break; //j=nrxn[i];
//         }
//      }
//   }
//							 
//	return 0;
//}


/* morebireact.  Given a probable reaction from bireact, this orders the
reactants, checks for reaction permission, moves a reactant in case of periodic
boundaries, increments the appropriate event counter, and calls doreact to
perform the reaction.  The return value is 0 for success (which may include no
reaction) and 1 for failure. */



