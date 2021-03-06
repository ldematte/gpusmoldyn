/*
 * GPU Smoldyn: Smoldyn algorithm ported to the GPU using CUDA 2.2
 * Writtern By Lorenzo Dematt�, 2010-2011
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

#ifndef SMOLREACT_H_INCLUDED
#define SMOLREACT_H_INCLUDED

#include "particleSystem.h"

int rxnsetrate(int rxn, Reaction* reactionList, ParticleSystem* sim);
int rxnsetproduct(ParticleSystem* sim, Reaction* reactionList, int order, int rxn);
void fillReactionOrderInfo(int* reactionsPerOrder,
                           int numReactions,
                           Reaction* reactions, 
                           ReactionOrderInfo* rxnss, 
                           int maxSpecies);

#endif //SMOLREACT_H_INCLUDED
