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
 
#include "rxnparam.h"
#include "reaction.h"
#include "particleSystem.h"
#include <math.h>
#include <assert.h>

#define MSMAX 5
#define MSMAX1 6
#define MAXORDER 3

/// superstructure stuff
//////////////////////////////////////////

int Zn_permute(int *a,int *b,int n,int k) 
{
	int ans;

	if(n==0) ans=0;

	else if(n==1) {b[0]=a[0];ans=0; }

	else if(n==2) {
		if(k==0) {b[0]=a[0];b[1]=a[1];ans=1; }
		else     {b[0]=a[1];b[1]=a[0];ans=0; }
		if(a[0]==a[1]) ans=0; }

	else if(n==3) {
		if(k==0)      {b[0]=a[0];b[1]=a[1];b[2]=a[2];ans=1;}
		else if(k==1) {b[0]=a[0];b[1]=a[2];b[2]=a[1];ans=2;}
		else if(k==2) {b[0]=a[1];b[1]=a[0];b[2]=a[2];ans=3;}
		else if(k==3) {b[0]=a[1];b[1]=a[2];b[2]=a[0];ans=4;}
		else if(k==4) {b[0]=a[2];b[1]=a[0];b[2]=a[1];ans=5;}
		else          {b[0]=a[2];b[1]=a[1];b[2]=a[0];ans=0;}
		if(a[1]==a[2]&&ans==1) ans=2;
		else if(a[1]==a[2]&&(ans==4||ans==5)) ans=0;
		if(a[0]==a[1]&&(ans==2||ans==3)) ans=4;
		else if(a[0]==a[1]&&ans==5) ans=0;
		if(a[0]==a[2]&&(ans==3||ans==4||ans==5)) ans=0; 
   }

	else ans=-1;

	return ans; 
}


bool findreverserxn(ParticleSystem* sim, Reaction* reactionList, int order, int rxn, int& reverseOrder, int& reverseIndex) 
{
	if (sim == NULL || order<0 || order > 2 || rxn < 0) 
      assert(false);

   ReactionOrderInfo* rxnss = &(sim->reactionOrderInfo[order]);


	reverseOrder = 0;
   reverseIndex = 0;

   if (order == 0 || reactionList->nprod[rxn] < 1 || sim->reactionOrderInfo[reactionList->nprod[rxn]].reactionsPerOrder == 0) 
   {
      return false;
   }
	else 
   {
		reverseOrder = reactionList->nprod[rxn];
		ReactionOrderInfo* rxnssr = &(sim->reactionOrderInfo[reverseOrder]);
      

      // Three cases: 
      // A + B -> C     (reverse: C -> A + B)
      // A + B -> C + D (reverse: C + D -> A + B)
      // A -> C + D     (reverse: C + D -> A)

      // Pick "C"
      int firstProductId = reactionList->products[rxn].x;
		

      for(int j = 0; j < rxnssr->numberOfReactionsPerType[firstProductId]; j++) // nrxn number of reactions for reactant i
      { 
         int baseIdx = rxnssr->reactionsPerTypeIdx[firstProductId];
         int reverseGlobalId = rxnssr->localToGlobal[baseIdx + j]; //reactant firstProductId, reaction number for i -> reaction number (global)
         
         if (reverseOrder > 1)
         {
            // Among reactats we need the original product "D"         
            if (!(reactionList->reactants[reverseGlobalId].x == reactionList->products[rxn].x &&
                 reactionList->reactants[reverseGlobalId].y == reactionList->products[rxn].y) &&
                !(reactionList->reactants[reverseGlobalId].x == reactionList->products[rxn].y &&
                 reactionList->reactants[reverseGlobalId].y == reactionList->products[rxn].x))
            {
               continue;
            }
         }

         // Among product(s), we want the original reactant(s)
         if (order == 1)
         {
            if (reactionList->products[reverseGlobalId].x == reactionList->reactants[rxn].x)
            {
               // ok, found it
               reverseIndex = reverseGlobalId;
               return true;
            }
         }  
         else
         {
            assert (order == 2);
            if ((reactionList->products[reverseGlobalId].x == reactionList->reactants[rxn].x &&
                 reactionList->products[reverseGlobalId].y == reactionList->reactants[rxn].y) ||
                (reactionList->products[reverseGlobalId].x == reactionList->reactants[rxn].y &&
                 reactionList->products[reverseGlobalId].y == reactionList->reactants[rxn].x))
            {
               // ok, found it
               reverseIndex = reverseGlobalId;
               return true;
            }
         }
      }
   }
   return false;
}





/* rxnpackident.  Packs a list of order identities that are listed in ident into
a single value, which is returned.  maxspecies is the maximum number of
identities, from either the reaction superstructure or the simulation structure.
*/
int rxnpackident(int order,int maxspecies,int *ident) 
{
	if(order==0) return 0;
	if(order==1) return ident[0];
	if(order==2) return ident[0]*maxspecies+ident[1];
	return 0; 
}

/* RxnAddReaction.  Adds a reaction to the simulation, including all necessary
memory allocation.  rname is the name of the reaction, order is the order of the
reaction, and nprod is the number of products.  rctident and rctstate are
vectors of size order that contain the reactant identities and states,
respectively.  Likewise, prdident and prdstate are vectors of size nprod that
contain the product identities and states.  This returns the just added reaction
for success and NULL for inability to allocate memory.  This allocates reaction
superstuctures and reaction structures, and will enlarge any array, as needed. */
// Here only superstructure (rxnssptr / reactionOrderInfo)
void fillReactionOrderInfo(int* reactionsPerOrder,
                           int numReactions,
                           Reaction* reactions, 
                           ReactionOrderInfo* rxnss, 
                           int maxSpecies)
{
   //temporary storage
   int* reactionsPerTypeIdxTemp[3];
   

   // Allocate space
   for (int iOrder = 0; iOrder < 3; ++iOrder)
   {
      rxnss[iOrder].numberOfReactionsPerType = new int[maxSpecies];
      rxnss[iOrder].reactionsPerTypeIdx = new int[maxSpecies]; //compacted reaction index array
      for (int j = 0; j < maxSpecies; ++j)
         rxnss[iOrder].numberOfReactionsPerType[j] = 0;

      rxnss[iOrder].reactionsPerOrder = 0;
      reactionsPerTypeIdxTemp[iOrder] = new int[maxSpecies];

      rxnss[iOrder].reactionIndexes = new int[reactionsPerOrder[iOrder]];
   }

   // Step 1: count number of reactions per type   
   for (int iReaction = 0; iReaction < numReactions; ++iReaction)
   {
      int iOrder = reactions->order[iReaction];

      rxnss[iOrder].reactionIndexes[rxnss[iOrder].reactionsPerOrder] = iReaction;
      ++(rxnss[iOrder].reactionsPerOrder);

      if(iOrder > 0) 
      {	
         int2 rctident = reactions->reactants[iReaction];

         // First reactant
         int reactIdent = rctident.x;            
         ++(rxnss[iOrder].numberOfReactionsPerType[reactIdent]);
         
         //Second reactant (we don't really need it: this structure is only for first order reactions)
         if (iOrder == 2)
         {
            reactIdent = rctident.y;
            ++(rxnss[iOrder].numberOfReactionsPerType[reactIdent]); 
         }
      }
   }

   //Step 2: compact the index array, allocate table localToGlobalReaction space
   for (int iOrder = 0; iOrder < 3; ++iOrder)
   {
      rxnss[iOrder].reactionsPerTypeIdx[0] = 0;
      reactionsPerTypeIdxTemp[iOrder][0] = 0;
      for (int j = 1; j < maxSpecies; ++j)
      {
         rxnss[iOrder].reactionsPerTypeIdx[j] = rxnss[iOrder].reactionsPerTypeIdx[j - 1] + rxnss[iOrder].numberOfReactionsPerType[j - 1];
         reactionsPerTypeIdxTemp[iOrder][j] = rxnss[iOrder].reactionsPerTypeIdx[j];
      }

      rxnss[iOrder].localToGlobalSize = rxnss[iOrder].reactionsPerTypeIdx[maxSpecies - 1] + rxnss[iOrder].numberOfReactionsPerType[maxSpecies - 1];
      rxnss[iOrder].localToGlobal = new int[rxnss[iOrder].localToGlobalSize];
   }

   // Step 3: fill localToGlobal
   for (int iReaction = 0; iReaction < numReactions; ++iReaction)
   {
      int iOrder = reactions->order[iReaction];

      if(iOrder > 0) 
      {	
         int2 rctident = reactions->reactants[iReaction];

         // First reactant
         int reactIdent = rctident.x;            
         // get index
         int idx = reactionsPerTypeIdxTemp[iOrder][reactIdent];
         rxnss[iOrder].localToGlobal[idx] = iReaction;
         ++(reactionsPerTypeIdxTemp[iOrder][reactIdent]);
         
         //Second reactant (we don't really need it: this structure is only for first order reactions)
         if (iOrder == 2)
         {
            reactIdent = rctident.y;
            idx = reactionsPerTypeIdxTemp[iOrder][reactIdent];
            rxnss[iOrder].localToGlobal[idx] = iReaction;
            ++(reactionsPerTypeIdxTemp[iOrder][reactIdent]);
         }
      }
   }


   // Check
   for (int iOrder = 0; iOrder < 3; ++iOrder)
   {
      assert (rxnss[iOrder].reactionsPerOrder == reactionsPerOrder[iOrder]);
      delete[] reactionsPerTypeIdxTemp[iOrder];
   }
}


/* rxnsetrate.  Sets the internal reaction rate parameters for reaction r of
order order.  These parameters are the squared binding radius, bindrad2, and the
reaction probability, prob.  Zero is returned and erstr is unchanged if the
function is successful.  Possible other return codes are: 1 for a negative input
reaction rate (implies that this value has not been defined yet), 2 for order 1
reactions for which different reactant states would have different reaction
probabilities, 3 for confspread reactions that have a different number of
reactants and products, or 4 for non-confspread bimolecular reactions that have
non-diffusing reactants. */
int rxnsetrate(int rxn, Reaction* reactionList, ParticleSystem* sim) 
{
   int order = reactionList->order[rxn];

   ReactionOrderInfo& rxnss = sim->reactionOrderInfo[order];

	int i,j,i1,i2;
	float vol;
   float sum[MSMAX];
   float sum2,rate3,dsum;


	if(order==0) 
   {															// order 0
		if(reactionList->rate[rxn] < 0) 
         return 1;
      // Obtain volume
		
      vol = sim->getSystemVolume();
      reactionList->prob[rxn] = reactionList->rate[rxn] * sim->getDeltaTime() * vol;   
   }
	else if(order==1) 
   {															// order 1
		if(reactionList->rate[rxn] < 0) 
         return 1;
      i = reactionList->reactants[rxn].x;
		
      for(int ms=0 ; ms<MSMAX; ms++) 
      {
			sum[ms]=0;

         // our only reactant (i) may be involved in many reactions
         // rxnss.numberOfReactionsPerType[i] is the number of reactions; each of them is indexed 
         // (in the global reaction structure) by rxnss.localToGlobalReaction[i][j]
			for(j=0; j < rxnss.numberOfReactionsPerType[i]; j++) // nrxn number of reactions for reactant i
         { 
            int baseIdx = rxnss.reactionsPerTypeIdx[i];
            int r2 = rxnss.localToGlobal[baseIdx + j]; //reactant i, reaction number for i -> reaction number (global)
				if (reactionList->rate[r2] > 0)
               sum[ms] += reactionList->rate[r2]; 
         }
      }   

	   if(reactionList->rate[rxn] == 0) 
         reactionList->prob[rxn] = 0;
		else 
      {
			sum2 = sum[0];
			for(int ms=1; ms<MSMAX; ms++) 
         {
				if(sum2!=sum[ms]) 
            {
					printf("Cannot assign reaction probability because different values are needed for different states");
					return 2; 
            }
         }
			if(sum2 > 0)
            reactionList->prob[rxn] = reactionList->rate[rxn] / sum2 * (1.0f-exp(-sim->getDeltaTime()*sum2));	// desired probability
			sum2 = 0;
         for(j = 0; j < rxnss.numberOfReactionsPerType[i]; j++) 
         {
            int baseIdx = rxnss.reactionsPerTypeIdx[i];
            int rxn2 = rxnss.localToGlobal[baseIdx + j];
				if(rxn2==rxn)
               j = rxnss.numberOfReactionsPerType[i];
				else 
               sum2 += reactionList->prob[rxn2]; 
         }
			reactionList->prob[rxn] = reactionList->prob[rxn]/(1.0f-sum2); // probability, accounting for prior reactions
      }
   }
	else if(order==2) 
   {															// order 2
		if(reactionList->rate[rxn] < 0) 
      {
			if(reactionList->prob[rxn] < 0) 
            reactionList->prob[rxn] = 1;
			return 1; 
      }
      i1 = reactionList->reactants[rxn].x;
		i2 = reactionList->reactants[rxn].y;

		rate3 = reactionList->rate[rxn];
		if(i1 == i2) 
         rate3 *= 2;				
      
      // same reactants
		dsum = sim->getDiffusionCoefficients()[i1] + sim->getDiffusionCoefficients()[i2];
		
      int revOrder, rxnr;
      if (findreverserxn(sim, reactionList, 2, rxn, revOrder, rxnr))
		{
         reactionList->geminateProbability[rxn] = 0.2f; 
      }
      else
      {
         reactionList->unbindRadius[rxn] = 0.0f; 
      }

		if(reactionList->prob[rxn] < 0) 
         reactionList->prob[rxn] = 1;
		
      if(rate3<=0) 
         reactionList->bindRadiusSquared[rxn] = 0;
		else if(dsum<=0) 
      {
         //sprintf(erstr,"Both diffusion coefficients are 0");
         return 4;
      }
		
      float bindRadius = (float)bindingradius(rate3, sim->getDeltaTime(), dsum, -1, 0);
		reactionList->bindRadiusSquared[rxn] = bindRadius * bindRadius;
   }
	return 0; 
}


/* rxnsetproduct.  Sets the initial separations for the products of reaction r
of order order.  This uses the rparamt and rparam elements of the reaction to do
so, along with other required values such as the binding radius and parameters
from any reverse reaction.  The unbindrad and prdpos elements are set up here.
If rparamt is either RPoffset or RPfixed, then it is assumed  that the product
positions have already been set up; they are not modified again by this routine.
Otherwise, it is assumed that the product position vectors and the unbinding
radius have all values equal to 0 initially.  This returns 0 for success or any
of several error codes for errors.  For each error, a message is written to
erstr, which needs to have been pre-allocated to size STRCHAR. */
int rxnsetproduct(ParticleSystem* sim, Reaction* reactionList, int order, int rxn) 
{
	int er = 0;

	//ReactionOrderInfo& rxnss = sim->reactionOrderInfo[order];
	int nprod = reactionList->nprod[rxn];

   er=0;

	if (nprod == 2) 
   {
      float* difc = sim->getDiffusionCoefficients();
      float dc1 = difc[reactionList->products[rxn].x];
		float dc2 = difc[reactionList->products[rxn].y];
		float dsum = dc1 + dc2;

      int reverseOrder, reverseIndex;
      bool hasRev = findreverserxn(sim, reactionList, order, rxn ,reverseOrder, reverseIndex);
      
      if(hasRev == false) 
      {
			if (dsum == 0) 
         {
            dc1 = 1.0;
            dc2 = 1.0;
            dsum = 0.5;
         }
         float rpar = reactionList->unbindRadius[rxn];
         
         reactionList->productPositions[rxn * 6] = rpar*dc1/dsum;
         reactionList->productPositions[rxn * 6 + 1] = 0;
         reactionList->productPositions[rxn * 6 + 2] = 0;
		   reactionList->productPositions[rxn * 6 + 3] = -rpar*dc2/dsum; 
         reactionList->productPositions[rxn * 6 + 4] = 0;
         reactionList->productPositions[rxn * 6 + 5] = 0;
      }
		else 
      {
         float bindradr;
			if (reactionList->bindRadiusSquared[reverseIndex] >=0) 
            bindradr = sqrt(reactionList->bindRadiusSquared[reverseIndex]);
			else bindradr = -1;

			if (dsum<=0) 
         {
				printf("Cannot set unbinding distance because sum of product diffusion constants is 0");
            er=4; 
         }
			
         float rpar = (float)unbindingradius(reactionList->geminateProbability[rxn], sim->getDeltaTime(), dsum, bindradr);
		   if(rpar == -2) 
         {
			   printf("Cannot create an unbinding radius due to illegal input values");
            er=7; 
         }
		   else if (rpar < 0) 
         {
			   printf("Maximum possible geminate binding probability is %g",-rpar);
            er=8; 
         }
		   else 
         {
			   reactionList->unbindRadius[rxn] = rpar;
			   reactionList->productPositions[rxn * 6] = rpar*dc1/dsum;
            reactionList->productPositions[rxn * 6 + 1] = 0;
            reactionList->productPositions[rxn * 6 + 2] = 0;
			   reactionList->productPositions[rxn * 6 + 3] = -rpar*dc2/dsum; 
            reactionList->productPositions[rxn * 6 + 4] = 0;
            reactionList->productPositions[rxn * 6 + 5] = 0;
         }
      }
   }
	return er; 
}
