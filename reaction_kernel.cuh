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

/* doreact.  Executes a reaction that has already been determined to have
happened.  rxn is the reaction and mptr1 and mptr2 are the reactants, where
mptr2 is ignored for unimolecular reactions, and both are ignored for zeroth
order reactions.  ll1 is the live list of mptr1, m1 is its index in the master
list, ll2 is the live list of mptr2, and m2 is its index in the master list; if
these donÕt apply (i.e. for 0th or 1st order reactions, set them to -1 and if
either m1 or m2 is unknown, again set the value to -1.  If there are multiple
molecules, they need to be in the same order as they are listed in the reaction
structure (which is only important for confspread reactions and for a completely
consistent panel destination for reactions between two surface-bound molecules).
Reactants are killed, but left in the live lists.  Any products are created on
the dead list, for transfer to the live list by the molsort routine.  Molecules
that are created are put at the reaction position, which is the average position
of the reactants weighted by the inverse of their diffusion constants, plus an
offset from the product definition.  The cluster of products is typically
rotated to a random orientation.  If the displacement was set to all 0Õs
(recommended for non-reacting products), the routine is fairly fast, putting
all products at the reaction position.  If the rparamt character is RPfixed, the
orientation is fixed and there is no rotation.  Otherwise, a non-zero
displacement results in the choosing of random angles and vector rotations.  If
the system has more than three dimensions, only the first three are randomly
oriented, while higher dimensions just add the displacement to the reaction
position.  The function returns 0 for successful operation and 1 if more
molecules are required than were initially allocated.  This function lists the
correct box in the box element for each product molecule, but does not add the
product molecules to the molecule list of the box.  The bptr input is only
looked at for 0th order reactions; for these, NULL means that products should be
placed uniformly throughout the system whereas a non-NULL value means that
products should be placed uniformly throughout the listed box. */


//__device__
//void doReact(Reaction* reaction, 
//             int reactIndexes[2], //mol1Index: for zeroth-reaction?
//             int reactTypes[2], 
//             float3 pos1, float3 pos2,
//             float3* pos, int* types,  //lenght = numParticles
//             int* birthDeath,                //lenght = numParticle. Initially set to 0, changed to +1 or -1 by necessity
//             float3* bornPos, int* bornTypes, //lenght = numParticles: at most one new every cycle
//             float* diffusionCoefficients) //numTypes
//{
//	int order = reaction->order;
//
//   int mol1Index = reactIndexes[0];
//   int mol2Index = reactIndexes[1];
//
//   int mol1Type = reactTypes[0];
//   int mol2Type = reactTypes[1];
//
//   float3 reactPos;
//
//	if(order < 2) // order 0 or 1   														
//		reactPos = pos1;
//   else 
//   {  // order 2
//		float dc1 = diffusionCoefficients[mol1Type];
//		float dc2 = diffusionCoefficients[mol2Type];
//      float x;
//		if (dc1==0 && dc2==0) 
//         x=0.5f;
//		else 
//         x= dc2 / (dc1+dc2);
//		reactPos = x * pos1 + (1.0f-x) * pos2;
//   }
//
//   
//
//   // place products
//	int nprod = reaction->nprod;
//
//   birthDeath[mol1Index] = nprod - order;
//
//   //TODO: make writes anyway?
//
//   if (order == nprod)
//   {
//      //simply replace reactants with products
//	   for (int iProd = 0; iProd < nprod; ++iProd) 
//      {
//         //TODO!! Change from array of structs to struct of arrays
//         int molIdx = reactIndexes[iProd];
//		   types[molIdx] = reaction->products[iProd];
//         pos[molIdx] = reactPos;
//         
//
//			//for(d=0;d<dim&&rxn->prdpos[d]==0;d++);
//
//         //prdpos???
//
//			/*if(d!=dim) {
//				if(rxn->rparamt==RPfixed) {
//					for(d=0;d<dim;d++) v1[d]=rxn->prdpos[prd][d]; }
//				else if(dim==1) {
//					if(!calc) {m3[0]=signrand();calc=1;}
//					v1[0]=m3[0]*rxn->prdpos[prd][0]; }
//				else if(dim==2) {
//					if(!calc) {DirCosM2D(m3,unirandCOD(0,2*PI));calc=1;}
//					dotMVD(m3,rxn->prdpos[prd],v1,2,2); }
//				else if(dim==3) {
//					if(!calc) {DirCosMD(m3,thetarandCCD(),unirandCOD(0,2*PI),unirandCOD(0,2*PI));calc=1;}
//					dotMVD(m3,rxn->prdpos[prd],v1,3,3); }
//				else {
//					if(!calc) {DirCosMD(m3,thetarandCCD(),unirandCOD(0,2*PI),unirandCOD(0,2*PI));calc=1;}
//					dotMVD(m3,rxn->prdpos[prd],v1,3,3);
//					for(d=3;d<dim;d++) v1[d]=rxn->prdpos[prd][d]; }
//				for(d=0;d<dim;d++)
//					mptr->pos[d]=mptr->posx[d]+v1[d]; }
//			else {
//				for(d=0;d<dim;d++)
//					mptr->pos[d]=mptr->posx[d]; 
//         }*/
//      }
//   }
//   else if (order == 1) //nprod == 2
//   {
//		types[mol1Index] = reaction->products[0];
//      pos[mol1Index] = reactPos;
//
//      bornTypes[mol1Index] = reaction->products[1];
//      bornPos[mol1Index] = reactPos;
//   }
//   else if (order == 0) //nprod ==1
//   {
//      bornTypes[mol1Index] = reaction->products[1];
//      bornPos[mol1Index] = reactPos;
//   }
//   else if (order == 2) //nprod ==1
//   {
//      types[mol1Index] = reaction->products[0];
//      pos[mol1Index] = reactPos;
//
//      types[mol2Index] = -1;
//      //pos[mol2Index] = reactPos;
//   }
//}
//

#include "rand_kernel.cuh"

inline __device__ void dotMV(float *a,float *b,float *c, int m, int n)	
{
	int i,j;
   
	for(i=0;i<m;i++)	{
		c[i]=0;
		for(j=0;j<n;j++)	
         c[i]+=a[n*i+j]*b[j];	
   }
}

inline __device__ void DirCosM(float *c,float theta,float phi,float chi)	
{
	float cp,ct,cc,sp,st,sc;

   __sincosf(phi, &sp, &cp);
   __sincosf(theta, &st, &ct);
   __sincosf(chi, &sc, &cc);

	//cp=cosf(phi);
	//ct=cosf(theta);
	//cc=cosf(chi);
	//sp=sinf(phi);
	//st=sinf(theta);
	//sc=sinf(chi);
	c[0]=cp*ct*cc-sp*sc;
	c[1]=sp*ct*cc+cp*sc;
	c[2]=-st*cc;
	c[3]=-cp*ct*sc-sp*cc;
	c[4]=-sp*ct*sc+cp*cc;
	c[5]=st*sc;
	c[6]=cp*st;
	c[7]=sp*st;
	c[8]=ct;
}

__device__
void bimReact(Reaction* reactionList, 
              int r,
              int mol1Index, int mol2Index, //indexes in UNSORTED array
              int mol1Type, int mol2Type,
              float3 pos1, float3 pos2,
              float4* pos, int* types,      //lenght = numParticles, unsorted
              int* counter,                 //lenght = numParticles
              uint* gridParticleIndex,      //length = numParticles
              float* diffusionCoefficients, //lenght = numTypes
              MersenneTwisterState* randState, 
              unsigned int tid) 
{
	// int order = reaction->order; -> always 2
   const int dim = 3;

   //int mol1Type = types[mol1Index];
   //int mol2Type = types[mol2Index];
   float3 reactPos;

   float m3[dim * dim];//of float3 m3[dim];

	float dc1 = diffusionCoefficients[mol1Type];
   float dc2 = diffusionCoefficients[mol2Type];
   float x;
	if (dc1==0 && dc2==0) 
      x=0.5f;
   else 
      x= dc2 / (dc1+dc2);

	reactPos = x * pos1 + (1.0f-x) * pos2;

   // place products
   counter[mol1Index] = 1;
   counter[mol2Index] = reactionList->nprod[r] - 1; //0 if only one product, 1 otherwise

   types[mol1Index] = reactionList->products[r].x;

   // if (reaction->nprod == -1) 
   //    types[mol2Index] = -1;
   // else
   types[mol2Index] = reactionList->products[r].y;

   float v1[3];
   float* endPos1 = &(reactionList->productPositions[r * 6]);
   float* endPos2 = &(reactionList->productPositions[r * 6 + 3]);

   DirCosM(m3, thetarandCCF(randState, tid), 
               unirandCOF(0, 2 * CUDART_PI_F, randState, tid), 
               unirandCOF(0, 2 * CUDART_PI_F, randState, tid));
         
   dotMV(m3, endPos1, v1, 3, 3); 
   pos[mol1Index] = make_float4(reactPos.x + v1[0], reactPos.y + v1[1], reactPos.z + v1[2], 0.0f);

   dotMV(m3, endPos2, v1, 3, 3); 
   pos[mol2Index] = make_float4(reactPos.x + v1[0], reactPos.y + v1[1], reactPos.z + v1[2], 0.0f);
}

/* unireact.  Identifies and performs all unimolecular reactions.  Reactions
that should occur are sent to doreact to process them.  The function returns 0
for success or 1 if not enough molecules were allocated initially. */
__global__
void uniReact(int*    nrxn,           // rxnss.numberOfReactionsPerType,
              int*    table,          // rxnss.localToGlobal, 
              int*    tableTypeIdx,   // rxnss.reactionsPerTypeIdx,
              //int*    rxnlist,      // rxnss.reactionIndexes, 
              float4* pos,
              int*    types,
              float4* addedPos,       // lenght = numParticles              
              int*    addedTypes,     // lenght = numParticles              
              int*    birthArray,     // lenght = numParticles
              int*    deathArray,     // lenght = numParticles

              MersenneTwisterState* rngStateArray,
              Reaction* reactionList,
              int nParticles)
{
	//for (int p = 0; p < nParticles; ++p) 
   uint pidx = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;
   if (pidx >= nParticles)  // handle case when no. of particles not multiple of block size
      return;

   //default: keep it, and do not add anything
   birthArray[pidx] = 0;  //no birth
   deathArray[pidx] = 1;  //no death (keep exisiting)

   // type of this particle
   int type = types[pidx];

   if (type <= 0) // should never happen here!
   {
      deathArray[pidx] = 0; //TODO: remove if not necessary
      return;
   }

   int rngIndex = pidx % MT_RNG_COUNT;
   MersenneTwisterState* rngState = &(rngStateArray[rngIndex]);

	float4 reactPos = pos[pidx];

   // scroll the list of reactions for that particle
   for (int j=0; j < nrxn[type]; ++j) 
   {
      int rxn = table[tableTypeIdx[type] + j];
      if (coinrandF(reactionList->prob[rxn], rngState, pidx)) 
      {
         // do reaction
         if (reactionList->products[rxn].x <= 0)
            deathArray[pidx] = 0; // it was a decay, delete this
         else
         {
            types[pidx] = reactionList->products[rxn].x;
            int type2 = reactionList->products[rxn].y;
            if (type2 > 0)
            {

               addedTypes[pidx] = type2;
               birthArray[pidx] = 1; //we need another spot for it...
#if 1
               addedPos[pidx] = reactPos;
#else
               float m3[dim * dim];//or float3 m3[dim];
               float v1[3];
               float* endPos1 = &(reactionList->productPositions[r * 6]);
               float* endPos2 = &(reactionList->productPositions[r * 6 + 3]);

               DirCosM(m3, thetarandCCF(randState, tid), 
                  unirandCOF(0, 2 * CUDART_PI_F, randState, tid), 
                  unirandCOF(0, 2 * CUDART_PI_F, randState, tid));

               dotMV(m3, endPos1, v1, 3, 3); 
               addedPos[pidx] = make_float4(reactPos.x + v1[0], reactPos.y + v1[1], reactPos.z + v1[2], 0.0f);
#endif               
            }
         }

         // no other reactions for this guy
         break; //j=nrxn[i];
      }
   }
}





