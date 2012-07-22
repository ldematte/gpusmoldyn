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


__device__
void doReact(Reaction* reaction, 
             int reactIndexes[2],
             int reactTypes[2], 
             float3 pos1, float2 pos2,
             float* currentPos, int* types,  //numParticles
             float* bornPos, int* bornTypes, //numParticles: at most one new every cycle
             float* diffusionCoefficients) //numTypes
{

	//int order,prd,d,nprod,dim;
	//int calc;
	//double dc1,dc2,x,dist;
	//molssptr mols;
	//moleculeptr mptr,mptrallo;
	//boxptr rxnbptr;
	//double v1[DIMMAX],rxnpos[DIMMAX],m3[DIMMAX*DIMMAX];
	//enum MolecState ms;

	//mols=sim->mols;
	//dim=sim->dim;
	int order = reaction->order;

   int mol1Index = reactIndexes[0];
   int mol2Index = reactIndexes[1];

   int mol1Type = reactTypes[0];
   int mol2Type = reactTypes[1];

   float3 reactPos;

	if(order < 2) // order 0 or 1   														
		reactPos = pos1;
   else 
   {  // order 2
		float dc1 = diffusionCoefficients[mol1Type];
		float dc2 = diffusionCoefficients[mol2Type];
      float x;
		if (dc1==0 && dc2==0) 
         x=0.5f;
		else 
         x= dc2 / (dc1+dc2);
		reactPos = x * pos1 + (1.0f-x) * pos2;
   }

   

   // place products
	int nprod = rxn->nprod;

   if (order == nprod)
   {
      //simply replace reactants with products
	   for (int iProd = 0; iProd < nprod; ++iProd) 
      {
		   types[ = rxn->prdident[prd];

			mptr->box=rxnbptr;
			for(d=0;d<dim;d++) mptr->posx[d]=rxnpos[d];

			mptr->mstate=ms=rxn->prdstate[prd];
			mptr->pnl=pnl;			

			for(d=0;d<dim&&rxn->prdpos[d]==0;d++);

         //prdpos???

			if(d!=dim) {
				if(rxn->rparamt==RPfixed) {
					for(d=0;d<dim;d++) v1[d]=rxn->prdpos[prd][d]; }
				else if(dim==1) {
					if(!calc) {m3[0]=signrand();calc=1;}
					v1[0]=m3[0]*rxn->prdpos[prd][0]; }
				else if(dim==2) {
					if(!calc) {DirCosM2D(m3,unirandCOD(0,2*PI));calc=1;}
					dotMVD(m3,rxn->prdpos[prd],v1,2,2); }
				else if(dim==3) {
					if(!calc) {DirCosMD(m3,thetarandCCD(),unirandCOD(0,2*PI),unirandCOD(0,2*PI));calc=1;}
					dotMVD(m3,rxn->prdpos[prd],v1,3,3); }
				else {
					if(!calc) {DirCosMD(m3,thetarandCCD(),unirandCOD(0,2*PI),unirandCOD(0,2*PI));calc=1;}
					dotMVD(m3,rxn->prdpos[prd],v1,3,3);
					for(d=3;d<dim;d++) v1[d]=rxn->prdpos[prd][d]; }
				for(d=0;d<dim;d++)
					mptr->pos[d]=mptr->posx[d]+v1[d]; }
			else {
				for(d=0;d<dim;d++)
					mptr->pos[d]=mptr->posx[d]; 
         }

		mptr->list=sim->mols->listlookup[mptr->ident][mptr->mstate]; 
   }

	if(mptr1) molkill(sim,mptr1,ll1,m1);					// kill reactants
	if(mptr2) molkill(sim,mptr2,ll2,m2);
}


   /* zeroreact.  Figures out how many molecules to create for each zeroth order
reaction and then tells doreact to create them.  It returns 0 for success or 1
if not enough molecules were allocated initially. */
int zeroreact(simptr sim) {
	int i,r,nmol;
	rxnptr rxn;
	rxnssptr rxnss;
	double pos[DIMMAX];
	panelptr pnl;

	pnl=NULL;
	rxnss=sim->rxnss[0];
	if(!rxnss) return 0;
	for(r=0;r<rxnss->totrxn;r++) {
		rxn=rxnss->rxn[r];
		nmol=poisrandD(rxn->prob);
		for(i=0;i<nmol;i++) {
			if(rxn->cmpt) compartrandpos(sim,pos,rxn->cmpt);
			else if(rxn->srf) pnl=surfrandpos(rxn->srf,pos,sim->dim);
			else systemrandpos(sim,pos);
			if(doreact(sim,rxn,NULL,NULL,-1,-1,-1,-1,pos,pnl)) return 1; }
		sim->eventcount[ETrxn0]+=nmol; }
	return 0; }


/* unireact.  Identifies and performs all unimolecular reactions.  Reactions
that should occur are sent to doreact to process them.  The function returns 0
for success or 1 if not enough molecules were allocated initially. */
int unireact(simptr sim) {
	rxnssptr rxnss;
	rxnptr rxn,*rxnlist;
	moleculeptr *mlist,mptr;
	int *nrxn,**table;
	int i,j,m,nmol,ll;
	enum MolecState ms;

	rxnss=sim->rxnss[1];
	if(!rxnss) return 0;
	nrxn=rxnss->nrxn;
	table=rxnss->table;
	rxnlist=rxnss->rxn;
	for(ll=0;ll<sim->mols->nlist;ll++)
		if(rxnss->rxnmollist[ll]) {
			mlist=sim->mols->live[ll];
			nmol=sim->mols->nl[ll];
			for(m=0;m<nmol;m++) {
				mptr=mlist[m];
				i=mptr->ident;
				ms=mptr->mstate;
				for(j=0;j<nrxn[i];j++) {
					rxn=rxnlist[table[i][j]];
					if((!rxn->cmpt&&!rxn->srf)||(rxn->cmpt&&posincompart(sim,mptr->pos,rxn->cmpt))||(rxn->srf&&mptr->pnl&&mptr->pnl->srf==rxn->srf))
						if(coinrandD(rxn->prob)&&rxn->permit[ms]&&mptr->ident!=0) {
							if(doreact(sim,rxn,mptr,NULL,ll,m,-1,-1,NULL,NULL)) return 1;
							sim->eventcount[ETrxn1]++;
							j=nrxn[i]; }}}}
	return 0; }


/* morebireact.  Given a probable reaction from bireact, this orders the
reactants, checks for reaction permission, moves a reactant in case of periodic
boundaries, increments the appropriate event counter, and calls doreact to
perform the reaction.  The return value is 0 for success (which may include no
reaction) and 1 for failure. */





