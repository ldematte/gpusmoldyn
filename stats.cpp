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

#include <math.h>
#include <limits>

#include "SFMT/SFMT.h"
#include "rnd/dci.h"

const double PI = 3.14159265358979323846;
const double SQRT2 = 1.41421356237;
const double SQRTPI = 1.7724538509;
const double SQRT2PI = 2.50662827462;

const float PIf = 3.14159265358979323846f;
const float SQRT2f = 1.41421356237f;
const float SQRTPIf = 1.7724538509f;
const float SQRT2PIf = 2.50662827462f;

const int MT_NN = 19;
static mt_struct MT;
static uint32_t state[MT_NN];

void set_seed(unsigned int seed)
{
   MT.state = state;
   sgenrand_mt(seed, &MT);
}

inline static unsigned long int randULI(void) 
{
   MT.state = state;
	return (unsigned long int) genrand_mt(&MT); 
}

inline static int intrand(int n) 
{
	return (int)(randULI()%n); 
}

inline static float randCCF(void) 
{
   return (float)genrand_real1(); 
}

float gammaln(float x)	
{	
   /* Returns natural log of gamma function, partly from Numerical Recipies and part from Math CRC */
	float sum,t;
	int j;
	static float c[6]={76.18009173f,-86.50532033f,24.01409822f,-1.231739516f,0.120858003e-2f,-0.536382e-5f};
	
	if(x==floor(x)&&x<=0) 
      sum = std::numeric_limits<float>::max();					// 0 or negative integer
	else if(x==floor(x))	
   {											// positive integer
		sum=0;
		for(t=2;t<x-0.1f;t+=1)	sum+=log(t);	
   }
	else if(x==0.5f)	sum=0.572364942f;						// 1/2
	else if(2*x==floor(2*x)&&x>0)	
   {							// other positive half integer
		sum=0.572364942f;
		for(t=0.5;t<x-0.1;t+=1)	sum+=log(t);	
   }
	else if(2*x==floor(2*x))	
   {									// negative half integer
		sum=0.572364942f;
		for(t=0.5;t<-x+0.1;t+=1)	sum-=log(t);	
   }
	else if(x<0)																// other negative
		sum=gammaln(x+1)-log(-x);
	else	{																			// other positive
		x-=1.0f;
		t=x+5.5f;
		t-=(x+0.5f)*log(t);
		sum=1.0f;
		for(j=0;j<=5;j++)	{
			x+=1.0f;
			sum+=c[j]/x;	}
		sum=-t+log(2.50662827465f*sum);	
   }
	return(sum);	
}

// Returns incomplete gamma function, partly from Numerical Recipes
float gammp(float a,float x)	
{ 
	float sum,del,ap,eps;
	float gold=0,g=1,fac=1,b1=1,b0=0,anf,ana,an=0,a1,a0=1;

	eps=3e-7f;
	if(x<0||a<=0) 
      return -1;			// out of bounds
	else if(x==0) 
      return 0;
	else if(x<a+1)	
   {
		ap=a;
		del=sum=1/a;
		while(fabs(del)>fabs(sum)*eps&&ap-a<100)	
      {
			ap+=1;
			del*=x/ap;
			sum+=del;	
      }
		return sum*exp(-x+a*log(x)-gammaln(a));	
   }
	else 
   {
		a1=x;
		for(an=1;an<100;an++)	
      {
			ana=an-a;
			a0=(a1+a0*ana)*fac;
			b0=(b1+b0*ana)*fac;
			anf=an*fac;
			a1=x*a0+anf*a1;
			b1=x*b0+anf*b1;
			if(a1)	
         {
				fac=1.0f/a1;
				g=b1*fac;
				if(fabs((g-gold)/g)<eps)
					return 1.0f-exp(-x+a*log(x)-gammaln(a))*g;
				gold=g;	
         }
      }
   }
	return -1;	
}							// shouldn't ever get here

float erfn(float x)	
{				// Numerical Recipies
	return (x<0?-gammp(0.5,x*x):gammp(0.5,x*x));	
}

float inversefn(float (*fn)(float),float y,float x1,float x2,int n) 
{
	float dx,y2;

	if((*fn)(x1)<(*fn)(x2))	dx=x2-x1;
	else 
   {
		dx=x1-x2;
		x1=x2; 
   }

	for(;n>0;n--) 
   {
		dx*=0.5f;
		x2=x1+dx;
		y2=(*fn)(x2);
		if(y2<y) x1=x2; 
   }
	return x1+0.5f*dx; 
}

/* erfcc returns the complementary error function, calculated with a method
copied verbatim from Numerical Recipies in C, by Press et al., Cambridge
University Press, Cambridge, 1988.  It works for all x and has fractional error
everywhere less than 1.2e-7.  */

float erfcc(float x) 
{
	float t,z,ans;

	z=fabs(x);
	t=1.0f/(1.0f+0.5f*z);
	ans=t*exp(-z*z-1.26551223f+t*(1.00002368f+t*(0.37409196f+t*(0.09678418f+t*(-0.18628806f+t*(0.27886807f+t*(-1.13520398f+t*(1.48851587f+t*(-0.82215223f+t*0.17087277f)))))))));
	return x>=0?ans:2.0f-ans; 
}

float erfcintegral(float x) 
{
	return (1.0f-exp(-x*x))/SQRTPIf+x*erfcc(x); 
}

void randtableF(float *a, int n, int eq) {
	int i;
	float dy;

	if(eq==1) {
		dy=2.0f/n;
	  for(i=0;i<n/2;i++)
	    a[i]=SQRT2f*inversefn(erfn,(i+0.5f)*dy-1.0f,-20,20,30);
	  for(i=n/2;i<n;i++)
	    a[i]=-a[n-i-1]; }
	else if(eq==2) {
		dy=1.0f/SQRTPIf/n;
		for(i=0;i<n;i++)
			a[i]=SQRT2f*inversefn(erfcintegral,(i+0.5f)*dy,0,20,30); 
   }
	return; 
}

void randshuffletableF(float *a,int n) 
{
	int i,j;
	float x;

	for(i=0;i<n;i++) 
   {
		j=intrand(n);
		x=a[i];
		a[i]=a[j];
		a[j]=x; 
   }
}

int poisrandF(float xm) 
{
   static float sq,alxm,g,oldm=-1.0;
   float em,t,y;

   if(xm<=0) 
      return 0;
   else if(xm<12.0) 
   {
      if(xm!=oldm) 
      {
         oldm=xm;
         g=exp(-xm);
      }
      em=0;
      for(t=randCCF();t>g;t*=randCCF()) 
      {
         em+=1.0; 
      }
   }
   else 
   {
      if(xm!=oldm) 
      {
         oldm=xm;
         sq=sqrtf(2.0f*xm);
         alxm=logf(xm);
         g=xm*alxm-gammaln(xm+1.0f); 
      }
      do 
      {
         do 
         {
            y=tanf(PIf*randCCF());
            em=sq*y+xm; 
         } while(em<0);
         em=floor(em);
         t=0.9f*(1.0f+y*y)*expf(em*alxm-gammaln(em+1.0f)-g); 
      } while(randCCF()>t); 
   }
   return (int) em; 
}

void createGaussTable(float* gaussianLookupTable, size_t gaussianTableDim)
{
   set_seed(777);
	randtableF(gaussianLookupTable,gaussianTableDim,1);
	randshuffletableF(gaussianLookupTable,gaussianTableDim);
}

