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

__device__
double Geo_LineNormal3D(double *pt1,double *pt2,double *point,double *ans) 
{
   double dot,line[3];

   line[0]=pt2[0]-pt1[0];
   line[1]=pt2[1]-pt1[1];
   line[2]=pt2[2]-pt1[2];
   dot=line[0]*line[0]+line[1]*line[1]+line[2]*line[2];
   if(dot==0) {								// pt1 == pt2
      ans[0]=point[0]-pt1[0];
      ans[1]=point[1]-pt1[1];
      ans[2]=point[2]-pt1[2];
      dot=ans[0]*ans[0]+ans[1]*ans[1]+ans[2]*ans[2];
      if(dot==0) {							// pt1 == pt2 and point == pt1 so return unit x
         ans[0]=1;
         ans[1]=0;
         ans[2]=0;
         return 0; }
      dot=sqrt(dot);						// pt1 == pt2 so return normalized point
      ans[0]/=dot;
      ans[1]/=dot;
      ans[2]/=dot;
      return dot; }
   dot=sqrt(dot);
   line[0]/=dot;
   line[1]/=dot;
   line[2]/=dot;

   ans[0]=point[0]-pt1[0];
   ans[1]=point[1]-pt1[1];
   ans[2]=point[2]-pt1[2];
   dot=ans[0]*line[0]+ans[1]*line[1]+ans[2]*line[2];
   ans[0]-=dot*line[0];
   ans[1]-=dot*line[1];
   ans[2]-=dot*line[2];
   dot=ans[0]*line[0]+ans[1]*line[1]+ans[2]*line[2];
   ans[0]-=dot*line[0];
   ans[1]-=dot*line[1];
   ans[2]-=dot*line[2];
   dot=ans[0]*ans[0]+ans[1]*ans[1]+ans[2]*ans[2];
   if(dot==0) {											// point is on line
      if(line[0]==0 && line[1]==0) {		// line parallel to z so return unit x
         ans[0]=1;
         ans[1]=0;
         ans[2]=0;
         return 0; 
      }
      ans[0]=line[1];									// return right side perpendicular in x,y plane
      ans[1]=-line[0];
      ans[2]=0;
      dot=sqrt(ans[0]*ans[0]+ans[1]*ans[1]+ans[2]*ans[2]);
      ans[0]/=dot;
      ans[1]/=dot;
      ans[2]/=dot;
      return 0; 
   }
   dot=sqrt(dot);
   ans[0]/=dot;
   ans[1]/=dot;
   ans[2]/=dot;
   return dot; 
}


/* Returns the side of the panel pnl that point pt is on, which is either a "f"
or a "b" for front or back, respectively. "b" is returned if the point is
exactly at the panel.  The value returned by this function defines the side that
pt is on, so should either be called or exactly copied for other functions that
care. */
enum PanelFace panelside(double* pt,panelptr pnl,int dim,double *distptr) 
{
	enum PanelFace face;
	double **point,*front,dist,cylnorm[3];
	int d;

	point=pnl->point;
	front=pnl->front;
	dist=0;

	if(pnl->ps==PSrect) {														// rectangle
		d=(int)front[1];
		dist=front[0]*(pt[d]-point[0][d]); }
	else if(pnl->ps==PStri || pnl->ps==PSdisk) {			// triangle, disk
		for(d=0;d<dim;d++) dist+=(pt[d]-point[0][d])*front[d]; }
	else if(pnl->ps==PSsph || pnl->ps==PShemi) {			// sphere, hemisphere
		for(d=0;d<dim;d++) dist+=(pt[d]-point[0][d])*(pt[d]-point[0][d]);
		dist=front[0]*(sqrt(dist)-point[1][0]); }
	else if(pnl->ps==PScyl) {												// cylinder
		if(dim==2) {
			for(d=0;d<dim;d++) dist+=(pt[d]-point[0][d])*front[d];
			dist=front[2]*(fabs(dist)-point[2][0]); }
		else {
			dist=Geo_LineNormal3D(point[0],point[1],pt,cylnorm);
			dist=front[2]*(dist-point[2][0]); }}
	else
		dist=0;

	face=dist>0?PFfront:PFback;
	if(distptr) *distptr=dist;
	return face; 
}


/* 	lineXpanel.  Determines if line from pt1 to pt2 crosses panel pnl, using a
dim dimensional system.  The panel includes all of its edges.  If it crosses, 1
is returned, the face that is on the pt1 side of the panel is returned in
faceptr, crsspt is set to the coordinates of the crossing point, and cross
points to the crossing position on the line, which is a number between 0 and 1,
inclusive.  If it does not cross, 0 is returned and the other values are
undefined.  Crossing is handled very carefully such that the exact locations of
pt1 and pt2, using tests that are identical to those in panelside, are used to
determine which sides of the panel they are on.  While crsspt will be returned
with coordinates that are very close to the panel location, it may not be
precisely at the panel, and there is no certainty about which side of the panel
it will be on; if it matters, fix it with fixpt2panel.

If the line crosses the panel more than once, which can happen for spherical or
other curved panels, the smaller of the two crossing points is returned.  For
sphere and cylinder, 0 is returned if either both points are inside or both
points are outside and the line segment does not cross the object twice.

Each portion of this routine does the same things, and usually in the same
order.  First, the potential intersection face is determined, then the crossing
value, then the crossing point, and finally it finds if intersection actually
happened.  For hemispheres and cylinders, if intersection does not happen for
the first of two possible crossing points, it is then checked for the second
point. */
int lineXpanel(double *pt1,double *pt2,panelptr pnl,int dim,double *crsspt,enum PanelFace *face1ptr,enum PanelFace *face2ptr,double *crossptr,double *cross2ptr) {
	surfaceptr srf;
	double **point,*front,dist1,dist2;
	double dot,cross,cross2,nrdist,nrpos;
	int intsct,d;
	enum PanelFace face1,face2,facein;

	srf=pnl->srf;
	point=pnl->point;
	front=pnl->front;
	face1=panelside(pt1,pnl,dim,&dist1);
	face2=panelside(pt2,pnl,dim,&dist2);
	cross=cross2=-1;

	if(pnl->ps==PSrect) {														// rectangle
		if(face1==face2) return 0;
		cross=dist1/(dist1-dist2);
		for(d=0;d<dim;d++) crsspt[d]=pt1[d]+cross*(pt2[d]-pt1[d]);
		if(dim==1) intsct=1;
		else if(dim==2) {
			d=(int)front[2];
			intsct=((point[0][d]<=crsspt[d] && crsspt[d]<=point[1][d]) || (point[1][d]<=crsspt[d] && crsspt[d]<=point[0][d])); }
		else {
			d=(int)front[2];
			intsct=((point[0][d]<=crsspt[d] && crsspt[d]<=point[1][d]) || (point[1][d]<=crsspt[d] && crsspt[d]<=point[0][d]));
			d=(d+1)%3;
			if(d==(int)front[1]) d=(d+1)%3;
			intsct=intsct && ((point[1][d]<=crsspt[d] && crsspt[d]<=point[2][d]) || (point[2][d]<=crsspt[d] && crsspt[d]<=point[1][d])); }}

	else if(pnl->ps==PStri) {												// triangle
		if(face1==face2) return 0;
		cross=dist1/(dist1-dist2);
		for(d=0;d<dim;d++) crsspt[d]=pt1[d]+cross*(pt2[d]-pt1[d]);
		if(dim==1) intsct=1;
		else if(dim==2) {
			intsct=((point[0][0]<=crsspt[0] && crsspt[0]<=point[1][0]) || (point[1][0]<=crsspt[0] && crsspt[0]<=point[0][0]));
			intsct=intsct && ((point[0][1]<=crsspt[1] && crsspt[1]<=point[1][1]) || (point[1][1]<=crsspt[1] && crsspt[1]<=point[0][1])); }
		else {
			intsct=Geo_PtInTriangle(point[0],point[1],point[2],front,crsspt); }}

	else if(pnl->ps==PSsph || pnl->ps==PShemi) {		// sphere, hemisphere
		facein=front[0]>0?PFback:PFfront;
		if(face1==facein && face1==face2) return 0;
		cross=Geo_LineXSphs(pt1,pt2,point[0],point[1][0],dim,&cross2,&nrdist,&nrpos);
		if(face1==face2 && (nrdist>point[1][0] || nrpos<0 || nrpos>1)) return 0;
		if(face1==facein)
			cross=cross2;
		for(d=0;d<dim;d++) crsspt[d]=pt1[d]+cross*(pt2[d]-pt1[d]);
		if(pnl->ps==PSsph) intsct=1;
		else {
			dot=0;
			for(d=0;d<dim;d++) dot+=(crsspt[d]-point[0][d])*point[2][d];
			intsct=(dot<=0);
			if(!intsct && face1==face2) {
				cross=cross2;
				face1=(face1==PFfront)?PFback:PFfront;
				for(d=0;d<dim;d++) crsspt[d]=pt1[d]+cross*(pt2[d]-pt1[d]);
				dot=0;
				for(d=0;d<dim;d++) dot+=(crsspt[d]-point[0][d])*point[2][d];
				intsct=(dot<=0); }}}

	else if(pnl->ps==PScyl) {									// cylinder
		facein=(int)front[2]>0?PFback:PFfront;
		if(face1==facein && face1==face2) return 0;
		if(dim==2) cross=Geo_LineXCyl2s(pt1,pt2,point[0],point[1],front,point[2][0],&cross2,&nrdist,&nrpos);
		else cross=Geo_LineXCyls(pt1,pt2,point[0],point[1],point[2][0],&cross2,&nrdist,&nrpos);
		if(face1==face2 && (nrdist>point[2][0] || nrpos<0 || nrpos>1)) return 0;
		if(face1==facein)
			cross=cross2;
		for(d=0;d<dim;d++) crsspt[d]=pt1[d]+cross*(pt2[d]-pt1[d]);
		intsct=Geo_PtInSlab(point[0],point[1],crsspt,dim);
		if(!intsct && face1==face2) {
			cross=cross2;
			face1=(face1==PFfront)?PFback:PFfront;
			for(d=0;d<dim;d++) crsspt[d]=pt1[d]+cross*(pt2[d]-pt1[d]);
			intsct=Geo_PtInSlab(point[0],point[1],crsspt,dim); }}

	else if(pnl->ps==PSdisk) {												// disk
		if(face1==face2) return 0;
		cross=dist1/(dist1-dist2);
		for(d=0;d<dim;d++) crsspt[d]=pt1[d]+cross*(pt2[d]-pt1[d]);
		dot=0;
		for(d=0;d<dim;d++) dot+=(crsspt[d]-point[0][d])*(crsspt[d]-point[0][d]);
		intsct=(dot<=point[1][0]*point[1][0]); }

	else
		intsct=0;

	if(face1ptr) *face1ptr=face1;
	if(face2ptr) *face2ptr=face2;
	if(crossptr) *crossptr=cross;
	if(cross2ptr) *cross2ptr=cross2;
	return intsct; 
}
