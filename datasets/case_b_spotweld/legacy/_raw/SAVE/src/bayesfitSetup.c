/*
 *  bayesfitSetup.c
 *  R_Package
 *
 *  Created by Jesus Palomo on 9/10/12.
 *  Copyright (C) 2013-present Jesus Palomo, Gonzalo Garcia-Donato,
 *							  and Rui Paulo
 *  All rights reserved.
 *
 */

#include "bayesfitSetup.h"
#include "setup.h"
#include "iostuff.h"

#include <string.h>
#include <stdlib.h>
#include <R_ext/Print.h>
#include <math.h>
#include <R_ext/Lapack.h>

void bayesfitSetupCalib(int screen, char home[], double multmle, int p, int q, int pstar, int pcont, int NM, int NF, double **thetaF, 
				   double **thetaM, double **thetaL, double **priorshapes, double **prioriscales, 
				   double ***ubounds, double **yF, double **yM, double **y, int *NFtot, int **Nrep, double ***ZF, 
				   double ***ZM, double ***Z, double *s2F, double ***X, double ***Xt, char **fileF, char **filepath, 
				   char **fileU, char **frateU)
{
	if (screen != 0){
		Rprintf("\n--- Entra en bayesfitSetupCalib ---\n");
		Rprintf("Home is: %s\n",home);
	}

	// FILENAMES
	char strtmp[100];
	
	/* file where sequence of thetaF goes */
	char afileF [100];
	strcpy(strtmp,home);
	strcat(strtmp,"thetaF.out");
	if (screen != 0) Rprintf("strtmp is: %s\n",strtmp);
	strcpy(afileF,strtmp);
	*fileF = malloc(sizeof(afileF));
	memcpy(*fileF, afileF, sizeof(afileF));
	if (screen != 0) Rprintf("FileF is: %s\n",*fileF);
	
	/* file where sequence of (yM*,b) goes */
	char afilepath [100];
	strcpy(strtmp,home);
	strcat(strtmp,"filepath.out");
	strcpy(afilepath,strtmp);
	*filepath = malloc(sizeof(afilepath));
	memcpy(*filepath, afilepath, sizeof(afilepath));
	if (screen != 0){
		Rprintf("filepath.out is: %s\n",*filepath);
	}
	
	
	////////////////////////////////////////////////////////
	//
	/* file where the sequence of ustars goes */
	char afileU [100];
	strcpy(strtmp,home);
	strcat(strtmp,"ustar.out");
	strcpy(afileU,strtmp);
	*fileU = malloc(sizeof(afileU));
	memcpy(*fileU, afileU, sizeof(afileU));
		
	/* and its acceptance rate */
	char afrateU [100];
	strcpy(strtmp,home);
	strcat(strtmp,"rate.out");
	strcpy(afrateU,strtmp);
	*frateU = malloc(sizeof(afrateU));
	memcpy(*frateU, afrateU, sizeof(afrateU));
		
	int i, j; //counters
	
	/*********************************/
	/****** INPUTS TO THE PROBLEM ****/
	/*********************************/
	if (screen != 0){
		Rprintf("\n--- Options: output files ---\n");
		
		// FILENAMES
		/* working directory path */
		Rprintf("working directory path=%s\n",home);
		
		Rprintf("file where the sequence of ustars goes=%s\n",*fileU);		  
		Rprintf("file where the acceptance rate goes=%s\n",*frateU);
		/* file where sequence of thetaF goes */
		Rprintf("file where sequence of thetaF goes=%s\n",*fileF);
		
		/* file where sequence of (yM*,b) goes */
		Rprintf("file containing the sequence of (yM*,b) goes=%s\n",*filepath);
	}
	

	////////////////////////////////////////////////////////
	// READ MLE OF THETAF
	*thetaF = get_mlethetaF(home,pcont);
	if (screen != 0) Rprintf("thetaF read\n");
	// READ MLE OF THETAM
	*thetaM = get_mlethetaM(home,p);
	if (screen != 0) Rprintf("thetaM read\n");
	// READ MLE OF THETAL
	*thetaL= get_mlethetaL(home,q);
	if (screen != 0) Rprintf("thetaL read\n");
				
	/* read the matrix with bounds on u */
	/* and the matrix specifying the priors and the bounds */
	*ubounds = get_ubounds(home, pstar);
	if (screen != 0) Rprintf("ubounds read\n");

	//PRIORS
	*priorshapes=dvector(0,pcont+2);
	if (screen != 0) Rprintf("priorshapes set\n");
	*prioriscales=dvector(0,pcont+2);
				
	for(i=0;i<pcont+2;i++) (*priorshapes)[i]=1.;
			
		for(i=0;i<pcont+1;i++){
			(*prioriscales)[i]=(*thetaF)[i];
		}
	(*prioriscales)[pcont+1]=(*thetaF)[2*(pcont)+1];
	
	int k=pcont+2;
	int one=1;
	dscal_(&k,&multmle,&*prioriscales[0],&one);
	for(i=0;i<pcont+2;i++)
		(*prioriscales)[i]=1./(*prioriscales)[i];
	if (screen != 0) Rprintf("priorscales set\n");

	*Nrep = get_inputsrep(home, NF);
	if (screen != 0) Rprintf("Nrep read\n");

	//READ/CONSTRUCT DATA
	int N;       /* NM+NF */
	N = NM+NF;
	
	/* total number of field observations (this number includes 
							  the replicates)
	   compute total number of field observations */
	for(i=0;i<NF;i++){
		*NFtot = *NFtot+(*Nrep)[i];
	}
	if (screen != 0) Rprintf("NFtot set\n");
	
	*yM = get_datacode(home,NM);
	if (screen != 0) Rprintf("yM read\n");
	
	*yF = get_datafield_s2F(home,*Nrep, NF,&*s2F); /* raw field data */ 
	if (screen != 0) Rprintf("yF read\n");
		
	*y = dvector(0,N);
	dcopy_(&NM,&(*yM)[0],&one,&(*y)[0],&one);
	dcopy_(&NF,&(*yF)[0],&one,&(*y)[NM],&one);
	if (screen != 0) Rprintf("y set\n");

	*ZM = get_inputscode(home,NM,p); /* read model design set */
	if (screen != 0) Rprintf("ZM read\n");
	
	*ZF = get_inputsfield(home, NF, pcont); /* read field design set */
	if (screen != 0) Rprintf("ZF read\n");

	
	//CONSTRUCT GLOBAL DESIGN SET
	*Z=darray2(p,N);
	
	for(i=0;i<p;i++){
		for(j=0;j<N;j++){
			(*Z)[i][j]=0.0;
		}
	}
	
	for(i=0;i<p-pstar;i++){ /* Z = (ZM' ZF')' */
		dcopy_(&NM,(*ZM)[i],&one,(*Z)[i],&one);
		dcopy_(&NF,(*ZF)[i],&one,&(*Z)[i][NM],&one);
	}  
	for(i=p-pstar;i<p;i++){
		dcopy_(&NM,(*ZM)[i],&one,(*Z)[i],&one);
	}
	if (screen != 0) Rprintf("Z set\n");
	/* missing in Z are the values for the calibration parameters, 
	 if they exist */
	
	// CONSTRUCT GLOBAL DESIGN MATRIX
	
	*X= get_designMatrix(home, NF, NM, q);
	if (screen != 0) Rprintf("X read\n");
	*Xt=darray2(q,N);

	for(i=0;i<N;i++){
		for(j=0;j<q;j++){
			(*Xt)[j][i]=(*X)[i][j];
		}
	}
	if (screen != 0) Rprintf("Xt set\n");

	if (screen !=0){
		Rprintf("ThetaF is:\n");
		dprintvec(*thetaF, 2*(pcont)+2);
		Rprintf("ThetaM is:\n");
		dprintvec(*thetaM, 2*p+1);
		Rprintf("ThetaL is:\n");
		dprintvec(*thetaL, q); 
		Rprintf("prioriscales is:\n");
		dprintvec(*prioriscales, pcont+2);
		Rprintf("ubounds is:\n");
		dprintmat(*ubounds, pstar, 5);
		Rprintf("yF is:\n");
		dprintvec(*yF, NF);
		Rprintf("yM is:\n");
		dprintvec(*yM, NM);
		Rprintf("y is:\n");
		dprintvec(*y, N);
		Rprintf("NFtot is: %u\n",*NFtot);
		Rprintf("Nrep is:\n");
		iprintvec(*Nrep, NF);
		Rprintf("ZF is:\n");
		dprintmat(*ZF, pcont,NF);
		Rprintf("ZM is:\n");
		dprintmat(*ZM, p,NM);
		Rprintf("ZF is:\n");
		dprintmat(*Z, p,N);
		Rprintf("s2f is: %f\n",*s2F);
		Rprintf("X is:\n");
		dprintmat(*X, N,q);
		Rprintf("Xt is:\n");
		dprintmat(*Xt, q,N);
		Rprintf("FileF is: %s\n",*fileF);
		Rprintf("Filepath is: %s\n",*filepath);
		Rprintf("FileU is: %s\n",*fileU);
		Rprintf("FrateU is: %s\n",*frateU);
	}
}


void bayesfitSetup(int screen, char home[], double multmle, int p, int q, int pstar, int pcont, int NM, int NF, double **thetaF, 
				   double **thetaM, double **thetaL, double **priorshapes, double **prioriscales, 
				   double **yF, double **yM, double **y, int *NFtot, int **Nrep, double ***ZF, 
				   double ***ZM, double ***Z, double *s2F, double ***X, double ***Xt, char **fileF, char **filepath)
{
	if (screen != 0){
		Rprintf("\n--- Entra en bayesfitSetup ---\n");
		Rprintf("Home is: %s\n",home);
	}
	// FILENAMES
	char strtmp[100];
	
	/* file where sequence of thetaF goes */
	char afileF [100];
	strcpy(strtmp,home);
	strcat(strtmp,"thetaF.out");
	if (screen != 0) Rprintf("strtmp is: %s\n",strtmp);
	strcpy(afileF,strtmp);
	*fileF = malloc(sizeof(afileF));
	memcpy(*fileF, afileF, sizeof(afileF));
	//Rprintf("FileF is: %s\n",fileF);
	if (screen != 0){
		Rprintf("thetaF.out is: %s\n",*fileF);
	}
	
	/* file where sequence of (yM*,b) goes */
	char afilepath [100];
	strcpy(strtmp,home);
	strcat(strtmp,"filepath.out");
	strcpy(afilepath,strtmp);
	*filepath = malloc(sizeof(afilepath));
	memcpy(*filepath, afilepath, sizeof(afilepath));
	if (screen != 0){
		Rprintf("filepath.out is: %s\n",*filepath);
	}
	
	int i, j; //counters

	/*********************************/
	/****** INPUTS TO THE PROBLEM ****/
	/*********************************/
	if (screen != 0){
		Rprintf("\n--- Options: Files ---\n");
		
		// FILENAMES
		/* working directory path */
		Rprintf("working directory path=%s\n",home);

		Rprintf("OUTPUTS:\n");
		/* file where sequence of thetaF goes */
		Rprintf("file containing the stage I GASP mle of thetaF=%s\n",*fileF);
		
		/* file where sequence of (yM*,b) goes */
		Rprintf("file containing the sequence of (yM*,b) goes=%s\n",*filepath);
		
	}

	////////////////////////////////////////////////////////
	// READ MLE OF THETAF
	*thetaF = get_mlethetaF(home,pcont);
	if (screen != 0) Rprintf("thetaF read\n");
	// READ MLE OF THETAM
	*thetaM = get_mlethetaM(home,p);
	if (screen != 0) Rprintf("thetaM read\n");
	// READ MLE OF THETAL
	*thetaL= get_mlethetaL(home,q);
	if (screen != 0) Rprintf("thetaL read\n");
    
	
	//PRIORS
	*priorshapes=dvector(0,pcont+2);
	if (screen != 0) Rprintf("priorshapes set\n");
	*prioriscales=dvector(0,pcont+2);
    
	for(i=0;i<pcont+2;i++) (*priorshapes)[i]=1.;
    
    for(i=0;i<pcont+1;i++){
        (*prioriscales)[i]=(*thetaF)[i];
    }
	(*prioriscales)[pcont+1]=(*thetaF)[2*(pcont)+1];

    
	int k=pcont+2;
	int one=1;
	dscal_(&k,&multmle,&*prioriscales[0],&one);
	for(i=0;i<pcont+2;i++)
		(*prioriscales)[i]=1./(*prioriscales)[i];
	if (screen != 0) Rprintf("priorscales set\n");
    
	*Nrep = get_inputsrep(home, NF);
    if (screen != 0) Rprintf("Nrep read\n");

	//READ/CONSTRUCT DATA
	int N;       /* NM+NF */
	N = NM+NF;

	/* total number of field observations (this number includes 
	 the replicates)
	 compute total number of field observations */
	for(i=0;i<NF;i++){
		*NFtot = *NFtot+(*Nrep)[i];
	}
	if (screen != 0) Rprintf("NFtot set\n");
    
	*yM = get_datacode(home,NM);
   	if (screen != 0) Rprintf("yM read\n");

	*yF = get_datafield_s2F(home,*Nrep, NF,&*s2F); /* raw field data */
	if (screen != 0) Rprintf("yF read\n");
	
	*y = dvector(0,N);
	dcopy_(&NM,&(*yM)[0],&one,&(*y)[0],&one);
	dcopy_(&NF,&(*yF)[0],&one,&(*y)[NM],&one);
	if (screen != 0) Rprintf("y set\n");
	
	*ZM = get_inputscode(home,NM,p); /* read model design set */
	if (screen != 0) Rprintf("ZM read\n");
	
	*ZF = get_inputsfield(home, NF, pcont); /* read field design set */
	if (screen != 0) Rprintf("ZF read\n");
	
	//CONSTRUCT GLOBAL DESIGN SET
	*Z=darray2(p,N);
	
	for(i=0;i<p;i++){
		for(j=0;j<N;j++){
			(*Z)[i][j]=0.0;
		}
	}
	
	for(i=0;i<p-pstar;i++){ /* Z = (ZM' ZF')' */
		dcopy_(&NM,(*ZM)[i],&one,(*Z)[i],&one);
		dcopy_(&NF,(*ZF)[i],&one,&(*Z)[i][NM],&one);
	}  
	for(i=p-pstar;i<p;i++){
		dcopy_(&NM,(*ZM)[i],&one,(*Z)[i],&one);
	}
   	if (screen != 0) Rprintf("Z set\n");
	/* missing in Z are the values for the calibration parameters, 
	 if they exist */
	
	// CONSTRUCT GLOBAL DESIGN MATRIX
	
	*X=get_designMatrix(home, NF, NM,q);
   	if (screen != 0) Rprintf("X read\n");
	*Xt=darray2(q,N);
	
	for(i=0;i<N;i++){
		for(j=0;j<q;j++){
			(*Xt)[j][i]=(*X)[i][j];
		}
	}
	if (screen != 0) Rprintf("Xt set\n");

	if (screen !=0){
		Rprintf("ThetaF is:\n");
		dprintvec(*thetaF, 2*(pcont)+2);
		Rprintf("ThetaM is:\n");
		dprintvec(*thetaM, 2*p+1);
		Rprintf("ThetaL is:\n");
		dprintvec(*thetaL, q); 
		Rprintf("prioriscales is:\n");
		dprintvec(*prioriscales, pcont+2);
		Rprintf("yF is:\n");
		dprintvec(*yF, NF);
		Rprintf("yM is:\n");
		dprintvec(*yM, NM);
		Rprintf("y is:\n");
		dprintvec(*y, N);
		Rprintf("NFtot is: %u\n",*NFtot);
		Rprintf("Nrep is:\n");
		iprintvec(*Nrep, NF);
		Rprintf("ZF is:\n");
		dprintmat(*ZF, pcont,NF);
		Rprintf("ZM is:\n");
		dprintmat(*ZM, p,NM);
		Rprintf("ZF is:\n");
		dprintmat(*Z, p,N);
		Rprintf("s2f is: %f\n",*s2F);
		Rprintf("X is:\n");
		dprintmat(*X, N,q);
		Rprintf("Xt is:\n");
		dprintmat(*Xt, q,N);
		Rprintf("FileF is: %s\n",*fileF);
		Rprintf("Filepath is: %s\n",*filepath);
	}
}
