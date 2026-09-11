/*************************************************
 *   Copyright Nov 12 2003                       *
 *  This file was created and copyrighted by the *
 *  National Institute of Statistical Sciences   *
 *************************************************/

/***************************************************************
  Predict code output at a fixed set of inputs

-> Produce realizations from the posterior distribution of the model
output yM. This is part of producing a fast approximation to the
code output.

-> mcmc.model.c has been used to produce an MCMC sample from the
posterior distribution of the parameters involved in the parametric
specification of the mean and covariance function of the Gaussian
process

-> the other option is we have computed the MLE of the stage I
parameters and want to produce realizations from the ensuing
Gaussian process

-> this source code is written for the second option
***************************************************************/

#include "common.h"
#include "chol_pivot.h"
#include "setup.h"
#include "predict_code.h"
#include "iostuff.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <R.h>
#include <Rmath.h>
#include <Rdefines.h>
#include <R_ext/Lapack.h>
#include <R_ext/Print.h>

void predict_code(int *FlagOutput,int *P, int *Q, int *nm,
			int *nmnew, double *TOL, int *SIM, int *ismleormcmc, int *sampledraws, char *homePath[])
{
	int screen = *FlagOutput; // Flag to print output if 1, history is printed to the screen */
	int p = *P;      /* number of inputs */
	int q = *Q;      /* number of parameters of the linear model X thetaL */
					 /* number of parameters in the linear model for the mean */

	/*
	 sampledraws is a flag where the user selects 
	 if the draws are obtained (sampledraws = 1) or not (sampledraws = 0)
	 In case the (sampledraws = 0) only conditional mean and cholesky decomposition will be
	 computed.
	 */
	
	// TOLERANCE FOR THE PIVOTING ALGORITHM	
	double tol = *TOL; /* tol = tolerance for the pivoting algorithm */
	//PARAMETERS FROM MCMC
	int sim = *SIM;			/* length of the chain */
	int burn ;	/* burn-in */
	int thin ;	/* thinning */

	int Nold = *nm; 		/* number of observations in the code data */
	int Nnew = *nmnew;		/* dimension of the grid at which to predict, */
	int N;					/* dimension of the design set and N=Nnew+Nold */
	N=Nnew+Nold;

	int stageonemle = *ismleormcmc; /* if 0, there an mcmc sample from the posterior; if 1,
					  only the MLe has been computed */

	char home[100]; /* working directory path */
	strcpy(home,*homePath);
	
	//int alphasfixed;	
	//alphasfixed = 1;//scanf("%d",&alphasfixed);
	
//	int r;      /* number of parameters excluding the ones in the linear model */
//	r = 2*p+1; /* total number of parameters in the covariance structure */
	
  //int semente;
	
	int i,j,k,s;  /* iterators */

/* this is for the lapack calls */
  //char uplo[]="U"; /* it means we are feeding the routine with the 
  //		      upper triangle */
	char trans[]="T";/* it means we want to solve the system with the 
					  transpose of the matrix we are feeding it */
  //char diag[]="N"; /* means our matrix is not unit-triangular */
  //int info; /* returns sucess status */
	int one=1;
	double onedouble=1.;
  //double minusone=-1.;
	double zero=0.;

	char krig[100], mean_vecfile[100], cov_matfile[100], strtmp[100];

/*********************************/
/****** INPUTS TO THE PROBLEM ****/
/*********************************/

	// Debugging
	if (screen != 0) {
		Rprintf("Verbose output predictcode=%d\n",screen);
		Rprintf("\n--- Options ---\n");
		Rprintf("total number of inputs=%d\n",p);
		Rprintf("dimension of linear model=%d\n",q);
		Rprintf("alphas and betas are fixed at mle\n");
		Rprintf("dimension of old code design=%d\n",Nold);
		Rprintf("dimension of new code design=%d\n",Nnew);
		Rprintf("length of simulation=%d\n",sim);
		Rprintf("burn-in of simulation=%d\n",burn);
		Rprintf("thinning of simulation=%d\n",thin);
		Rprintf("tolerance for the pivoting algorithm =%g\n",tol);
		Rprintf("Home path %s\n",*homePath);
  //scanf("%d",&burn);
  //scanf("%d",&thin);
	}

  //READ DATA
	double *y;  /* data */
	y = get_datacode(home, Nold); /* file where the model data are */
	if (screen != 0) Rprintf("model data read\n");

	double **Zold; /* points in each 1-dimensional space (by lines) */
	Zold = get_inputscode (home, Nold, p); /* old design set */
							/* points in each 1-dimensional space (by columns) */
	 /* read old design set */ /* file where the old inputs are */
	if (screen != 0) Rprintf("old design read\n");

	char inputsinnew[100] = "inputs_pure.dat";	/* file where the new inputs are */
	strcpy(strtmp,home);
	strcat(strtmp,inputsinnew);
	strcpy(inputsinnew,strtmp);
	double **Znewaux;	
	Znewaux = darray2(Nnew,p); /* new design set */
	
	int auxError = 0;
	auxError = dreadmat(inputsinnew,Znewaux,Nnew,p);
	if (auxError!=0) Rprintf("Error while getting access to the file where the new inputs are =%s\n",inputsinnew);
	double **Znew;
	Znew = darray2(p,Nnew); /* new design set */
	for(i=0;i<p;i++){
		for(j=0;j<Nnew;j++){
			Znew[i][j]=Znewaux[j][i];
		}
	}
	if (screen != 0) Rprintf("new inputs read\n");

	free_darray2(Znewaux,Nnew,p);
  
  //CONSTRUCT GLOBAL DESIGN SET
	double **Z; /* Z= (Zold Znew) */
	Z = darray2(p,N); /* Z contains the design points by columns */
	for(i=0;i<p;i++){
		dcopy_(&Nold,Zold[i],&one,Z[i],&one);
		dcopy_(&Nnew,Znew[i],&one,&Z[i][Nold],&one);
	}
	if (screen != 0) Rprintf("Z built\n");
	
  //CONSTRUCT GLOBAL DESIGN MATRIX
	double **Xnew, **Xold; /* design matrices */
	strcpy(strtmp,home);
	strcat(strtmp,"predictionsI.design.old.matrix.dat");
	Xold = darray2(N,q);
	dreadmat(strtmp,Xold,Nold,q);
	strcpy(strtmp,home);
	strcat(strtmp,"predictionsI.design.new.matrix.dat");
	Xnew = darray2(N,q);
	dreadmat(strtmp,Xnew,Nnew,q);
	if (screen != 0) Rprintf("Xnew read\n");

	double **Xt, **X; /* design matrix and its transpose */
	X = darray2(N,q); /* X is the global design matrix */
	Xt = darray2(q,N);
	
	for(j=0;j<q;j++){
		for(i=0;i<Nold;i++){
			X[i][j]=Xold[i][j];
		}
		for(i=0;i<Nnew;i++){
			X[i+Nold][j]=Xnew[i][j];
		}
	}
	if (screen != 0) Rprintf("X built\n");

	free_darray2(Xnew,Nnew,q);
	free_darray2(Xold,Nold,q);

	for(i=0;i<N;i++){
		for(j=0;j<q;j++){
			Xt[j][i]=X[i][j];
		}
	}
	if (screen != 0) Rprintf("Xt built\n");

/********* INICIALIZATIONS *******/
	double *path;
	path=dvector(0,Nnew); /* simulated path */

	double ***D;/* stores the "distances" (x_i-x_j)^alpha_k, k=1:p */
	D = darray3(p,N,N); /* vector of matrices |x_i-x_j|^alpha_k */

	double ***sigmas; /* individual correlation matrices */	
	sigmas=darray3(p,N,N); /* vector of matrices sigma_k */

	double **sigma; /* correlation matrix, but content changes throughout 
					 the code */
	sigma=darray2(N,N);  /* correlation matrix */

	double **sigma11, **sigma11inv, **sigma22, **sigma21, **L;
	sigma11=darray2(Nold,Nold);
	sigma11inv=darray2(Nold,Nold);
	sigma22=darray2(Nnew,Nnew);
	sigma21=darray2(Nnew,Nold);
	L=darray2(Nold,Nold);

	int *pivot;
	pivot=ivector(0,Nold);

	double *theta; /* current value of the parameter */
					/* vector of lambda + betas and alphas */
					/* file where the sample of thetaM, or its mle, is */
	double *thetaL;

	double *mu; /* mean of the observed vector, mu=X thetaL */
	mu=dvector(0,N);
	for(i=0;i<N;i++) mu[i]=0.;

	double *munew, *muold, *mean_vec; 
	munew=dvector(0,Nnew);
	muold=dvector(0,Nold);
	mean_vec = dvector(0, Nnew);
	
	double **chol_cov_mat, **cov_mat;
	chol_cov_mat = darray2(Nnew, Nnew);
	cov_mat = darray2(Nnew, Nnew);


/********** START ****************/
	//FILE NAMES
	strcpy(strtmp,home);
	strcat(strtmp,"path_pure.dat"); /* file where the sample paths go */
	strcpy(krig,strtmp);

	strcpy(strtmp,home);
	strcat(strtmp,"mean_vector.dat"); /* file where the conditional mean goes */
	strcpy(mean_vecfile,strtmp);

	strcpy(strtmp,home);
	strcat(strtmp,"cov_mat.dat"); /* file where the cholesky decomposition goes */
	strcpy(cov_matfile,strtmp);
	
	FILE *fF;
	fF=fopen(krig,"w");
	fprintf(fF,"%s","");
	fclose(fF); 

	fF=fopen(mean_vecfile,"w");
	fprintf(fF,"%s","");
	fclose(fF); 

	fF=fopen(cov_matfile,"w");
	fprintf(fF,"%s","");
	fclose(fF); 
	
	//stageonemle; /* if 0, there an mcmc sample from the posterior; if 1,
	//				  only the MLe has been computed */
	
	int rank = 0;

  /********* LOOP ************************/
	for(i=0;i<sim;i++){

		R_CheckUserInterrupt();
		if(stageonemle!=1 || i==0){
			theta = get_mlethetaM (home,p);   
			if (screen != 0) Rprintf("thetaM read\n");
			thetaL = get_mlethetaL(home, q);
			if (screen != 0) Rprintf("thetaL read\n");
		}
    
		switch(stageonemle){
		case 1:
			if(i==0){
                if (screen != 0) Rprintf("Before dgemv_ case 1\n");
				dgemv_(trans,&q,&N,&onedouble,&X[0][0],&q,&thetaL[0], 
					   &one,&zero,&mu[0],&one); /* mu=X theta^L */
                if (screen != 0) Rprintf("After dgemv_ case 1\n");

				getD(Z,D,N,theta,p);
				getsigmas(theta,D,sigmas,N,p);
				getsigma(sigma,theta,sigmas,N,p);
	  
				for(k=0;k<Nold;k++){ 
					sigma11[k][k]=sigma[k][k];
					for(s=0;s<k;s++){
						sigma11[k][s]=sigma[k][s];
						sigma11[s][k]=sigma11[k][s];
					}
				}

				for(k=0;k<Nnew;k++){ 
					sigma22[k][k]=sigma[k+Nold][k+Nold];
					for(s=0;s<k;s++){
						sigma22[k][s]=sigma[k+Nold][s+Nold];
						sigma22[s][k]=sigma22[k][s];
					}
				}

				for(k=0;k<Nnew;k++){
					for(s=0;s<Nold;s++){
						sigma21[k][s]=sigma[k+Nold][s];
					}
				}

				dcopy_(&Nold,&mu[0],&one,&muold[0],&one);
				dcopy_(&Nnew,&mu[Nold],&one,&munew[0],&one);

				chol_pivot(sigma11, L, pivot, &rank, tol, Nold);
	  
				inverse_using_chol_pivot(L, sigma11inv, pivot, Nold);
				
				// Computes the conditional mean and covariance
				dcopy_(&Nnew,&munew[0],&one,&mean_vec[0],&one);
				if (auxError!=0) Rprintf("Computing the conditional parameters\n");
				//Rprintf("mean_vec[0]");
				//Rprintf("is:%f\n",mean_vec[0]);
				if (screen != 0) {
					char chol_mat[100] ="cov_mat_initial.dat";
					strcpy(strtmp,home);
					strcat(strtmp,chol_mat);
					strcpy(chol_mat,strtmp);
					Rprintf("Antes del calculo\n");
					dappendmat(chol_mat, chol_cov_mat, Nnew, Nnew);
				}
				compute_cond_parameters(y, muold, sigma11inv, sigma21, 
										sigma22, tol, 1./theta[0], Nold, 
										Nnew,&mean_vec,&cov_mat,&chol_cov_mat);
				if (screen != 0) {
					char cov_mat_computed[100] ="cov_mat_computed.dat";
					strcpy(strtmp,home);
					strcat(strtmp,cov_mat_computed);
					strcpy(cov_mat_computed,strtmp);
					Rprintf("Despues del calculo\n");
					dappendmat(cov_mat_computed, cov_mat, Nnew, Nnew);
				}
				auxError = dappendvecnotverb(mean_vecfile,mean_vec,Nnew);
				if (auxError!=0) Rprintf("Error while getting access to the file where the mean_vector goes =%s\n",mean_vecfile);

				auxError = dappendmat(cov_matfile, cov_mat, Nnew, Nnew);
				if (auxError!=0) Rprintf("Error while getting access to the file where the covariate matrix goes =%s\n",cov_matfile);
				
				}
				
				if (screen != 0) Rprintf("Sample draws is %d\n",*sampledraws);
				if (*sampledraws == 1) {
					/* Generate from the multivariate normal */
					//Rprintf("Sampling from the multivariate normal.\n");
					rmvnormd_pivot(mean_vec, chol_cov_mat, Nnew, path);
				} 
				else {
					if (screen != 0) Rprintf("Only the conditional mean and cholesky decomposition is computed\n");
					i = sim; // so we QUIT the loop and avoid sampling
					break;
				}

			auxError = dappendvecnotverb(krig,path,Nnew);
			if (auxError!=0) Rprintf("Error while getting access to the file where the real goes =%s\n",krig);
	
			break;
		case 0:
			if(i>burn && fmod((double) i, (double) thin)==0.){
                if (screen != 0) Rprintf("Before dgemv_ case 2\n");
				dgemv_(trans,&q,&N,&onedouble,&X[0][0],&q,&thetaL[0],
					   &one,&zero,&mu[0],&one); /* mu=X theta^L */
                if (screen != 0) Rprintf("After dgemv_ case 2\n");
	  
				getD(Z,D,N,theta,p);
				getsigmas(theta,D,sigmas,N,p);
				getsigma(sigma,theta,sigmas,N,p);

				for(k=0;k<Nold;k++){ 
					sigma11[k][k]=sigma[k][k];
					for(s=0;s<k;s++){
						sigma11[k][s]=sigma[k][s];
						sigma11[s][k]=sigma11[k][s];
					}
				}

				for(k=0;k<Nnew;k++){ 
					sigma22[k][k]=sigma[k+Nold][k+Nold];
					for(s=0;s<k;s++){
						sigma22[k][s]=sigma[k+Nold][s+Nold];
						sigma22[s][k]=sigma22[k][s];
					}
				}

				for(k=0;k<Nnew;k++){ 
					for(s=0;s<Nold;s++){
						sigma21[k][s]=sigma[k+Nold][s];
					}
				}

				dcopy_(&Nold,&mu[0],&one,&muold[0],&one);
				dcopy_(&Nnew,&mu[Nold],&one,&munew[0],&one);
	  
				chol_pivot(sigma11, L, pivot, &rank, tol, Nold);
	  
				inverse_using_chol_pivot(L, sigma11inv, pivot, Nold);
	  
				gen_cond_mv_normal(path, y, muold, munew, sigma11inv, sigma21, 
								   sigma22, tol, 1./sqrt(theta[0]), Nold, Nnew);

				auxError = dappendvecnotverb (krig,path,Nnew);
				if (auxError!=0) Rprintf("Error while getting access to the file where the real goes =%s\n",krig);
	
			}
			break;
		}
	}
	Rprintf("\n--- Finished ---\n");
}
