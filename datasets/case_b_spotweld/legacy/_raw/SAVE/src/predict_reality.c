/*************************************************
 *   Copyright Nov 12 2003                       *
 *  This file was created and copyrighted by the *
 *  National Institute of Statistical Sciences   *
 *************************************************/

#include "common.h"
#include "chol_pivot.h"
#include "setup.h"
#include "iostuff.h"
#include "predict_reality.h"

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <R.h>
#include <Rmath.h>
#include <Rdefines.h>
#include <R_ext/Lapack.h>
#include <R_ext/Print.h>


void predict_reality(int *FlagOutput,int *P, int *PSTAR, int *Q, int *nm, int *nf,
					int *nmnew, int *nfnew, double *TOL, int *SIM, int *nBurning, int * nThinning, char *homePath[])
{
  int screen = *FlagOutput; // Flag to print output if 1, history is printed to the screen */
	int p = *P;				/* total number of inputs */
	int pstar = *PSTAR;		/* number of calibration inputs */
	int q = *Q;				/* number of parameters in the linear model X thetaL */ 
	
	/* working directory path */
	char home[100];
	strcpy(home,*homePath);

	int pcont;   /* number of controllable inputs */
	pcont = p - pstar;
	if(pcont==0) pcont=1;

	// TOLERANCE FOR THE PIVOTING ALGORITHM	
	double tol = *TOL; /* tol = tolerance for the pivoting algorithm */
	
	//PARAMETERS FROM MCMC
	//scanf("%d",&semente);
	int sim = *SIM;			/* length of the chain */
	int burn = *nBurning;
	int thin = *nThinning;
	
	
	int state,i,j,k; /* iterators */
	
	int NMold = *nm;				/* NM is the size of the design set for the code data */
									/* number of observations in the code data */
	int NFold = *nf;				/* NF is the size of the design set for the field data,
								 ie, the number of DISTINCT inputs considered */
									/* number of unique inputs in the field data */
	
	int NMnew = *nmnew;				/* number of points at which the model is being predicted */
	int NFnew = *nfnew;				/* number of points at which the bias is being predicted */

	int Nold, Nnew;
	Nold = NMold + NFold;
	Nnew = NMnew + NFnew;
	
	int NM = NMnew + NMold;
	int NF = NFnew + NFold;
	
	int N;
	N = Nold + Nnew;

	// Debugging
	if (screen != 0) {
		Rprintf("Verbose output predictreality=%d\n",screen);
		Rprintf("\n--- Options ---\n");
		Rprintf("Total number of inputs=%d\n",p);
		Rprintf("Total number of calibration inputs=%d\n",pstar);
		Rprintf("Number of parameters on the linear model X thetaL=%d\n",q);
		Rprintf("Size of the design set for the code data=%d\n",NM);
		Rprintf("Size of the design set for the field data=%d\n",NF);
		Rprintf("length of simulation=%d\n",sim);
		Rprintf("burn-in of simulation=%d\n",burn);
		Rprintf("thinning of simulation=%d\n",thin);		
		Rprintf("tolerance for the pivoting algorithm =%g\n",tol);
		Rprintf("Home path %s\n",*homePath);
	}
	
  //int meanbias; /* if 1, the bias has an unknown mean; if zero then it has mean zero */

  //double *by;

  //int info; /* returns sucess status */
  int one=1;
  double onedouble=1.;
  //double minusone=-1.;
  double zero=0.;

  //char filemuB[100];

/*********************************/
/****** INPUTS TO THE PROBLEM ****/
/*********************************/

  //PARAMETERS FROM MCMC
  //int alphasfixedB; /* if 1 then alphas were kept fixed in the model */
  //int alphasfixedM; /* if 1 then alphas were kept fixed in the field */
  //alphasfixedM = 1; // scanf("%d",&alphasfixedM);
  //alphasfixedB =1; // scanf("%d",&alphasfixedB);
  //meanbias = 0; // scanf("%d",&meanbias);

  //FILE NAMES
  char strtmp[100];
  char filemcmcF[100] = "thetaF.dat"; /* file contaning the mcmc of lambdaB and 
									   lambdaF - stage II */
  strcpy(strtmp,home);
  strcat(strtmp,filemcmcF);
  strcpy(filemcmcF,strtmp);

  char filepath[100] = "real.dat"; /* file where the simulated paths go */
  strcpy(strtmp,home);
  strcat(strtmp,filepath);
  strcpy(filepath,strtmp);

  char fileU[100];
  char flearn[100];
  if(pstar!=0){
	/* file containing a sample from the calibration parameters - stage II*/
    strcpy(strtmp,home);
    strcat(strtmp,"ustar.dat");
    strcpy(fileU,strtmp);

    /* the vector indicating which are unmeasured */
    strcpy(strtmp,home);
    strcat(strtmp,"learn.dat");
    strcpy(flearn,strtmp);
  }

  /*
  if(meanbias!=0){
    scanf("%s",filemuB); 
    strcpy(strtmp,home);
    strcat(strtmp,filemuB);
    strcpy(filemuB,strtmp);
  }
  */

  //READ/CONSTRUCT DATA
  double *yM;
  yM = get_datacode(home, NMold);   /* read model data */

  int *Nrep;
  Nrep = get_inputsrep(home,NFold); /* compute total number of field observations */

  double *yF;
  yF = get_datafield(home, Nrep, NFold); /* read field data and compute means and sum of squares */

  /* construct data vector (stack model and field) */
  double *y;
  y = dvector(0,Nold);
  dcopy_(&NMold,&yM[0],&one,&y[0],&one);
  dcopy_(&NFold,&yF[0],&one,&y[NMold],&one);

  // PRIORS AND WHICH PARAMETERS TO LEARN ABOUT
  /* read matrix with bounds on u and indicator of learning */
  int *indlearn; /* vector indicating which calibration parameters you want to 
					learn about (1) and which you just want to sample from the
					prior */
  int plearn=0;  /* total number of "true" calibration parameters you want to learn; the rest are unmeasured inputs */
  double **ubounds; /* calibration parameters and priors */	
  if(pstar!=0){
    ubounds = get_ubounds(home, pstar);

    indlearn = ivector(0,pstar);
    ireadvec(flearn,indlearn,pstar);
    for(i=0;i<pstar;i++)
      plearn = plearn + indlearn[i]; 
  }


  // READ DESIGN SETS
  double **ZMold, **ZFold; /* design sets */
  /* old ones */
  ZMold = get_inputscode(home, NMold, p);
	
  ZFold = get_inputsfield(home, NFold, pcont);

  /* new ones */
  double **ZMnewaux;
  char inputscodenew[100] = "inputs_real.dat";
  strcpy(strtmp,home);
  strcat(strtmp,inputscodenew);
  strcpy(inputscodenew,strtmp);
	
  ZMnewaux = darray2(NMnew,pcont);
  int auxError = 0;
  auxError = dreadmat(inputscodenew,ZMnewaux,NMnew,pcont); /* read model design set */
  if (auxError!=0) Rprintf("Error while getting access to the file where the new model design set is =%s\n",inputscodenew);

  double **ZMnew; /* design sets */
  ZMnew = darray2(pcont,NMnew);
  for(i=0;i<pcont;i++){
	for(j=0;j<NMnew;j++){
		ZMnew[i][j]=ZMnewaux[j][i];
	}
  }
  free_darray2(ZMnewaux,NMnew,pcont);

  //CONSTRUCT GLOBAL DESIGN SET
  double **ZM;
  ZM=darray2(p,Nold+NMnew); /* ZM = [ZMold | ZFold | ZMnew] */
  for(i=0;i<pcont;i++){
	  dcopy_(&NMold,ZMold[i],&one,ZM[i],&one);
	  dcopy_(&NFold,ZFold[i],&one,&ZM[i][NMold],&one);
	  dcopy_(&NMnew,ZMnew[i],&one,&ZM[i][Nold],&one);
  }
	
  for(i=pcont;i<p;i++){
	  dcopy_(&NMold,ZMold[i],&one,ZM[i],&one);
  }
  /* missing are the values for the calibration parameters, if they exist */
	
  double **ZFnew; /* design sets */
  double **ZFnewaux;
  char inputsfieldnew[100] = "inputs_real.dat";
  strcpy(strtmp,home);
  strcat(strtmp,inputsfieldnew);
  strcpy(inputsfieldnew,strtmp);

  ZFnew = darray2(pcont,NFnew);
  ZFnewaux = darray2(NFnew,pcont);
  auxError = dreadmat(inputsfieldnew,ZFnewaux,NFnew,pcont); /* read field design set */
  if (auxError!=0) Rprintf("Error while getting access to the file where the new field design set is =%s\n",inputsfieldnew);
  for(i=0;i<pcont;i++){
    for(j=0;j<NFnew;j++){
      ZFnew[i][j]=ZFnewaux[j][i];
    }
  }
  free_darray2(ZFnewaux,NFnew,pcont);
  
  //CONSTRUCT GLOBAL DESIGN SET
  double **ZF;
  ZF=darray2(pcont,NF); /* ZF = [ZFold | ZFnew] */
  for(i=0;i<pcont;i++){
    dcopy_(&NFold,ZFold[i],&one,ZF[i],&one);
    dcopy_(&NFnew,ZFnew[i],&one,&ZF[i][NFold],&one);
  }

  // CONSTRUCT GLOBAL DESIGN MATRIX
  double **XMoldaux, **XFoldaux, **XMnewaux; /* design matrices */
  strcpy(strtmp,home);
  strcat(strtmp,"predictionsII.design.M.old.matrix.dat");
  XMoldaux=darray2(NMold,q);
  dreadmat(strtmp,XMoldaux,NMold,q);
  strcpy(strtmp,home);
  strcat(strtmp,"predictionsII.design.F.old.matrix.dat");
  XFoldaux=darray2(NFold,q);
  dreadmat(strtmp,XFoldaux,NFold,q);
  strcpy(strtmp,home);
  strcat(strtmp,"predictionsII.design.M.new.matrix.dat");
  XMnewaux=darray2(NMnew,q);
  dreadmat(strtmp,XMnewaux,NMnew,q);

  double **X, **Xt; /* design matrix and its transpose */
  X=darray2(N,q);
  for(j=0;j<q;j++){
    for(i=0;i<NMold;i++){
      X[i][j]=XMoldaux[i][j];
    }
    for(i=0;i<NFold;i++){
      X[i+NMold][j]=XFoldaux[i][j];
    }
    for(i=0;i<NMnew;i++){
      X[i+Nold][j]=XMnewaux[i][j];
    }
    for(i=0;i<NFnew;i++){
      X[i+Nold+NMnew][j]=0.0;
    }
  }
  Xt=darray2(q,N);
  for(i=0;i<N;i++){
    for(j=0;j<q;j++){
      Xt[j][i]=X[i][j];
    }
  }

  free_darray2(XMoldaux,NMold,q);
  free_darray2(XFoldaux,NFold,q);
  free_darray2(XMnewaux,NMnew,q);

  //dprintmat(X,N,q); exit(1);
/**** INITIALIZATIONS ************/	
  double *thetaF;
  double *thetaM;
  double *thetaL;
  thetaF = get_mlethetaF(home,pcont); /* contanins the mle of thetaF - stage II */
  thetaM = get_mlethetaM(home,p); /* contains the mle of thetaM - stage I */
  thetaL = get_mlethetaL(home,q); /* contains the mle of thetaL - stage I */

  double *ustar, *ustaraux; /* calibration parameters and priors */
  FILE *fu;
  if(pstar!=0) {
	  fu = fopen(fileU,"r");
	  ustar = dvector(0,pstar);
	  ustaraux = dvector(0,pstar);
  }

  double ***DM; /* distance matrices */
  double ***sigmasM;
  double **sigmaM;
  DM = darray3(p,Nold+NMnew,Nold+NMnew);
  sigmasM = darray3(p,Nold+NMnew,Nold+NMnew);
  sigmaM = darray2(Nold+NMnew,Nold+NMnew);
   
  double ***DF; /* distance matrices */
  double ***sigmasB;
  double **sigmaB;
  DF = darray3(pcont,NF,NF);
  sigmasB = darray3(pcont,NF,NF);
  sigmaB = darray2(NF,NF);

  double **var;
  var=darray2(N,N);
  for(i=0;i<N;i++){
    for(j=0;j<N;j++){
      var[i][j]=0.0;
    }
  }

  double **sigma11, **sigma11inv, **sigma22, **sigma21, **L;
  int *pivot;
  sigma11 = darray2(Nold,Nold);
  sigma11inv = darray2(Nold,Nold);
  L = darray2(Nold,Nold);
  pivot = ivector(0,Nold);
  sigma22 = darray2(Nnew,Nnew);
  sigma21 = darray2(Nnew,Nold);

  double *mu;	
  mu=dvector(0,N); /* global mean */
  for(i=0;i<N;i++) mu[i]=0.;

  double *munew, *muold; 	
  munew=dvector(0,Nnew);
  muold=dvector(0,Nold);

  double *path;
  path=dvector(0,Nnew);

  FILE *fcal;
  fcal=fopen(filepath,"w"); // this is just to clear the file
  fprintf(fcal,"%s","");
  fclose(fcal);
	
	
  /* this is for the lapack calls */
  //char uplo[]="U"; /* it means we are feeding the routine with the upper triangle */
  char trans[]="T";/* it means we want to solve the system with the 
					  transpose of the matrix we are feeding it */
  //char diag[]="N"; /* means our matrix is not unit-triangular */
	

  dgemv_(trans,&q,&N,&onedouble,&X[0][0],&q,&thetaL[0],
	 &one,&zero,&mu[0],&one); /* mu=X theta^L */
  
  FILE *fmcmc;
  fmcmc = fopen(filemcmcF,"r");

  getD(ZF,DF,NF,thetaF,pcont);
  getsigmas(thetaF,DF,sigmasB,NF,pcont);
  getsigma(sigmaB,thetaF,sigmasB,NF,pcont);

  /********** START ****************/
  int rank = 0;
  int res;
  /********* LOOP ************************/
  for(state=0;state<sim;state++){
	R_CheckUserInterrupt();

    res = fscanf(fmcmc,"%lf",&thetaF[0]);
    res = fscanf(fmcmc,"%lf",&thetaF[2*(pcont)+1]);

    for(j=0;j<pstar;j++){
    	  res = fscanf(fu,"%lf",&ustar[j]);
    }

    if(state>burn && fmod((double) state, (double) thin)==0.){
      
      if (pstar!=0){
    	  dcopy_(&pstar,&ustar[0],&one,&ustaraux[0],&one);

    	  if(plearn < pstar) getpriordraws(ustar,pstar,indlearn,ubounds);
      
    	  for(i=p-pstar;i<p;i++){
    		  for(j=NMold;j<Nold;j++){
    			  ZM[i][j]=ustar[i-(p-pstar)];
    		  }
    		  for(j=Nold;j<Nold+NMnew;j++){
    			  ZM[i][j]=ustaraux[i-(p-pstar)];
    		  }
    	  }
      }

      getD(ZM,DM,Nold+NMnew,thetaM,p);
      getsigmas(thetaM,DM,sigmasM,Nold+NMnew,p);
      getsigma(sigmaM,thetaM,sigmasM,Nold+NMnew,p);

      for(i=0;i<N;i++){
    	  for(j=0;j<N;j++){
    		  var[i][j]=0.0;
    	  }
      }

      /* construct global covariance matrix */
      getvar_pr(sigmaM,sigmaB,var,thetaM,thetaF,Nrep,NMold,NMnew,
	     NFold,NFnew,p,pstar);

      //dwritemat("var.tmp",var,N,N); exit(1);

      for(k=0;k<Nold;k++){ 
    	  sigma11[k][k]=var[k][k];
    	  for(j=0;j<k;j++){
    		  sigma11[k][j]=var[k][j];
    		  sigma11[j][k]=sigma11[k][j];
    	  }
      }

      for(k=0;k<Nnew;k++){ 
    	  sigma22[k][k]=var[k+Nold][k+Nold];
    	  for(j=0;j<k;j++){
    		  sigma22[k][j]=var[k+Nold][j+Nold];
    		  sigma22[j][k]=sigma22[k][j];
    	  }
      }

      for(k=0;k<Nnew;k++){ 
    	  for(j=0;j<Nold;j++){
    		  sigma21[k][j]=var[k+Nold][j];
    	  }
      }

      dcopy_(&Nold,&mu[0],&one,&muold[0],&one);
      dcopy_(&Nnew,&mu[Nold],&one,&munew[0],&one);

      chol_pivot(sigma11, L, pivot, &rank, tol, Nold);
      
      inverse_using_chol_pivot(L, sigma11inv, pivot, Nold);

      gen_cond_mv_normal(path, y, muold, munew, sigma11inv, sigma21, sigma22, 
			 tol, onedouble, Nold, Nnew);
      
	  auxError = dappendvecnotverb(filepath,path,Nnew);  
	  if (auxError!=0) Rprintf("Error while getting access to the file where the real goes =%s\n",filepath);
    
	}
  }
  if(pstar!=0)fclose(fu);
  Rprintf("\n--- Finished ---\n");
}


/*********************************/
/*           FUNCTIONS           */
/*********************************/

void getpriordraws(double *ustar, int pstar, int *indlearn, double **bounds){
  int i;
  double sigma2, tmp;
  double mean;
  //double p;

  if (pstar != 0){
	  GetRNGstate();
	  for(i=0;i<pstar;i++){
		  if(indlearn[i]==0){ // draw from prior
      
			  if(bounds[i][0]==0.0)
				  ustar[i]= runif(0.0,1.0)*(bounds[i][2]-bounds[i][1]) + bounds[i][1];
			  else{
				  mean = bounds[i][3];
				  sigma2 = bounds[i][4];
				  tmp = rnorm(mean,sqrt(sigma2));
				  while(tmp<bounds[i][1] || tmp>bounds[i][2]){
					  tmp = rnorm(mean,sqrt(sigma2));
				  }
				  ustar[i]=tmp;
			  }
		  }
	  }
	  PutRNGstate();
  } else Rprintf ("WARNIGN!:Trying to draw priors when there are no calibration parameters.\n");
}

void getvar_pr(double **RM, double **RB, double **result, double *parM,
	    double *parF, int *reps, int NMold, int NMnew, int NFold, 
	    int NFnew, int p, int pstar)
{
  int i,j;
  int N,Nold,NM;
  int pcont;

  N = NFold + NFnew + NMnew + NMold;
  Nold = NFold + NMold;
  NM = NMnew + NMold;

  pcont=p-pstar;
  if(pcont==0) pcont=1;

  for(i=0;i<N-NFnew;i++){
    result[i][i]=1./parM[0];
    for(j=0;j<i;j++){
      result[i][j]=RM[i][j]/parM[0];
    }
  }
    
  for(i=NMold;i<Nold;i++){
    result[i][i]=1./parM[0]+1./parF[0]+1./(reps[i-NMold]*parF[2*(pcont)+1]);
    for(j=NMold;j<i;j++){
      result[i][j]=result[i][j]+RB[i-NMold][j-NMold]/parF[0];
    }
  }
  
  for(i=N-NFnew;i<N;i++){
    for(j=NMold;j<Nold;j++){
      result[i][j]=RB[i-NM][j-NMold]/parF[0];
    }
  }

  for(i=N-NFnew;i<N;i++){
    result[i][i]=1./parF[0];
    for(j=N-NFnew;j<i;j++){
      result[i][j]=RB[i-NM][j-NM]/parF[0];
    }
  }
  //dprintmat(RB,NFold + NFnew,NFold + NFnew);exit(1);
}
