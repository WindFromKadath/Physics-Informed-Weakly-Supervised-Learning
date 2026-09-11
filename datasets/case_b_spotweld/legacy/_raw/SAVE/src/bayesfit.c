/*************************************************
 *  Copyright Nov 12 2003                        *
 *  This file was created and copyrighted by the *
 *  National Institute of Statistical Sciences   *
 *************************************************/

#include "common.h"
#include "bayesfit.h"
#include "setup.h"
#include "iostuff.h"
#include "bayesfitSetup.h"


#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <R.h>
#include <Rmath.h>
#include <Rdefines.h>
#include <R_ext/Lapack.h>
#include <R_ext/Print.h>
#include <R_ext/Utils.h>

/*#define PI 3.141592653589793 */
#define PROP 0.3

void bayesfit(int *FlagOutput,int *P, int *PSTAR, int *Q, int *nm, int *nf,
			  double *PROB, double *MULTMLE, int *FlagMethod, int *SIM, 
			  int *METROP, char *homePath[])
{

  void R_CheckUserInterrupt(void);
  int screen = *FlagOutput; // Flag to print output if 1, history is printed to the screen */
  int p = *P;				/* total number of inputs */
  int pstar = *PSTAR;		/* number of calibration inputs */
  int q = *Q;				/* number of parameters in the linear model X thetaL */ 
  int NM = *nm;				/* NM is the size of the design set for the code data */
  int NF = *nf;				/* NF is the size of the design set for the field data,
								 ie, the number of DISTINCT inputs considered */

  // Debugging
	if (screen != 0) {
		Rprintf("Verbose output=%d\n",screen);
		Rprintf("\n--- Options ---\n");
		Rprintf("Total number of inputs=%d\n",p);
		Rprintf("Total number of calibration inputs=%d\n",pstar);
		Rprintf("Number of parameters on the linear model X thetaL=%d\n",q);
		Rprintf("Size of the design set for the code data=%d\n",NM);
		Rprintf("Size of the design set for the field data=%d\n",NF);
		Rprintf("Home path %s\n",*homePath);
	}
	
  //PARAMETERS FOR MCMC
  // To get the RNG state from R when this C fuction is call from R
  GetRNGstate();
	
  double prob = *PROB;		//probability of drawing from the prior
  double multmle = *MULTMLE; // hyperparameter
  int method = *FlagMethod;  /* method of sampling; */
  int sim = *SIM;			/* length of the chain */
  int metrop = *METROP;		/* length of metropolis chain */
  char home[100];
  strcpy(home,*homePath);

/*
 meanbias is a variable that is always equal to zero, so that's why qB won't be used and 
 the variable meanbias is not a parameter in the fuction call.
*/
//	
// int meanbias = *MEANBIAS; /* if 1, the bias has an unknown mean; if zero then it has 
//							 mean zero */
	int meanbias = 0;
	if (screen != 0){
		// Debugging
		Rprintf("Meanbias=%d\n",meanbias);
		Rprintf("Probability of drawing from the prior=%lf\n",prob);
		Rprintf("Hyperparameter =%lf\n",multmle);
		Rprintf("Sampling method=%d\n",method);
		Rprintf("Length of the chain=%d\n",sim);
		Rprintf("Length of the metropolis chain=%d\n",metrop);
		Rprintf("Home path %s\n",home);
	}
	

  double ll, llaux, lp, lpaux; // to compute acceptance ratio
  double ltarget;

  int i,j,state,r;  /* iterators */

  /* this is for the lapack calls */
  char uplo[]="U"; /* it means we are feeding the routine with the 
		      upper triangle */
  char trans[]="T";/* it means we want to solve the system with the 
		      transpose of the matrix we are feeding it */
  int info; /* returns sucess status */

  double onedouble=1.;
  double zero=0.;

  // FILENAMES
  FILE *fF, *fcal;
  char *fileF; /* file where sequence of thetaF goes */
  char *filepath; /* file where sequence of (yM*,b) goes */
  char *fileU; /* file where the sequence of ustars goes */
  char *frateU; /* and its acceptance rate */	

	
  // READ MLE OF THETAF
  int pcont;   /* number of controllable inputs */
  pcont = p-pstar;
  if(pcont==0) pcont=1; /* if there are no controllable inputs, alphaB and betaB
			   won't matter, but we'll keep a slot in thetaF for
			   one alphaB and one betaB (fixed at 2 and 1, resp.)
			*/
  
  double *thetaF; // parameter vectors
  double *thetaM;
  double *thetaL;

  double *priorshapes, *prioriscales; /* hyperparameters */

  double **ubounds; // calibration parameters and bounds
  ubounds = NULL;

  double *yM, *yF, *y;  /* data -- model and field (averages), and stacked */

  int *Nrep;   /* vector containing the number of replicates at each
				  distinct input */
   
  int NFtot = 0;   /* total number of field observations (this number includes 
					  the replicates) */

  double **ZF;  /* design sets */
  double **ZM, **Z;
  double s2F; /* data -- sum of squared deviations from individual means */

  double **X, **Xt; /* design matrix and its transpose */

  if (screen != 0){
		Rprintf("\n--- Setting up the input data and creating output files ---\n");
  }

  if (pstar != 0)
	  bayesfitSetupCalib(screen,home,multmle,p,q,pstar,pcont,NM,NF,&thetaF,&thetaM,&thetaL,&priorshapes,&prioriscales,
				&ubounds,&yF,&yM,&y,&NFtot,&Nrep,&ZF,&ZM,&Z,&s2F,&X,&Xt,&fileF,&filepath,&fileU,&frateU);	
  else bayesfitSetup(screen,home,multmle,p,q,pstar,pcont,NM,NF,&thetaF,&thetaM,&thetaL,&priorshapes,&prioriscales,
			&yF,&yM,&y,&NFtot,&Nrep,&ZF,&ZM,&Z,&s2F,&X,&Xt,&fileF,&filepath);


  if (screen != 0){
	  Rprintf("\n--- Finished. Ready to start MCMC ---\n");
	  Rprintf("ThetaF is:\n");
	  dprintvec(thetaF, 2*(pcont)+2);
	  Rprintf("ThetaM is:\n");
	  dprintvec(thetaM, 2*p+1);
	  Rprintf("ThetaL is:\n");
	  dprintvec(thetaL, q); 
	  Rprintf("prioriscales is:\n");
	  dprintvec(prioriscales, pcont+2);
	  Rprintf("ubounds is:\n");
	  dprintmat(ubounds, pstar, 5);
	  Rprintf("yF is:\n");
	  dprintvec(yF, NF);
	  Rprintf("yM is:\n");
	  dprintvec(yM, NM);
	  Rprintf("y is:\n");
	  dprintvec(y, NM+NF);
	  Rprintf("NFtot is: %u\n",NFtot);
	  Rprintf("Nrep is:\n");
	  iprintvec(Nrep, NF);
	  Rprintf("ZF is:\n");
	  dprintmat(ZF, pcont,NF);
	  Rprintf("ZM is:\n");
	  dprintmat(ZM, p,NM);
	  Rprintf("ZF is:\n");
	  dprintmat(Z, p,NM+NF);
	  Rprintf("s2f is: %f\n",s2F);
	  Rprintf("X is:\n");
	  dprintmat(X, NM+NF,q);
	  Rprintf("Xt is:\n");
	  dprintmat(Xt, q,NM+NF);
	  Rprintf("FileF is: %s\n",fileF);
	  Rprintf("Filepath is: %s\n",filepath);
	  Rprintf("FileU is: %s\n",fileU);
	  Rprintf("FrateU is: %s\n",frateU);
  }

  //READ/CONSTRUCT DATA
  int N;
  N = NM+NF;
  int one=1;

  //CONSTRUCT GLOBAL DESIGN SET
  double **Zaux;
  Zaux=darray2(p,N);
  
  for(i=0;i<p;i++){
    for(j=0;j<N;j++){
      Zaux[i][j]=0.0;
    }
  }
	
/*********************************/
 
/**** STARTING TARGET  ***********/

  // INITIALIZATIONS	
  double ***DM; /* distance matrices */	
  DM = darray3(p,N,N);
  double ***sigmasM; /* variance matrix */
  sigmasM = darray3(p,N,N);
  double **sigmaM, **sigmaMaux;
  sigmaM = darray2(N,N);
  sigmaMaux = darray2(N,N);

  double ***DF; /* distance matrices */	
  DF = darray3(pcont,NF,NF);
  double ***sigmasB; /* variance matrix */
  sigmasB = darray3(pcont,NF,NF);
  double **sigmaB, **sigmaBChol;
  sigmaB = darray2(NF,NF);
  sigmaBChol = darray2(NF,NF);

  double **var, **varM, **varMaux;
  var=darray2(N+2*NF,N+2*NF);
  varM=darray2(N,N);
  varMaux=darray2(N,N);

  double *mu, *muaux, *muM, *muMaux; // mean vectors 
  mu=dvector(0,N); /* global mean */
  muaux=dvector(0,N+2*NF);
  muM=dvector(0,N);
  muMaux=dvector(0,N);

  double *path; /* stores the current draw of y*M and b* */ 
  path=dvector(0,2*NF);
	
  if(meanbias==1){
	  double *muB; // mean vector
	  double **Bt;
	  double *thetaB;
	  double *mean, **cholmean;
	  int qB; /* number of parameters in the linear model X thetaB */


	  char filemuB [100]; /* file where the sequence of muB should go */
	  char strtmp [100]="";
	  strcpy(strtmp,home);
	  strcat(strtmp,filemuB);
	  strcpy(filemuB,strtmp);
	  if (screen!=0) {
		  Rprintf("file where the sequence of muB should go=%s\n",filemuB);
		  Rprintf("number of parameters of the linear model X thetaB=%d\n",qB);
	  }
    thetaB=dvector(0,qB);
    muB=dvector(0,NF);
    for(i=0;i<NF;i++) muB[i]=0.;
    Bt=darray2(qB,NF); /* transpose of the solution to C'B=X; C'C=sigma */
    mean=dvector(0,qB);
    cholmean=darray2(qB,qB);
  }

  double *ustar, *ustaraux; // calibration parameters and bounds	
  int ok;
	
  if(pstar!=0){ /* draw calibration inputs and put them in Z */
	  if (screen !=0) {
		  Rprintf("About to draw calibration inputs and put them in Z\n");
		  dprintmat(ubounds, pstar, 5);
	  }
    ustar=dvector(0,pstar);
    ustaraux=dvector(0,pstar);

    for(i=0;i<pstar;i++) ustaraux[i]=(ubounds[i][1]+ubounds[i][2])*0.5;
    ok=0;
    //Rprintf("Before getting ustar\n");
    getustar(ustar,ustaraux,ubounds,prob,&ok,pstar);
    //Rprintf("Back: The value of ok is %u\n",ok);
    while(ok==0) getustar(ustar,ustaraux,ubounds,prob,&ok,pstar);
    //Rprintf("Back: The value of ok is %u\n",ok);
    //ustar[0]=0.169781; ustar[1]=0.446781; ustar[2]=0.512020;

    for(i=p-pstar;i<p;i++){
      for(j=NM;j<N;j++){
		  Z[i][j]=ustar[i-(p-pstar)];
      }
    }
    dappendvecnotverb(fileU,ustar,pstar);
  }

  // COMPUTE SIGMAB FOR THE STARTING VALUES OF THETAF
  getD(ZF,DF,NF,thetaF,pcont);
  getsigmas(thetaF,DF,sigmasB,NF,pcont);
  getsigma(sigmaB,thetaF,sigmasB,NF,pcont);

  /* do Cholesky */
  dcopymatrix(sigmaB,sigmaBChol,NF, NF);
  //dprintmat(sigmaB,NF,NF);exit(1);  
  dpotrf_(uplo,&NF,&sigmaBChol[0][0],&NF,&info);
  check(info,1);

  // COMPUTE SIGMAM
  getD(Z,DM,N,thetaM,p);
  getsigmas(thetaM,DM,sigmasM,N,p);
  getsigma(sigmaM,thetaM,sigmasM,N,p);
  dcopymatrix(sigmaM,sigmaMaux,N,N); // NEW

  // COMPUTE GLOBAL MEAN: mean of (yM,y*M); this never changes 
  dgemv_(trans,&q,&N,&onedouble,&X[0][0],&q,&thetaL[0],
	 &one,&zero,&mu[0],&one); /* mu=X theta^L */

  // COMPUTE VARM -- preliminary calculations for drawing ustar
  dcopymatrix(sigmaM,varM,N,N);

  /* do Cholesky of upper-left corner */
  dpotrf_(uplo,&NM,&varM[0][0],&N,&info);
  check(info,2);

  dcopymatrix(varM,varMaux,N,N); 
  // the upper-left corner of varM and varMaux will never change throughout
  
  dcopy_(&N,&mu[0],&one,&muM[0],&one);
  
  getkrigpar(varM,muM,yM,NM,NF); /* the lower right corner of varM now
				    has the corr of yM* DyM*| yM DyM the last
				    NF elements of muM have its mean */

/********** MCMC *****************/

  double rateU = 0.0; /* acceptance rate */
  double accept = 0.0; /* acceptance ratio */
  double u = 0.0; /* uniform draw */

  fF=fopen(fileF,"w"); // this is just to clear the file
  fprintf(fF,"%s","");

  if(pstar!=0){
    fcal=fopen(fileU,"w"); // this is just to clear the file
    fprintf(fcal,"%s","");
    fclose(fcal);
  }
   
  fcal=fopen(filepath,"w"); // this is just to clear the file
  fprintf(fcal,"%s","");
  fclose(fcal);

  //###########
  // MCMC LOOP
  //###########
  Rprintf("\n--- MCMC ---\n");
  for(state=0;state<sim;state++){
	R_CheckUserInterrupt();

	if (screen==1) Rprintf("the iteration is %i\n",state);

	/* *****************/
    /* draw b* and y*M */
    /* *****************/
    // assemble covariance matrix of yM,yF,yM*,b* -> var
    getvar(sigmaM,sigmaB,var,thetaM,thetaF,NF,NM,Nrep,p,pstar);

    // assemble mean of the same vector -> muaux
    dcopy_(&N,&mu[0],&one,&muaux[0],&one);
    for(i=N;i<N+NF;i++) muaux[i]=muaux[i-NF];
    for(i=N+NF;i<N+2*NF;i++) muaux[i]=0.0;
    // use assembled mean and variance to draw path:
    // path = (y*M',b*')'
    
    getpath(var,muaux,path,y,N,2*NF);
    dappendvecnotverb(filepath,path,2*NF);

    // if b==0 then uncomment
    // for(i=NF;i<2*NF;i++) path[i]=0.0;
    //dappendvec("tmp.txt",path,2*NF);

    /* **************/
    /* draw lambdaF */
    /* **************/
    // comment this command if wanting to fix lambdaF at value in file.
    
    getnewlambdaF(thetaF,path,yF,s2F,priorshapes,prioriscales,NF,
		  Nrep,NFtot,p,pstar);
    
    /* ********************************/
    /* Metropolis for (ustar,lambdaB) */
    /* ********************************/

    if(pstar!=0){
      for(r=0;r<metrop;r++){
	// draw ustar; ok=1 if all u's are within bounds
	getustar(ustaraux,ustar,ubounds,prob,&ok,pstar);
	// dappendvec("tmp.tmp",ustaraux,pstar);
	//ok=2;
	switch(ok){
	case 0:
	  accept = 0.0;
	  break;
	case 1:
	  dcopymatrix(Z,Zaux,p,N);
	  // update Zaux
	  for(i=p-pstar;i<p;i++){
	    for(j=NM;j<N;j++){
	      Zaux[i][j]=ustaraux[i-(p-pstar)];
	    }
	  }
	  
	  // update variance; the upper-left corner is actually the 
	  // Cholesky already
	  getD_partial(Zaux,DM,N,thetaM,p,NM);
	  getsigmas_partial(thetaM,DM,sigmasM,N,p,NM);
	  getsigma_partial(varMaux,thetaM,sigmasM,N,p,NM);
	  
	  for(i=NM;i<N;i++){
	    for(j=0;j<=i;j++){
	      sigmaMaux[i][j]=varMaux[i][j];
	    }
	  }
	  
	  // compute krigging parameters
	  // lower corner of varMaux and last NF components of muMaux
	  // will contain the parameters of yM*|yM
	  dcopy_(&N,&mu[0],&one,&muMaux[0],&one);  
	  getkrigpar(varMaux,muMaux,yM,NM,NF);
	  
	  // compute log(likelihood)
	  ll = llik(method,yF,path,varM,muM,thetaF[2*(pcont)+1],
		    thetaM[0],Nrep,NM,NF);
	  llaux = llik(method,yF,path,varMaux,muMaux,thetaF[2*(pcont)+1],
		       thetaM[0],Nrep,NM,NF);
	  
	  // compute log(proposal/prior)
	  lp = lprior(ustar,ustaraux,ubounds,prob,pstar);
	  lpaux = lprior(ustaraux,ustar,ubounds,prob,pstar);
	  
	  // acceptance ratio
	  ltarget = llaux - lpaux - ll  + lp;
	  accept = exp(ltarget);
	  break;
	case 2:
	  ustar[0]=1.690712;
	  accept=0.0;
	  break;
	}
	// draw uniform
	u = runif(0.0,1.0); //drand48();
	
	if(u<=accept && ok==1){ /* if accept */
	  if(r==(metrop-1)) rateU = rateU + 1.; // new way of computing rate
	  for(i=p-pstar;i<p;i++){
	    for(j=NM;j<N;j++){
	      Z[i][j]=ustaraux[i-(p-pstar)];
	    }
	  }
	  dcopy_(&pstar,&ustaraux[0],&one,&ustar[0],&one);
	  
	  for(i=NM;i<N;i++){
	    for(j=0;j<=i;j++){
	      varM[i][j]=varMaux[i][j];
	      sigmaM[i][j]=sigmaMaux[i][j];
	    }
	  }
	  dcopy_(&NF,&muMaux[NM],&one,&muM[NM],&one); 
	  /* only the last NF components are of interest */    
	  
	  //getnewlambdaB(path,thetaF,sigmaBChol,NF,priorshapes,prioriscales);
	}
      }
      // store ustar
      dappendvecnotverb(fileU,ustar,pstar);
    }
    // 
    //else{
    //}

    // comment if b==0
    // draw lambdaB
    getnewlambdaB(path,thetaF,sigmaBChol,NF,priorshapes,prioriscales);
    
    //store lambdaB and lambdaF
    fprintf(fF,"%g  %g\n",thetaF[0],thetaF[2*(pcont)+1]); 
    //fprintf(faux,"%lf\n",exp(llaux-ll));
  }
  fclose(fF);
  
  if(pstar!=0){
    fcal=fopen(frateU,"w");
    fprintf(fcal,"%lf",rateU/sim);
    fclose(fcal);
  }

  // to set back the RNG state from C to R
  PutRNGstate();
  Rprintf("\n--- Finished ---\n");
}




/*********************************/
/*           FUNCTIONS           */
/*********************************/

void getvar(double **RM, double **RB, double **result, double *parM,
	    double *parF, int NF, int NM, int *Nrep, int p, int pstar)
{
  int i,j,pcont;
  int N;
  double **aux;

  pcont = p-pstar;
  if(pcont==0) pcont=1;

  N = NM+NF;

  /* var=0.0 */
  for(i=0;i<N+2*NF;i++){
    for(j=0;j<=i;j++){
      result[i][j]=0.0;
    }
  }

  /* var(yM,ybarF), but not yet complet */
  for(i=0;i<N;i++){
    result[i][i]=1./parM[0];
    for(j=0;j<i;j++){
      result[i][j]=RM[i][j]/parM[0];
    }
  }
  
  /* cov(y*M,yM) */
  for(i=N;i<N+NF;i++){
    for(j=0;j<NM;j++){
      result[i][j]=result[i-NF][j];
    }
  }

  /* there are more intelligent ways of doing this, but more
     likely to have bugs */
  aux=darray2(NF,NF);
  for(i=0;i<NF;i++){
    for(j=0;j<=i;j++){
      aux[i][j]=result[i+NM][j+NM];
      aux[j][i]=aux[i][j];
    }
  }
  /* cov(y*M,ybarF) */
  for(i=N;i<N+NF;i++){
    for(j=NM;j<N;j++){
      result[i][j]=aux[i-N][j-NM];
    }
  }
  /* var(y*M) */
  for(i=N;i<N+NF;i++){
    for(j=N;j<=i;j++){
      result[i][j]=result[i-NF][j-NF];
    }
  }

  /* continue with var(yM,ybarF): var(ybarF) */
  for(i=NM;i<N;i++){
    result[i][i]=result[i][i]+
      1./(parF[2*(pcont)+1]*Nrep[i-NM])+1./parF[0];
    for(j=NM;j<i;j++){
      result[i][j]=result[i][j]+RB[i-NM][j-NM]/parF[0];
    }
  }// at this point, var(yM,ybarF) is complete */

  /* var(b*) */
  for(i=N+NF;i<N+2*NF;i++){
    result[i][i]=1./parF[0];
    for(j=N+NF;j<i;j++){
      result[i][j]=RB[i-(N+NF)][j-(N+NF)]/parF[0];
    }
  }
  
  /* same comment as above; this seems easier to read and debug */
  for(i=0;i<NF;i++){
    for(j=0;j<=i;j++){
      aux[i][j]=RB[i][j]/parF[0];
      aux[j][i]=aux[i][j];
    }
  }
  /* cov(b*,ybarF) */ 
  for(i=N+NF;i<N+2*NF;i++){
    for(j=NM;j<N;j++){
      result[i][j]=aux[i-(N+NF)][j-NM];
    }
  }
}

void getustar(double *cal, double *calold, double **bounds, double probab, 
	      int *ok, int dim)
{
  int i;
  double sigma2, tmp;
  double mean;
  double offset;
  double p;

  p = runif(0.0,1.0); //drand48();
  
  if(p<1.-probab){ // local perturbation
    for(i=0;i<dim;i++){ // perturbation lives in (-offset,offset)
      offset = bounds[i][2]-(bounds[i][2]+bounds[i][1])*0.5;
      offset = offset * PROP;
      cal[i]= 2.0*runif(0.0,1.0)*offset - offset; // random perturbation
      cal[i]= cal[i] + calold[i];
    }
  }
  else{ /* simulate from prior (not exactly, because we are not
	   simulating from truncated normal) */
    for(i=0;i<dim;i++)
      if(bounds[i][0]==0.0){
	cal[i]=runif(0,1)*(bounds[i][2]-bounds[i][1]) + bounds[i][1];
      }
      else{
	mean = bounds[i][3];
	sigma2 = bounds[i][4];
	tmp = rnorm(mean,sqrt(sigma2));
	cal[i]=tmp;
      }
  }
  //dprintvec(cal,dim);
  *ok = 1;
  
  for(i=0;i<dim;i++){
      if(cal[i]<bounds[i][1] || cal[i]>bounds[i][2]) *ok = 0;
  }

  /* OLD STUFF
  if(p<1.-probab){ // simulate from uniform
    for(i=0;i<dim;i++){
	cal[i]=runif(0,1)*(bounds[i][2]-bounds[i][1]) 
	  + bounds[i][1];
    }
  }
  */
}

double lprior(double *cal, double *calgiv, double **bounds, 
	      double probab, int dim) 
{
  int i;
  double tmp1, tmp2, tmp;
  double sigma2, mean;
  double offset;

// this function gives the log(proposal/prior) contribution
// to the acceptance ratio

  tmp1=0.0;
  tmp2=0.0;

  // determine whether u is in the locally perturbed interval
  // around calgiv
  for(i=0;i<dim;i++){
    offset = (bounds[i][2]-bounds[i][1])*0.5;
    offset = offset * PROP;
    if(cal[i]<calgiv[i]-offset || cal[i]>calgiv[i]+offset){
      tmp = log(probab);
      return(tmp);
      break;
    }
  }

  // u is in the locally perturbed interval around calgiv
  for(i=0;i<dim;i++){
    if(bounds[i][0]==1.0){
      sigma2 = bounds[i][4];
      mean= bounds[i][3];
      tmp1 = tmp1 - 0.5*log(2.* PI) - 0.5*log(sigma2) - 
	pow(cal[i]-mean,2.0)/(2.*sigma2);
    }
    if(bounds[i][0]==0.0)
      tmp1 = tmp1 - log(bounds[i][2]-bounds[i][1]);
  }
  
  for(i=0;i<dim;i++){
    offset = (bounds[i][2]-bounds[i][1])*0.5;
    offset = offset * PROP;
    tmp2 = tmp2 - log(2.0)-log(offset);
  }
  
  tmp = probab+(1.-probab)*exp(tmp2-tmp1);
  
  tmp = log(tmp);

  return(tmp);
}

void getpath(double **var, double *mean, double *result, double *x, int Nold, 
	     int Nnew)
{ // these computations have been checked against R
  int N,j;
  double *bx;

  /* this is for the lapack calls */
  char uplo[]="U"; /* it means we are feeding the routine with the 
		      upper triangle */
  char trans[]="T";/* it means we want to solve the system with the 
		      transpose of the matrix we are feeding it */
  char diag[]="N"; /* means our matrix is not unit-triangular */
  int info; /* returns sucess status */
  int one=1;
  double onedouble=1.;
  double minusone=-1.;
//  double zero=0.;

  N=Nold+Nnew;
  bx=dvector(0,Nold);

  /* do Cholesky of upper-left corner */
  dpotrf_(uplo,&Nold,&var[0][0],&N,&info);
  check(info,3);

  /* get the solution to C'bx=x, where C is cholesky of upper-left corner */
  dcopy_(&Nold,&x[0],&one,&bx[0],&one);
  dtrtrs_(uplo,trans,diag,&Nold,&one,&var[0][0],&N,
	  &bx[0],&Nold,&info);
  check(info,4);

  /* get the solution to C'mean=mean */
  dtrtrs_(uplo,trans,diag,&Nold,&one,&var[0][0],&N,
	  &mean[0],&Nold,&info);
  check(info,5); /* first Nold components of mean have the solution 
		  bmean */

  daxpy_(&Nold,&minusone,&bx[0],&one,&mean[0],&one); 
  /* first Nold components of mean have bmean-bx */


  /* get the solution to C'B=var.old.new and put it in var.old.new */
  dtrtrs_(uplo,trans,diag,&Nold,&Nnew,&var[0][0],&N,
	  &var[Nold][0],&N,&info);
  check(info,6);

  /* mean(last) <- -B'mean(first) + mean(last) */
  dgemv_(trans,&Nold,&Nnew,&minusone,&var[Nold][0],&N,&mean[0], 
	 &one,&onedouble,&mean[Nold],&one);
      
  /* at this point, the last Nnew components of the vector mean have 
     the krigging mean */

  /* var <- var - B'B */
  dsyrk_(uplo,trans,&Nnew,&Nold,&minusone,
	 &var[Nold][0],&N,&onedouble,&var[Nold][Nold],&N);
  /* at this point, the lower corner of the matrix var has the 
     covariance */

  //dwritemat("tmp.tmp",var,N,N);
  /* do Cholesky */ 
  dpotrf_(uplo,&Nnew,&var[Nold][Nold],&N,&info);
  check(info,7);
  //exit(1);
  /* lower corner of the matrix has the Cholesky of the
     krigging covariance matrix */

  /* result[j]~iid N(0,1) */
  for(j=0;j<Nnew;j++) result[j]=rnorm(0.,1.);
  
  /* result = C result */
  dtrmv_(uplo,trans,diag,&Nnew,&var[Nold][Nold],&N,&result[0],&one);
  
  /* result = result + mean */
  daxpy_(&Nnew,&onedouble,&mean[Nold],&one,&result[0],&one);  
  
  free_dvector(bx,0,Nold);
}

void getnewlambdaF(double *parF, double *path, double *y, double s,
		   double *a, double *b, int N, int *Nreps, int Ntot, 
		   int p, int pstar)
{
  int i, pcont;
  double *aux;
  double rate, shape;
  int one=1;
  double minusone=-1.0;

  aux = dvector(0,N);

  pcont = p-pstar;
  if(pcont==0) pcont=1;

  //aux = y - path[0:N-1] - path[N:2*N-1]
  dcopy_(&N,&y[0],&one,&aux[0],&one); //aux <- y
  daxpy_(&N,&minusone,&path[0],&one,&aux[0],&one);//aux <- aux - path[0:N-1]
  daxpy_(&N,&minusone,&path[N],&one,&aux[0],&one);//aux <- aux - path[N:2*N-1]

  for(i=0;i<N;i++) aux[i]=aux[i]*sqrt((double)Nreps[i]);

  rate = pow(dnrm2_(&N,&aux[0],&one),2.);
  rate = rate + s;
  rate = 0.5*rate;
  rate = rate+b[pcont+1];

  shape = 0.5*(double)Ntot+a[pcont+1];

  parF[2*(pcont)+1]=rgamma(shape,1)/rate;
  
  free_dvector(aux,0,N);
}

void getnewlambdaB(double *path, double *parF, double **chol,
		   int NF, double *a, double *b)
{
  double *aux;
//  int N;
  double rate, shape;

  /* this is for the lapack calls */
  char uplo[]="U"; /* it means we are feeding the routine with the 
		      upper triangle */
  char trans[]="T";/* it means we want to solve the system with the 
		      transpose of the matrix we are feeding it */
  char diag[]="N"; /* means our matrix is not unit-triangular */
  int info; /* returns sucess status */
  int one=1;
//  double onedouble=1.;
//  double minusone=-1.;
//  double zero=0.;

  aux=dvector(0,NF);

  dcopy_(&NF,&path[NF],&one,&aux[0],&one); //aux <- b
  
  /* get the solution to C'x=b */
  dtrtrs_(uplo,trans,diag,&NF,&one,&chol[0][0],&NF,
	  &aux[0],&NF,&info);
  check(info,8);

  rate = pow(dnrm2_(&NF,&aux[0],&one),2.);

  rate = rate * 0.5 + b[0];
  shape = 0.5 * (double)NF + a[0];

  parF[0] = rgamma(shape,1)/rate;
}

void getkrigpar(double **var, double *mean, double *x, int Nold,
		int Nnew)
{
  double *bx;
  int N;

  /* this is for the lapack calls */
  char uplo[]="U"; /* it means we are feeding the routine with the 
		      upper triangle */
  char trans[]="T";/* it means we want to solve the system with the 
		      transpose of the matrix we are feeding it */
  char diag[]="N"; /* means our matrix is not unit-triangular */
  int info; /* returns sucess status */
  int one=1;
  double onedouble=1.;
  double minusone=-1.;
//  double zero=0.;

  N=Nold+Nnew;
  bx=dvector(0,Nold);

  /* get the solution to C'bx=x */
  dcopy_(&Nold,&x[0],&one,&bx[0],&one);
  dtrtrs_(uplo,trans,diag,&Nold,&one,&var[0][0],&N,
	  &bx[0],&Nold,&info);  
  check(info,9);
 
  /* get the solution to C'mu=mu */
  dtrtrs_(uplo,trans,diag,&Nold,&one,&var[0][0],&N,
	  &mean[0],&Nold,&info);
  check(info,10); /* first Nold components of mean have the 
		    solution, bmean */
      
  daxpy_(&Nold,&minusone,&bx[0],&one,&mean[0],&one); 
  /* first Nold components of mu have bmean-bx */

  /* get the solution to C'B=var.old.new */
  dtrtrs_(uplo,trans,diag,&Nold,&Nnew,&var[0][0],&N,
	  &var[Nold][0],&N,&info);
  check(info,11);

  /* mean(last) <- -B'mean(first) + mean(last) */
  dgemv_(trans,&Nold,&Nnew,&minusone,&var[Nold][0],&N,&mean[0], 
	 &one,&onedouble,&mean[Nold],&one);
    
  /* at this point, the last Nnew components of the vector mean have 
     the mean (of the unobserved code values given the observed ones) */
  
  /* varM <- varM - B'B */
  dsyrk_(uplo,trans,&Nnew,&Nold,&minusone,
	 &var[Nold][0],&N,&onedouble,&var[Nold][Nold],&N);
  /* at this point, the lower-right corner of the matrix var has the 
     covariance (of the unobserved code values given the observed ones),
     except for dividing by lambdaM */
  free_dvector(bx,0,Nold);
}

double llik(int method, double *yF, double *path, double **var, double *mu, double lambdaF, 
	    double lambdaM, int *rep, int NM, int NF)
{
  int i,j;
  double **aux;
  double *tmp;
  double quad, det;
  
  /* this is for the lapack calls */
  char uplo[]="U"; /* it means we are feeding the routine with the 
		      upper triangle */
  char trans[]="T";/* it means we want to solve the system with the 
		      transpose of the matrix we are feeding it */
  char diag[]="N"; /* means our matrix is not unit-triangular */
  int info; /* returns sucess status */
  int one=1;
//  double onedouble=1.;
  double minusone=-1.;
//  double zero=0.;

  aux = darray2(NF,NF);
  tmp = dvector(0,NF);

  // method = 1, direct Metropolis within Gibbs; method = 2, integrate out y^M_*

  for(i=0;i<NF;i++){
    for(j=0;j<=i;j++){
      aux[i][j]=var[i+NM][j+NM]/lambdaM;
    }
  }

  if(method==2){
    for(i=0;i<NF;i++){
      aux[i][i]=aux[i][i] + 1./((double) rep[i]*lambdaF);
    }
  }

  // do Cholesky
  dpotrf_(uplo,&NF,&aux[0][0],&NF,&info);
  check(info,12);

  det = 0.0;
  for(i=0;i<NF;i++) det = det - log(fabs(aux[i][i]));
  
  switch(method){
  case 2:
    // tmp <- yF-mu-b, where mu = E[yM* | yM]
    dcopy_(&NF,&yF[0],&one,&tmp[0],&one);
    daxpy_(&NF,&minusone,&mu[NM],&one,&tmp[0],&one);
    for(i=0;i<NF;i++) tmp[i] = tmp[i]-path[i+NF];
    break;
  
  case 1:
    //tmp <- yM*-mu where mu = E[yM* | yM]
    dcopy_(&NF,&path[0],&one,&tmp[0],&one);
    daxpy_(&NF,&minusone,&mu[NM],&one,&tmp[0],&one);
    break;
  
  default:
    Rprintf("Error: method must be 1 or 2");
    //exit(1);
  }

  /* get the solution to C'x=tmp */ 
  dtrtrs_(uplo,trans,diag,&NF,&one,&aux[0][0],&NF,&tmp[0],&NF,&info);
  check(info,13);

  quad = -0.5*pow(dnrm2_(&NF,&tmp[0],&one),2.);
  quad = quad + det;

  free_dvector(tmp,0,NF);
  free_darray2(aux,NF,NF);

  return(quad);
}
