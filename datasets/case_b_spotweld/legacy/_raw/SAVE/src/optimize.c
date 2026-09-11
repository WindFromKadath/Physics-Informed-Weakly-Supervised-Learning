/*************************************************
 *  Copyright Nov 12 2003                        *
 *  This file was created and copyrighted by the *
 *  National Institute of Statistical Sciences   *
 *************************************************/

#include <R_ext/Lapack.h>
#include <R_ext/Print.h>
#include "common.h"
#include <stdio.h>
#include <Rmath.h>
#include <stdlib.h>
#include <string.h>
#include <R.h>
#include <Rdefines.h>
#include "iostuff.h"
#include "optimize.h"
#include "setup.h"

void optimize(int *FlagOutput, int *maxiter, double *Eps, double *Err, double *Psi0, char *homePath[])
{
  int screen = *FlagOutput; // Flag to print output if 1, history is printed to the screen */
  double *thetaF;

  /* working directory path */
  char home[100];
  strcpy(home,*homePath);


  FILE *finput, *foutput;
  char strtmp [100]="";

  char fileOut[100] = "thetaF_mle.dat";
  strcpy(strtmp,home);
  strcat(strtmp,fileOut);
  strcpy(fileOut,strtmp);

  char fileIn[100] = "biasaux.tmp";
  strcpy(strtmp,home);
  strcat(strtmp,fileIn);
  strcpy(fileIn,strtmp);

  double *inputs; /* contains the inputs to the problem */
  inputs = dvector(0,3);
  dreadvec(fileIn,inputs,3);

  int N; /* number of observations */
  double so2, to2; /* sum of squares and squared total */
  N = (int) inputs[0];
  so2 = inputs[1];
  to2 = inputs[2];

  int maxiterations = *maxiter; /* maximum number of iterations */
  double eps = *Eps;
  double err = *Err;
  double psi0 = *Psi0;
  
  /*
  printf("%lf\t%lf\n",to2,so2);
  printf("%lf\n",score(psi0,N,to2,so2));
  printf("%lf\n",hess(psi0,N,to2,so2));
  exit(1);
  */
  /*
  double **r;
  r=darray2(3,1000);
  int i;
  for(i=0;i<1000;i++){
    r[0][i]=1.-(1000.-i)/1000.;
    r[1][i]=score(1.-(1000.-i)/1000.,N,to2,so2);
    r[2][i]=hess(1.-(1000.-i)/1000.,N,to2,so2);
  }
  */
  //dwritemat("r2.tmp",r,3,1000);exit(1);

  // To get the RNG state from R when this C fuction is call from R
  GetRNGstate();
	
  void R_CheckUserInterrupt(void);

  int counter;
  double psi;

  counter = 0;
  while(err>eps && counter<maxiterations){
    R_CheckUserInterrupt();
    
    psi = psi0-score(psi0,N,to2,so2)/hess(psi0,N,to2,so2);
    err = fabs((psi-psi0)/psi);
    psi0 = psi;
    //printf("%lf  %lf\n",psi0,err);
    //printf("%lf\n",score(psi0,N,to2,so2));
    //printf("%lf\n\n",hess(psi0,N,to2,so2));
    if(psi0>1. || psi0 <0.) psi0 = runif(0.0,1.0); //drand48();
    counter = counter + 1;
  }

  //
  // THIS HAS TO BE DOUBLE CHECKED -29/06/2013-
  // Changed 21/02/2014 (it was err<=eps)
  if(err > eps) {
	  psi = 0.0; // if no convergence
	  Rprintf("\n WARNING: No convergence has been achieved after %d iterations\n",maxiterations);
	  Rprintf(" WARNING: using the median instead \n");
	
	  double psimedian;
	  psimedian = 3./(N+3.);

	  if(psimedian>psi) psi=psimedian;
  }
  

/* **********************************
 * compute the corresponding lambda
 * **********************************/
  double lambda;
  lambda = N*(1.-psi) / ( so2-psi*to2/(1.+(N-1.)*psi) );
  
/* *****************************************
 * go back to the original parameterization
 * *****************************************/
  
  // NOTICE that changing the order of the output will have consequences on
  // any routine that reads fileOut
  thetaF = dvector(0,4);
  // write to the file using gasp's parameterization
  thetaF[0] = psi/lambda; // 1/lambdaB
  thetaF[1] = 1.; /* these are beta and alpha, which are irrelevant */
  thetaF[2] = 0.; /*			       in this case */
  thetaF[3] = (1.-psi)/lambda; // 1/lambdaF

  dwritevec(fileOut,thetaF,4);
  
}
 
double score(double psi, int n, double tot2, double sum2)
{
  double tmp, num, denom, N;
  N=(double)n;

  tmp = 1.+(N-1.)*psi;

  num = tmp * tmp * sum2 - (1. + (N-1)*pow(psi,2.) ) * tot2;
  num = num * N; // if integrating w/ respect 1/lambda d lambda
  //num = num * (N+2); // if integrating with respect to d lambda

  denom = tmp * sum2 - psi * tot2;
  denom = denom * (1.-psi) * tmp;

  tmp = num/denom + psi * N * (N-1.);
  tmp = -0.5 * tmp;
  return(tmp);
}

double hess(double psi, int n, double tot2, double sum2)
{
  double aux1, aux2, tmp1, tmp2, N, tmp;
  N=(double)n;

  aux1 = 1.+(N-1.) * psi;
  aux2 = sum2 * aux1 - psi * tot2;

  tmp1 = sum2 * pow(aux1,3.);
  tmp1 = tmp1 + ((N-1.) * pow(1.-psi,2.) - (1.+(N-1.)*pow(psi,2.)) * aux1) * tot2;

  tmp2 = sum2 * pow(aux1,2.)- (1.+(N-1.)* pow(psi,2.))*tot2;

  tmp = -2. * tmp1/(pow(aux1,2.) * pow(1.-psi,2.) * aux2);

  tmp = tmp + pow(tmp2/(aux1 * aux2 * (1.-psi)),2.);

  tmp = N * tmp - N*(N-1.);//if integrating with respect to 1/lambda d lambda
  //tmp = (N+2.) * tmp - N*(N-1.);//if integrating with respect to d lambda

  tmp = tmp*0.5;

  return(tmp);
}
