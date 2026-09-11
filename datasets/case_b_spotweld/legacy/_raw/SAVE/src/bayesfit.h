/*
 *  bayesfit.h
 *  R_Package
 *
 *  Created by Jesus Palomo on 23/5/12.
 *  Copyright 2012 Universidad Rey Juan Carlos. All rights reserved.
 *
 */

void getvar(double **RM, double **RB, double **result, double *parM,
			double *parF, int NF, int NM, int *Nrep, int p, int pstar);
void getustar(double *cal, double *calold, double **bounds, double probab, 
			  int *ok, int dim);
void getpath(double **var, double *mean, double *result, double *x, int Nold, 
			 int Nnew);
void getnewlambdaB(double *path, double *parF, double **chol,
				   int NF, double *a, double *b);
void getnewlambdaF(double *parF, double *path, double *y, double s,
				   double *a, double *b, int N, int *Nreps, int Ntot, 
				   int p, int pstar);
void getkrigpar(double **var, double *mean, double *x, int Nold,
				int Nnew);
double llik(int method, double *yF, double *path, double **var, double *mu, double lambdaF, 
			double lambdaM, int *rep, int NM, int NF);
double lprior(double *cal, double *calgiv, double **bounds, double probab, 
			  int dim);

//int bayesfit();
void bayesfit(int *FlagOutput,int *P, int *PSTAR, int *Q, int *nm, int *nf,
			  double *PROB, double *MULTMLE, int *FlagMethod, int *SIM, 
			  int *METROP, char *homePath[]);

