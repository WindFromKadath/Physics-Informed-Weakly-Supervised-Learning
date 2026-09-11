/*
 *  predict_reality.h
 *  R_Package
 *
 *  Created by Jesus Palomo on 23/5/12.
 *  Copyright 2012 Universidad Rey Juan Carlos. All rights reserved.
 *
 */

void getpriordraws(double *ustar, int pstar, int *indlearn, double **bounds);
void getvar_pr(double **RM, double **RB, double **result, double *parM,
			double *parF, int *reps, int NMold, int NMnew, int NFold, 
			int NFnew, int p, int pstar);

void predict_reality(int *FlagOutput,int *P, int *PSTAR, int *Q, int *nm, int *nf,
					 int *nmnew, int *nfnew, double *TOL, int *SIM, int *nBurning, 
					 int * nThinning, char *homePath[]);
