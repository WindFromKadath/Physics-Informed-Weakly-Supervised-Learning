/*
 *  bayesfitSetup.h
 *  R_Package
 *
 *  Created by Jesus Palomo on 9/10/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */


void bayesfitSetupCalib(int screen, char home[], double multmle, int p, int q, int pstar, int pcont, int NM, int NF, double **thetaF, 
				   double **thetaM, double **thetaL, double **priorshapes, double **prioriscales, 
				   double ***ubounds, double **yF, double **yM, double **y, int *NFtot, int **Nrep, double ***ZF, 
				   double ***ZM, double ***Z, double *s2F, double ***X, double ***Xt, char **fileF, char **filepath, 
				   char **fileU, char **frateU);
void bayesfitSetup(int screen, char home[], double multmle, int p, int q, int pstar, int pcont, int NM, int NF, double **thetaF, 
				   double **thetaM, double **thetaL, double **priorshapes, double **prioriscales, 
				   double **yF, double **yM, double **y, int *NFtot, int **Nrep, double ***ZF, 
				   double ***ZM, double ***Z, double *s2F, double ***X, double ***Xt, char **fileF, char **filepath);
