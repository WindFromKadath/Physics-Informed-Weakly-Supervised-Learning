/*
 *  optimize.h
 *  R_Package
 *
 *  Created by Jesus Palomo on 23/5/12.
 *  Copyright 2012 Universidad Rey Juan Carlos. All rights reserved.
 *
 */

void optimize(int *FlagOutput, int *maxiter, double *Eps, double *Err, double *Psi0, char *homePath[]);
double score(double psi, int n, double tot2, double sum2);
double hess(double psi, int n, double tot2, double sum2);
