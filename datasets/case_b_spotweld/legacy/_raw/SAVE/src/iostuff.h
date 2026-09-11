/*
 *  iostuff.h
 *  R_Package
 *
 *  Created by Jesus Palomo on 13/6/12.
 *  Copyright 2012 Universidad Rey Juan Carlos. All rights reserved.
 *
 */

void dprintmat(double **a, int n, int m);
int dreadmat(char file[], double **a, int n, int m);
int dwritemat(char *file,double **a, int n, int m);
int iappendmat(char *file, int **a, int n, int m);
int appendvec(char file[], float *a, int n);
int iappendvec(char *file, int *a, int n);
int dappendvec(char *file, double *a, int n);
int dappendvecnotverb(char *file, double *a, int n);
int appendmat(char *file, float **a, int n, int m);
int dappendmat(char *file, double **a, int n, int m);
void iprintmat(int **a, int n, int m);
int ireadmat(char file[], int **a, int n, int m);
int iwritemat(char *file, int **a, int n, int m);
void printmat(float **a, int n, int m);
int readmat(char file[], float **a, int n, int m);
void uliprintmat(unsigned long int **a, int n, int m);
int uliwritemat(char *file, unsigned long int **a, int n, int m);
int writemat(char *file, float **a, int n, int m);
void dprintvec(double *v, int n);
int dreadvec(char file[], double *v, int n);
void iprintvec(int *v, int n);
int ireadvec(char file[], int *v, int n);
int iwritevec(char file[], int *v, int n);
void printvec(float *v, int n);
int readvec(char file[], float *v, int n);
int uliwritevec(char file[], unsigned long int *v,int n);
int uliappendvec(char file[], unsigned long int *v, int n);
int writevec(char file[], float *v, int n);
int dwritevec(char file[], double *v, int n);
void iprinta3(int ***a, int n1, int n2, int n3);
void printa3(float ***a, int n1, int n2, int n3);
int ireada3(char file[], int ***a, int n1, int n2, int n3);
int reada3(char file[], float ***a, int n1, int n2, int n3);
int writea3(char file[], float ***a, int n1, int n2, int n3);
int dwritea3(char file[], double ***a, int n1, int n2, int n3);
int appenda3(char file[], float ***a, int n1, int n2, int n3);
/* Functions that read the input data */
// returns a vector with model data
double * get_datacode (char home[], int nm);
// returns a vector with the number of replicates at each unique design point 
int * get_inputsrep (char home[], int nf);
double * get_mlethetaF (char home[], int pcont);
double * get_mlethetaM (char home[], int p);
double * get_mlethetaL (char home[], int q);
// returns a vector with the field data
double * get_datafield (char home[], int *nrep, int nf);
// returns a vector with the field data and the data -- sum of squared deviations 
//from individual means */
double * get_datafield_s2F (char home[], int *nrep, int nf, double *s2F);
// returns a matrix with the code data inputs
double ** get_inputscode (char home[], int nm, int p);
double ** get_inputsfield (char home[], int nf, int pcont);
double ** get_designMatrix (char home[], int nf, int nm, int q);
// read the matrix with bounds on u
double ** get_ubounds (char home[], int pstar);
