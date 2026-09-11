#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <R.h>
#include <R_ext/Print.h>

#define DEATH {Rprintf("allocation failure in memory allocation\n");\
               error(".\n");}

int nrerror(char error_text[])
{
  REprintf("Numerical Recipes run-time error...\n");
  REprintf("%s\n",error_text);
  REprintf("...now exiting to system...\n");
  error(".\n");
}

float *vector(int nl, int nh)
{
  float *v;
  
  v=(float *)malloc((unsigned) (nh-nl+1)*sizeof(float));
  if (!v) nrerror("allocation failure in vector()");
  return v-nl;
}

int *ivector(int nl, int nh)
{
  int *v;
  
  v=(int *)malloc((unsigned) (nh-nl+1)*sizeof(int));
  if (!v) nrerror("allocation failure in ivector()");
  return v-nl;
}

double *dvector(int nl, int nh)
{
  double *v;
  //Rprintf("allocating some space - %u\n",(nh-nl+1)*sizeof(double));
  v=(double *)malloc((unsigned) (nh-nl+1)*sizeof(double));
  if (!v) nrerror("allocation failure in dvector()");
  //Rprintf("Finished getting memory for a vector - %u\n",nh);
  return v-nl;
}

unsigned long int *ulivector(int nl, int nh)
{
  unsigned long int *v;
  
  v=(unsigned long int *)malloc((unsigned) (nh-nl+1)*sizeof(double));
  if (!v) nrerror("allocation failure in ulivector()");
  return v-nl;
}

void free_vector(float *v, int nl, int nh)
{
  free((char*) (v+nl));
}

void free_ivector(int *v, int nl, int nh)
{
  free((char*) (v+nl));
}

void free_dvector(double *v, int nl, int nh)
{
  free((char*) (v+nl));
}

int ***iarray3( int n1, int n2, int n3)
/*
  allocates space for a 3 index array 0..n1-1, 0...n2-1, 0...n3-1
*/
{
  int ***a, i, j;
  
  a = (int ***) malloc(n1 * sizeof(int **));
  if(a == NULL) {REprintf("allocation failure in memory allocation\n");
        error(".\n");}
		  
  a[0] = (int **) malloc(n1 * n2 * sizeof(int *));
  if(a[0] == NULL) {REprintf("allocation failure in memory allocation\n");
        error(".\n");}
  for(i=1;i<n1;i++) a[i] = a[i-1] + n2;

  a[0][0] = (int *) malloc(n1 * n2 * n3 * sizeof(int));
  if(a[0][0] == NULL) {REprintf("allocation failure in memory allocation\n");
        error(".\n");}
  for(i=0;i<n1;i++) 
    for(j=0;j<n2;j++) 
      a[i][j] = a[0][0] + n2*n3*i + j*n3;
    
  return a;
}

float ***array3(int n1, int n2, int n3)
/*
  allocates space for a 3 index array 0..n1-1, 0...n2-1, 0...n3-1
*/
{
  float ***a;
  int i, j;

  a = (float ***) malloc(n1 * sizeof(float **));
  if(a == NULL) {REprintf("allocation failure in memory allocation\n");
        error(".\n");}

  a[0] = (float **) malloc(n1 * n2 * sizeof(float *));
  if(a[0] == NULL) {REprintf("allocation failure in memory allocation\n");
        error(".\n");}
  for(i=1;i<n1;i++) a[i] = a[i-1] + n2;

  a[0][0] = (float *) malloc(n1 * n2 * n3 * sizeof(float));
  if(a[0][0] == NULL) {REprintf("allocation failure in memory allocation\n");
        error(".\n");}
  for(i=0;i<n1;i++) 
    for(j=0;j<n2;j++) 
      a[i][j] = a[0][0] + n2*n3*i + j*n3;
    
  return a;
}

double ***darray3(int n1, int n2, int n3)
/*
  allocates space for a 3 index array 0..n1-1, 0...n2-1, 0...n3-1
*/
{
  double ***a;
  int  i, j;

  a = (double ***) malloc(n1 * sizeof(double **));
  if(a == NULL) {REprintf("allocation failure in memory allocation\n");
        error(".\n");}

  a[0] = (double **) malloc(n1 * n2 * sizeof(double *));
  if(a[0] == NULL) {REprintf("allocation failure in memory allocation\n");
        error(".\n");}
  for(i=1;i<n1;i++) a[i] = a[i-1] + n2;

  a[0][0] = (double *) malloc(n1 * n2 * n3 * sizeof(double));
  if(a[0][0] == NULL) {REprintf("allocation failure in memory allocation\n");
        error(".\n");}
  for(i=0;i<n1;i++) 
    for(j=0;j<n2;j++) 
      a[i][j] = a[0][0] + n2*n3*i + j*n3;
    
  return a;
}

int **iarray2(int n1, int n2)
/*
  allocates space for a 2 index array (matrix) 0..n1-1, 0...n2-1
*/
{
  int **a, i;

  a = (int **) malloc(n1 * sizeof(int *));
  if(a == NULL) {REprintf("allocation failure in memory allocation\n");
        error(".\n");}

  a[0] = (int *) malloc(n1 * n2 * sizeof(int));
  if(a[0] == NULL) {REprintf("allocation failure in memory allocation\n");
        error(".\n");}
  for(i=1;i<n1;i++) a[i] = a[i-1] + n2;

  return a;
}

float **array2(int n1, int n2)
/*
  allocates space for a 2 index array (matrix) 0..n1-1, 0...n2-1
*/
{
  float **a; 
  int i;

  a = (float **) malloc(n1 * sizeof(float *));
  if(a == NULL) {REprintf("allocation failure in memory allocation\n");
        error(".\n");}

  a[0] = (float *) malloc(n1 * n2 * sizeof(float));
  if(a[0] == NULL) {REprintf("allocation failure in memory allocation\n");
        error(".\n");}
  for(i=1;i<n1;i++) a[i] = a[i-1] + n2;

  return a;
}

double **darray2(int n1, int n2)
/*
  allocates space for a 2 index array (matrix) 0..n1-1, 0...n2-1
*/
{
  double **a; 
  int i;

  a = (double **) malloc((unsigned)n1 * sizeof(double *));
  if(a == NULL) {REprintf("allocation failure in memory allocation\n");
        error(".\n");}

  a[0] = (double *) malloc((unsigned)(n1*n2) * sizeof(double));
  if(a[0] == NULL) {REprintf("allocation failure in memory allocation\n");
        error(".\n");}
  for(i=1;i<n1;i++) a[i] = a[i-1] + n2;

  //Rprintf("Finished getting memory for a matrix %d x %d\n",n1,n2);

  return a;
}
/************ array4 not yet ready ******************
float ****array4(n1,n2,n3,n4)
***********************************************************************
  allocates space for a 4 index array 0..n1-1, 0..n2-1, 0..n3-1 0..n4-1
***********************************************************************
int n1,n2,n3,n4;
{
  float ****a;
  int  i, j, k;

  a = (double ****) malloc(n1 * sizeof(float ***));
  if(a == NULL) DEATH

  a[0] = (double ***) malloc(n1 * n2 * sizeof(float **));
  if(a[0] == NULL) DEATH
  for(i=1;i<n1;i++) a[i] = a[i-1] + n2;

  a[0][0] = (double **) malloc(n1 * n2 * n3 * sizeof(float *));
  if(a[0][0] == NULL) DEATH
  for(i=0;i<n1;i++) 
    for(j=0;j<n2;j++) 
      a[i][j] = a[0][0] + n2*n3*i + j*n3;
    
  a[0][0][0] = (double *) malloc(n1 * n2 * n3 * n4 * sizeof(float));
  if(a[0][0][0] == NULL) DEATH
  for(i=0;i<n1;i++) 
    for(j=0;j<n2;j++) 
      for(k=0;k<n3;k++) 
        a[i][j][k] = a[0][0][0] + n2*n3*n4*i + j*n3*n4; + k*n4;
    
  return a;
}
************ array4 not yet ready ******************/

void free_iarray2(int **a, int n1, int n2)
{
  free((char*) (a[0]));
  free((char*) (a));
}

void free_array2(float **a, int n1, int n2)
{
  free((char*) (a[0]));
  free((char*) (a));
}

void free_darray2(double **a, int n1, int n2)
{
  free((char*) (a[0]));
  free((char*) (a));
}

void free_array3(float ***a, int n1, int n2, int n3)
{
  free((char*) (a[0][0]));
  free((char*) (a[0]));
  free((char*) (a));
}

void free_darray3(double ***a, int n1, int n2, int n3)
{
  free((char*) (a[0][0]));
  free((char*) (a[0]));
  free((char*) (a));
}

void free_iarray3(int ***a, int n1, int n2, int n3)
{
  free((char*) (a[0][0]));
  free((char*) (a[0]));
  free((char*) (a));
}
