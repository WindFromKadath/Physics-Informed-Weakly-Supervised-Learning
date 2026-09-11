/*************************************************
 *   Copyright Nov 12 2003                       *
 *  This file was created and copyrighted by the *
 *  National Institute of Statistical Sciences   *
 *************************************************/

#include "common.h"
#include "setup.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <R.h>
#include <Rdefines.h>
#include <R_ext/Lapack.h>

void getD(double **points, double ***result, int dim, double *par, int p)
{
  int i,j,k;
  /* lower triangular part of the distance matrices */
  for(k=0;k<p;k++){
    for(i=0;i<dim;i++){
      result[k][i][i]=0.;
      for(j=0;j<i;j++){
	result[k][i][j]=pow(fabs(points[k][i]-points[k][j]),par[k+p+1]);
      }
    }
  }
}

void getsigmas(double *par, double ***dist, double ***result, int dim, int p)
{
  int i,j,k;
  /* lower triangular part of the correlation matrices */
  for(k=0;k<p;k++){
    for(i=0;i<dim;i++){
      result[k][i][i]=1.;
      for(j=0;j<i;j++){
	result[k][i][j]=exp(- dist[k][i][j] * par[k+1]);
      }
    }
  } 
}

void getsigma(double **result, double *par, double ***corrs, int dim, int p)
{
  int i,j,k;
  /* lower triangular part of the correlation matrix */
  for(i=0;i<dim;i++){
    result[i][i]=1.;
    for(j=0;j<i;j++){
      result[i][j]=1.;
      for(k=0;k<p;k++){
	result[i][j]=result[i][j]*corrs[k][i][j];
      }
    }
  }
}

void getD_partial(double **points, double ***result, int dim, double *par, int p, 
		  int dimM)
{
  int i,j,k;
  /* lower triangular part of the distance matrices 
     but starting at line DM, all rest is unchanged */
  for(k=0;k<p;k++){
    for(i=dimM;i<dim;i++){
      result[k][i][i]=0.;
      for(j=0;j<i;j++){
	result[k][i][j]=pow(fabs(points[k][i]-points[k][j]),par[k+p+1]);
      }
    }
  }
}

void getsigmas_partial(double *par, double ***dist, double ***result, int dim, int p, 
		       int dimM)
{
  int i,j,k;
  /* lower triangular part of the correlation matrices 
     but starting at line DM, all rest is unchanged */
  for(k=0;k<p;k++){
    for(i=dimM;i<dim;i++){
      result[k][i][i]=1.;
      for(j=0;j<i;j++){
	result[k][i][j]=exp(- dist[k][i][j] * par[k+1]);
      }
    }
  }
}

void getsigma_partial(double **result, double *par, double ***corrs, int dim, int p,
		      int dimM)
{
  int i,j,k;
  /* lower triangular part of the correlation matrix 
     but starting at line DM, all rest is unchanged */

  for(i=dimM;i<dim;i++){
    result[i][i]=1.;
    for(j=0;j<i;j++){
      result[i][j]=1.;
      for(k=0;k<p;k++){
	result[i][j]=result[i][j]*corrs[k][i][j];
      }
    }
  }
}

void algebra(double **tsolm, double *solv, double *m, double **choldcp, 
	       int dim, int q)
{
/* tsolm is the transpose of the solution to C'B=X, C'C=sigma 
   solv is the solution to C'b=y

   returns:
   -> m -- the mean
   -> choldcp -- the cholesky decomp of the correlation matrix
          of the full conditional of theta^L
*/
  int i,j,info;
  double **aux;
  double onedouble=1.;
  double zero=0.;
  int one=1;
  char uplo[]="U"; /* it means we are feeding the routine with the 
		      upper triangle */
  char trans[]="T";/* it means we want to solve the system with the 
		      transpose of the matrix we are feeding it */
  char diag[]="N"; /* means our matrix is not unit-triangular */

  aux=darray2(q,q);

  for(i=0;i<q;i++){
    for(j=0;j<=i;j++){
      aux[i][j]=0.;
    }
  }

  dsyrk_(uplo,trans,&q,&dim,&onedouble,&tsolm[0][0],&dim,&zero,&aux[0][0],&q);
  /* aux contains in its lower triangle B'B */

/* now I need to compute its Cholesky, its inverse, and the inverse of 
   its Cholesky */

  dpotrf_(uplo,&q,&aux[0][0],&q,&info); /* do cholesky; now aux has the chol
					    of B'B */
  check(info,91);

  dcopymatrix(aux,choldcp,q,q); 
  dtrti2_(uplo,diag,&q,&choldcp[0][0],&q,&info); /* cholesky of the inverse 
						    ie the inverse of the 
						    cholesky*/
  check(info,92);

  dpotri_(uplo,&q,&aux[0][0],&q,&info); /* compute inverse */
  /* at this point, aux = (B'B)^-1 */
  check(info,93);
  /* compute B'b */
  dgemv_(trans,&dim,&q,&onedouble,&tsolm[0][0],&dim,&solv[0],
	 &one,&zero,&m[0],&one); /* m=B'b */

  /* m=(B'B)^-1 B'b */
  dtrmv_(uplo,trans,diag,&q,&aux[0][0],&q,&m[0],&one);

  free_darray2(aux,q,q);
}

void dcopymatrix(double **B, double **A, int n, int m)
{ /* A <- B ; matrices are n x m*/

  int i, one=1;

  for(i=0; i<n; i++){
    dcopy_(&m,B[i],&one,A[i],&one);
  }
}

double Tr(double **mat, int r)
{
  int i;
  double aux;

  aux=0.;

  for(i=0;i<r;i++){
    aux=aux+mat[i][i];
  }
  return(aux);
}

void check(int info, int line){
  if(info!=0){
    Rprintf("ERROR, info=%d, line=%d\n",info,line);
    //exit(1);
  }
}
