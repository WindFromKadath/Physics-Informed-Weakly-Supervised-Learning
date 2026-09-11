/*************************************************
 *   Copyright Nov 12 2003                       *
 *  This file was created and copyrighted by the *
 *  National Institute of Statistical Sciences   *
 *************************************************/

#include "setup.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "chol_pivot.h"
#include <Rmath.h>
#include <R.h>
#include <R_ext/Lapack.h>


void chol_pivot(double **a, double **L, int *pivot, int *rank, 
		double tol, int n) 
/***************************************************************
Performs cholesky decomposition of the matrix a using a similar
pivoting procedure to the one used by S-Plus when the "pivot=T"
option is used.  The cholesky decomposition is returned in the
L matrix, such that a = L%*%t(L), and may not be lower triangular,
due to the pivoting algorithm.  The rank of the matrix is returned
in the rank variable.  The array pivot tells which pivots were found
in which columns.  For instance pivot[i]=j, means that the jth
column served as the ith pivot.  (Remember that i and j both go
from 0 to n-1.)
***************************************************************/
{
 int i,j,k, *orig_col;
 double sum, max, *vars, tmp;


 orig_col = ivector(0, n-1);
 vars = dvector(0, n-1);

 /* INITIALIZE */

 /* At the beginning, none of the cols have been used, so we denote
    this by setting the value of the orig_col array equal to -1 for all
    elements. Also, we initialize all elements of the Cholesky matrix
    to zero. */
 for (i=0; i<n; i++)
   {
     orig_col[i] = -1;
     pivot[i] = -1;
     vars[i] = sqrt(a[i][i]);
     for (j=0; j<n; j++)
       L[i][j] = 0.0;
   }

 *rank = 0;

 /* Begin meat of the algorithm. */
 for(i=0; i<n; i++)
   {

     /* We have to reset maximum diagonal value found so far to 0.0,
	because for column i we haven't calculated any diagonal
	elements so far. */
     max = 0.0;


     for (j=0; j<n; j++)
       {
	 if (orig_col[j] < 0)
	   {
	     /* First time through we will need to determine which is first
		pivot by comparing the sqrts of all the diagonals. */
	     sum = vars[j];

	     if (sum > max)
	       {
		 max = sum;
		 pivot[i] = j;
	       }
	   }
       }


     /* Set diagonal element appropriately.  If diagonal element is
	less than or equal to tolerance, then we can exit the
	function, because there are no more columns that meet the
	criterion. */
     if (max > tol)
       {
	 L[pivot[i]][pivot[i]] = max;
	 orig_col[pivot[i]] = i;
	 *rank = *rank + 1;
       }
     else
       {
	 /* The last pivot was not large enough to meet tol, so it
	    won't be used.  So, set pivot[i] back to initial value */
	 pivot[i] = -1;
	 free_ivector(orig_col, 0, n-1);
	 free_dvector(vars, 0, n-1);
	 return;
       }


     /* We found the pivot and its value.  Now we need to find the
	entries that make up the rest of the column. */
     if (i < (n-1))
       {

	 for(j=0; j<n; j++)
	   {
	     if ((orig_col[j] < 0) && (j != pivot[i]))
	       {
		 sum = 0.0;

		 for (k=0; k<i; k++)
		   sum = sum + (L[pivot[i]][pivot[k]]*L[j][pivot[k]]);
	
		 L[j][pivot[i]] = (a[j][pivot[i]] - sum)/L[pivot[i]][pivot[i]];
		     
	       }
	   }


	 /* Re-calculate diagonals based on new pivot value. */
	 for (j=0; j<n; j++)
	   {
	     if (orig_col[j] < 0)
	       {
		 tmp = (vars[j]*vars[j]) - (L[j][pivot[i]]*L[j][pivot[i]]);
		 
		 if (tmp < 0)
		   vars[j] = 0.0;
		 else
		   vars[j] = sqrt(tmp);
	       }
	   }

       }


   }


 free_ivector(orig_col, 0, n-1);
 free_dvector(vars, 0, n-1);
 return;
}

/* This function uses the Cholesky decompostion C of the matrix A (as
   given by chol_pivot) to calculate the inverse of the matrix A.  It
   also needs as input the pivot vector from chol_pivot.  If A is
   semi-positive definite, then we can subset the decomposition C and
   the pivot vector appropriately and use this function to get the
   inverse of that partition of A. */
 void inverse_using_chol_pivot(double **chol_A, double **A_inv, int
			       *pivot, int n)
{

  int i, j;
  double *ident_mat_col, *b, **B, **rearr_chol_A, **rearr_A_inv;

  /* Memory allocation here */
  ident_mat_col = dvector(0, n-1);
  b = dvector(0, n-1);
  B = darray2(n, n);
  rearr_chol_A = darray2(n, n);
  rearr_A_inv = darray2(n, n);

  ident_mat_col[0]=1.0;
  for (i=1; i<n; i++)
    ident_mat_col[i]=0.0;

  /* Permute the chol_A matrix according to the values of the pivot
     matrix (both rows and columns). */
  for (i=0; i<n; i++)
    for (j=0; j<n; j++)
      rearr_chol_A[j][i] = chol_A[pivot[j]][pivot[i]];


  for (i=0; i<n; i++)
    {

      if (i > 0)
	{
	  ident_mat_col[i]=1.0;
	  ident_mat_col[i-1]=0.0;
	}

      lower_bksub(rearr_chol_A, ident_mat_col, b, n);
      for (j=0; j<n; j++)
	B[j][i] = b[j];
    }


  RtR(B, rearr_A_inv, n, n);

  /* Permute the rearr_A_inv matrix according to the values of the
     pivot matrix (both rows and columns) to get the true A_inv */
  for (i=0; i<n; i++)
    for (j=0; j<n; j++)
      A_inv[pivot[j]][pivot[i]] = rearr_A_inv[j][i];


  free_dvector(ident_mat_col, 0, n-1);
  free_dvector(b, 0, n-1);
  free_darray2(B, n, n);
  free_darray2(rearr_chol_A, n, n);
  free_darray2(rearr_A_inv, n, n);

  return;
}



/* THIS FUNCTION GENERATES X2 GIVEN X1, WHERE X IS A VECTOR WITH A
   MULTIVARIATE NORMAL DISTRIBUTION WITH MEAN MU AND COVARIANCE MATRIX
   SIGMA.  YOU MUST FEED THIS FUNCTION PARTITIONS OF THE SIGMA MATRIX
   (SIGMA21 AND SIGMA22), THE INVERSE OF THE SIGMA11 PARTITION OF THE
   SIGMA MATRIX, THE PARTITIONS MU1 AND MU2 OF THE MEAN VECTOR
   MU, THE LENGTHS OF X2 (N2) AND X1 (N1), AND THE FACTOR TO MULTIPLY
   SIGMA PARTITIONS WITH IF THE SIGMA IS REALLY A CORRELATION, NOT A
   COVARIANCE, MATRIX.  IF THE SIGMA REALLY IS A COVARIANCE MATRIX,
   JUST MAKE THE VARIABLE "MULT_IF_CORR" EQUAL TO 1.0 */
void gen_cond_mv_normal(double *x2, double *x1, double *mu1, double
			*mu2, double **inv_sigma11, double **sigma21,
			double **sigma22, double chol_tol, double
			mult_if_corr, int n1, int n2)
{
  int i, j, rank_cov, *pivot;
  double **inv_sigma11_times_sigma12, **cov_mat, **subtract_term,
    **chol_cov_mat;
  double *x1_minus_mu1, *add_term, *mean_vec;


  inv_sigma11_times_sigma12 = darray2(n1, n2);
  cov_mat = darray2(n2, n2);
  subtract_term = darray2(n2, n2);
  chol_cov_mat = darray2(n2, n2);
  x1_minus_mu1 = dvector(0, n1-1);
  add_term = dvector(0, n2-1);
  mean_vec = dvector(0, n2-1);
  pivot = ivector(0, n2-1);


  /* FIND THE VARIANCE: SIGMA22 - (SIGMA21 %*% INV_SIGMA11 %*%
     SIGMA12) */

  /* First, find the inv_sigma11 %*% sigma12 part. */
  ARt(inv_sigma11, sigma21, inv_sigma11_times_sigma12, n1, n2);

  /* Multiply by sigma21 to get the product: sigma21 %*% inv_sigma11
     %*% sigma12. */
  CD_new(sigma21, n2, n1, inv_sigma11_times_sigma12, n1, n2,
     subtract_term);

  /* Subtract this last product from sigma22 to get variance we
     need. */
  for (i=0; i<n2; i++)
    for (j=0; j<n2; j++)
      cov_mat[i][j] = sigma22[i][j] - subtract_term[i][j];

  /* If mult_if_corr is not equal to 1.0, then these sigma matrices
     that we've been working with are really correlation matrices, and
     we need to take the multiplier into account.  The factor cancels
     out in the multiplication (inv_sigma11 %*% sigma12), so there's
     only one factor left.  The factor will not affect the calculation
     of the mean, since that only involves (inv_sigma11 %*%
     sigma12). */
  if (mult_if_corr != 1.0)
    {
      for (i=0; i<n2; i++)
	for (j=0; j<n2; j++)
	  cov_mat[i][j] = mult_if_corr * cov_mat[i][j];
    }


  /* Take cholesky of covariance, since we will need it later to do
     the random generation. */
  chol_pivot(cov_mat, chol_cov_mat, pivot, &rank_cov, chol_tol, n2);


  /* FIND THE MEAN: mu2 + (sigma21 %*% inv_sigma11 %*% (x1 - mu1)) */

  /* Find the difference x1 - mu1 */
  for (i=0; i<n1; i++)
    x1_minus_mu1[i] = x1[i] - mu1[i];

  /* Note that (sigma21 %*% inv_sigma11) equals
     t(inv_sigma11_times_sigma12). So we can use
     inv_sigma11_times_sigma12 with the Rtx function. */
  Rtx(inv_sigma11_times_sigma12, x1_minus_mu1, add_term, n2, n1);

  /* Add this last result to mu2 to get the mean. */
  for (i=0; i<n2; i++)
    mean_vec[i] = mu2[i] + add_term[i];
  
  //dwritevec("tmp.out",mean_vec,n2);

  /* Generate from the multivariate normal */
  rmvnormd_pivot(mean_vec, chol_cov_mat, n2, x2);


  free_darray2(inv_sigma11_times_sigma12, n1, n2);
  free_darray2(cov_mat, n2, n2);
  free_darray2(subtract_term, n2, n2);
  free_darray2(chol_cov_mat, n2, n2);
  free_dvector(x1_minus_mu1, 0, n1-1);
  free_dvector(add_term, 0, n2-1);
  free_dvector(mean_vec, 0, n2-1);
  free_ivector(pivot, 0, n2-1);

  return;
}


/* THIS FUNCTION COMPUTES THE CONDITIONAL MEAN AND CHOLESKY DECOMPOSITION,
 YOU MUST FEED THIS FUNCTION PARTITIONS OF THE SIGMA MATRIX
 (SIGMA21 AND SIGMA22), THE INVERSE OF THE SIGMA11 PARTITION OF THE
 SIGMA MATRIX, THE PARTITIONS MU1 AND MU2 OF THE MEAN VECTOR
 MU, THE LENGTHS OF X2 (N2) AND X1 (N1), AND THE FACTOR TO MULTIPLY
 SIGMA PARTITIONS WITH IF THE SIGMA IS REALLY A CORRELATION, NOT A
 COVARIANCE, MATRIX.  IF THE SIGMA REALLY IS A COVARIANCE MATRIX,
 JUST MAKE THE VARIABLE "MULT_IF_CORR" EQUAL TO 1.0 */
void compute_cond_parameters(double *x1, double *mu1, double **inv_sigma11, 
							 double **sigma21, double **sigma22, double chol_tol, 
							 double	mult_if_corr, int n1, int n2, double **mean_vec, double ***cov_mat,
							 double ***chol_cov_mat)
{
	int i, j, rank_cov, *pivot;
	double **inv_sigma11_times_sigma12, **subtract_term;
	double *x1_minus_mu1, *add_term;
	
	
	inv_sigma11_times_sigma12 = darray2(n1, n2);
	subtract_term = darray2(n2, n2);
	x1_minus_mu1 = dvector(0, n1-1);
	add_term = dvector(0, n2-1);
	pivot = ivector(0, n2-1);
	
	
	/* FIND THE VARIANCE: SIGMA22 - (SIGMA21 %*% INV_SIGMA11 %*%
     SIGMA12) */
	
	/* First, find the inv_sigma11 %*% sigma12 part. */
	ARt(inv_sigma11, sigma21, inv_sigma11_times_sigma12, n1, n2);
	
	/* Multiply by sigma21 to get the product: sigma21 %*% inv_sigma11
     %*% sigma12. */
	CD_new(sigma21, n2, n1, inv_sigma11_times_sigma12, n1, n2,
		   subtract_term);
	
	/* Subtract this last product from sigma22 to get variance we
     need. */
	for (i=0; i<n2; i++)
		for (j=0; j<n2; j++)
			(*cov_mat)[i][j] = sigma22[i][j] - subtract_term[i][j];
	
	/* If mult_if_corr is not equal to 1.0, then these sigma matrices
     that we've been working with are really correlation matrices, and
     we need to take the multiplier into account.  The factor cancels
     out in the multiplication (inv_sigma11 %*% sigma12), so there's
     only one factor left.  The factor will not affect the calculation
     of the mean, since that only involves (inv_sigma11 %*%
     sigma12). */
	if (mult_if_corr != 1.0)
    {
		for (i=0; i<n2; i++)
			for (j=0; j<n2; j++)
				(*cov_mat)[i][j] = mult_if_corr * (*cov_mat)[i][j];
    }
	
	
	/* Take cholesky of covariance, since we will need it later to do
     the random generation. */
	//Rprintf("Computing the cholesky decomposition of the covariance matrix\n");
	chol_pivot(*cov_mat, *chol_cov_mat, pivot, &rank_cov, chol_tol, n2);
	
	
	/* FIND THE MEAN: mu2 + (sigma21 %*% inv_sigma11 %*% (x1 - mu1)) */
	
	/* Find the difference x1 - mu1 */
	for (i=0; i<n1; i++)
		x1_minus_mu1[i] = x1[i] - mu1[i];
	
	/* Note that (sigma21 %*% inv_sigma11) equals
     t(inv_sigma11_times_sigma12). So we can use
     inv_sigma11_times_sigma12 with the Rtx function. */
	Rtx(inv_sigma11_times_sigma12, x1_minus_mu1, add_term, n2, n1);
	
	/* Add this last result to mu2 to get the mean. */
	//Rprintf("Computing the conditional mean\n");
	for (i=0; i<n2; i++){
		(*mean_vec)[i] = (*mean_vec)[i] + add_term[i];
		//Rprintf("mean_vec[%d]",i);
		//Rprintf("is:%f\n",(*mean_vec)[i]);
	}
	//dwritevec("tmp.out",mean_vec,n2);
	
	free_darray2(inv_sigma11_times_sigma12, n1, n2);
	free_darray2(subtract_term, n2, n2);
	free_dvector(x1_minus_mu1, 0, n1-1);
	free_dvector(add_term, 0, n2-1);
	free_ivector(pivot, 0, n2-1);
	
	return;
}


void rmvnormd_pivot(double *mean, double **L, int p, double *result)
/**************************************************************
random realization from a p-variate N(mean,Sigma) distribution
with Sigma = CC'.  Choleski decomp of Sigma is required.
  mean[p] mean of the nomral dist
  C[p][p] Choleski decomp of the covar matrix (does not have to
          be strictly lower, can be from chol_pivot routine.)
  result[p] The p-variate realization.
**************************************************************/
{
	GetRNGstate();
	 int i, j, k;
     double *z;
     z = dvector(0,p-1);

     
     for(k=0;k<p;k++) z[k] = rnorm(0.0,1.0);
                                                                               
     /* set result = Lz + mean*/
     for(i=0;i<p;i++)
       {
	 result[i] = 0.0;
	 for(j=0;j<p;j++)
	   result[i] += L[i][j]*z[j];
       }


     for(i=0;i<p;i++) result[i] += mean[i];
     

     free_dvector(z,0,p-1);
	PutRNGstate();
     return;
}



/* Since this was taking a lot of time with my old code in C, I'm
   making use of LAPACK stuff. This just multiplies two matrices C and
   D.  */
void CD_new(double **A, int nrowsA, int ncolsA, double **B, int
	    nrowsB, int ncolsB, double **C)
{
  char trans='N';
  double alpha=1.0, beta=0.0;

  /* Since Fortran is doing everything col by col (i.e. in
     transposes), we actaully want to ask it for t(C) - so that it
     will actually return C to us.  That means we have to multiply
     t(B) %*% t(A), which just means we pass B and A in as normal, and
     Fortran will read them as the transposes. */
  dgemm_(&trans, &trans, &ncolsB, &nrowsA, &nrowsB, &alpha, B[0],
	 &ncolsB, A[0], &ncolsA, &beta, C[0], &ncolsB);

  return;
}



void Rtx(double **R, double *x, double *y, int p, int n)
     /* Writes into y, y=transpose(R)%*%x , where R is n x p matrix, x
        is n vector - JLS 10MAR1999 */
{ 
  int i, j;

  for (i=0; i<p; i++)
    { 
      /* Initialize each element of y before getting its answer. */
      y[i] = 0.0;

      for (j=0; j<n; j++)
        y[i] = y[i] + (R[j][i]*x[j]);
    }

  return;

}


/* *************************************************
Do the matrix multiplication B=AtA.
 ************************************************* */
void RtR(double **A, double **B, int n, int p)
{
  int 
    i,j,k;
  
  for(i=0;i<p;i++)
    for(j=0;j<p;j++)
      for(k=0,B[i][j]=0.0;k<n;k++)
        B[i][j] += A[k][i]*A[k][j];
}



/* Using a matrix that is lower triangular, do back substitution to
   solve Lx=b, where L is nxn (and of course lower triangular). */
void lower_bksub(double **L, double *b, double *x, int n)
{

  double sum=0.0;
  int row, col;

  for (row = 0; row < n; row++)
    {
      sum=0.0;

      for (col = 0; col < row; col++)
        sum = sum + (L[row][col]*x[col]);
        
      x[row] = (b[row] - sum) / L[row][row];
    }

  return;
}



/* compute A*Rt, A is p x p matrix, R is n x p matrix */
void ARt(double **A, double **R, double **S, int p, int n)
{
  int 
    i,j,k;

  for(i=0;i<p;i++)
    for(j=0;j<n;j++){
      for(k=0,S[i][j]=0; k<p; k++)
        S[i][j] += A[i][k]*R[j][k];
    }
}
