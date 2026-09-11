/*************************************************
 *   Copyright Nov 12 2003                       *
 *  This file was created and copyrighted by the *
 *  National Institute of Statistical Sciences   *
 *************************************************/

void chol_pivot(double **, double **, int *, int *, double, int);
void inverse_using_chol_pivot(double **, double **, int *, int);
void gen_cond_mv_normal(double *, double *, double *, double *, double
			**, double **, double **, double, double, int,
			int);

void rmvnormd_pivot(double *, double **, int, double *);
void compute_cond_parameters(double *x1, double *mu1, double **inv_sigma11, 
							 double **sigma21, double **sigma22, double chol_tol, 
							 double	mult_if_corr, int n1, int n2, double **mean_vec, double ***cov_mat, 
							 double ***chol_cov_mat);
void lower_bksub(double **, double *, double *, int);
void Rtx(double **, double *, double *, int, int);
void RtR(double **, double **, int, int);
void ARt(double **, double **, double **, int, int);
void CD_new(double **, int, int, double **, int, int, double **);
